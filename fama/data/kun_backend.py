"""KunQuant 后端：解析符号化表达式并批量计算因子。"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, ConstantOp, Input, Output, Rank
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops.CompOp import (
    DecayLinear,
    TsArgMax,
    TsArgMin,
    TsRank,
    WindowedAvg,
    WindowedCorrelation,
    WindowedCovariance,
    WindowedMax,
    WindowedMin,
    WindowedProduct,
    WindowedStddev,
    WindowedSum,
)
from KunQuant.ops.ElewiseOp import (
    Abs as OpAbs,
    AddConst,
    And as OpAnd,
    Div as OpDiv,
    Equals as OpEquals,
    Exp as OpExp,
    GreaterEqual as OpGreaterEqual,
    GreaterThan as OpGreaterThan,
    LessEqual as OpLessEqual,
    LessThan as OpLessThan,
    Log as OpLog,
    Max as OpMax,
    Min as OpMin,
    Mul as OpMul,
    Not as OpNot,
    Or as OpOr,
    Select,
    SetInfOrNanToValue,
    Sign as OpSign,
    Sub as OpSub,
)
from KunQuant.ops.MiscOp import BackRef
from KunQuant.Op import Scale
from KunQuant.runner import KunRunner as kr


FIELD_MAP: Dict[str, str] = {
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
    "AMOUNT": "amount",
}

DSL_FUNCTIONS = {}


def _register_ops():
    def _ensure_expression(value):
        if isinstance(value, (int, float)):
            return ConstantOp(float(value))
        return value

    def _mask_to_float(mask):
        return Select(mask, ConstantOp(1.0), ConstantOp(0.0))

    def _to_bool_float(value):
        expr = _ensure_expression(value)
        return _mask_to_float(OpGreaterThan(expr, ConstantOp(0.5)))

    def _if_then_else(cond, a, b):
        cond_float = _to_bool_float(cond)
        a_expr = _ensure_expression(a)
        b_expr = _ensure_expression(b)
        return cond_float * a_expr + (ConstantOp(1.0) - cond_float) * b_expr

    def _logical_and(x, y):
        return _to_bool_float(x) * _to_bool_float(y)

    def _logical_or(x, y):
        bx = _to_bool_float(x)
        by = _to_bool_float(y)
        return bx + by - bx * by

    def _logical_not(x):
        return ConstantOp(1.0) - _to_bool_float(x)

    DSL_FUNCTIONS.update(
        {
            "RANK": lambda x: AddConst(
                OpMul(Rank(_ensure_expression(x)), ConstantOp(9.0)),
                1.0,
            ),
            "DELTA": lambda x, n: OpSub(x, BackRef(x, _to_int(n))),
            "TS_MEAN": lambda x, n: WindowedAvg(x, _to_int(n)),
            "TS_STDDEV": lambda x, n: WindowedStddev(x, _to_int(n)),
            "CORREL": lambda x, y, n: WindowedCorrelation(x, _to_int(n), y),
            "SIGN": lambda x: OpSign(x),
            "ABS": lambda x: OpAbs(x),
            "DELAY": lambda x, n: BackRef(x, _to_int(n)),
            "TS_SUM": lambda x, n: WindowedSum(x, _to_int(n)),
            "TS_MIN": lambda x, n: WindowedMin(x, _to_int(n)),
            "TS_MAX": lambda x, n: WindowedMax(x, _to_int(n)),
            "TS_PRODUCT": lambda x, n: WindowedProduct(x, _to_int(n)),
            "TS_ARGMAX": lambda x, n: TsArgMax(x, _to_int(n)),
            "TS_ARGMIN": lambda x, n: TsArgMin(x, _to_int(n)),
            "TS_RANK": lambda x, n: TsRank(x, _to_int(n)),
            "DECAY_LINEAR": lambda x, n: DecayLinear(x, _to_int(n)),
            "SCALE": lambda x: Scale(x),
            "IF": _if_then_else,
            "AND": _logical_and,
            "OR": _logical_or,
            "NOT": _logical_not,
            "GT": lambda x, y: _mask_to_float(
                OpGreaterThan(_ensure_expression(x), _ensure_expression(y))
            ),
            "GE": lambda x, y: _mask_to_float(
                OpGreaterEqual(_ensure_expression(x), _ensure_expression(y))
            ),
            "LT": lambda x, y: _mask_to_float(
                OpLessThan(_ensure_expression(x), _ensure_expression(y))
            ),
            "LE": lambda x, y: _mask_to_float(
                OpLessEqual(_ensure_expression(x), _ensure_expression(y))
            ),
            "EQ": lambda x, y: _mask_to_float(
                OpEquals(_ensure_expression(x), _ensure_expression(y))
            ),
            "MAX": lambda x, y: OpMax(x, y),
            "MIN": lambda x, y: OpMin(x, y),
            "LOG": lambda x: OpLog(x),
            "EXP": lambda x: OpExp(x),
            "REPLACE_NAN_INF": lambda x, value=0.0: SetInfOrNanToValue(x, _to_float(value)),
            "COVAR": lambda x, y, n: WindowedCovariance(x, _to_int(n), y),
        }
    )


try:  # Sign op在 ElewiseOp 中定义
    from KunQuant.ops.ElewiseOp import Sign as OpSign
except ImportError:  # pragma: no cover
    OpSign = None

_register_ops()


def compute_factor_values_kunquant(
    market_data: pd.DataFrame,
    expr_list: List[str],
    *,
    threads: int = 4,
    layout: str = "TS",
) -> Tuple[pd.DataFrame, List[str]]:
    if not expr_list:
        empty = pd.DataFrame(index=market_data.index)
        empty.index.names = ["date", "symbol"]
        return empty, []

    inputs_np, dates, symbols = _build_ts_inputs(market_data)
    builder = Builder()
    compiled_exprs: list[str] = []
    fallback_exprs: list[str] = []

    with builder:
        inp = {name: Input(name) for name in inputs_np.keys()}
        env = _build_env(inp)
        counter = 0
        for expr in expr_list:
            try:
                ir = _compile_expression(expr, env)
            except Exception:
                fallback_exprs.append(expr)
                continue
            counter += 1
            compiled_exprs.append(expr)
            Output(ir, f"f_{counter}")

    if not compiled_exprs:
        empty = pd.DataFrame(index=market_data.index)
        empty.index.names = ["date", "symbol"]
        return empty, expr_list.copy()

    func = Function(builder.ops)
    lib = cfake.compileit(
        [("fama_graph", func, KunCompilerConfig(input_layout=layout, output_layout=layout))],
        "fama_graph_lib",
        cfake.CppCompilerConfig(),
    )
    module = lib.getModule("fama_graph")
    executor = kr.createMultiThreadExecutor(max(1, int(threads)))
    first = next(iter(inputs_np.values()))
    length = first.shape[0]
    num_stocks = first.shape[1]
    out = kr.runGraph(
        executor,
        module,
        inputs_np,
        0,
        length,
        {},
        True,
        num_stocks=num_stocks,
    )

    stacked: Dict[str, pd.Series] = {}
    num_dates = len(dates)
    num_symbols = len(symbols)
    for idx, expr in enumerate(compiled_exprs, 1):
        raw = np.asarray(out[f"f_{idx}"])
        if raw.shape == (num_symbols, num_dates):
            matrix = raw.T
        elif raw.shape == (num_dates, num_symbols):
            matrix = raw
        else:  # fallback with explicit reshape if total size matches
            if raw.size != num_dates * num_symbols:
                raise ValueError(
                    f"Unexpected KunQuant output shape {raw.shape}; expected "
                    f"({num_dates}, {num_symbols}) or ({num_symbols}, {num_dates})."
                )
            matrix = raw.reshape(num_dates, num_symbols)
        df = pd.DataFrame(matrix, index=dates, columns=symbols)
        try:
            stacked_series = df.stack(future_stack=True)
        except TypeError:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The previous implementation of stack is deprecated",
                    category=FutureWarning,
                )
                stacked_series = df.stack(dropna=False)
        stacked[expr] = stacked_series

    result = pd.concat(stacked, axis=1)
    result.index.names = ["date", "symbol"]
    return result.sort_index(), fallback_exprs


def _compile_expression(expr: str, env: Dict[str, Input]):
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, env)


def _eval_ast(node: ast.AST, env: Dict[str, Input]):
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, env)
        right = _eval_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return OpMul(left, right)
        if isinstance(node.op, ast.Div):
            return _safe_div(left, right)
        if isinstance(node.op, ast.Pow):
            return _power_expr(left, right)
        raise NotImplementedError(f"Unsupported operator {node.op}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, env)
        if isinstance(node.op, ast.USub):
            return OpSub(ConstantOp(0.0), operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise NotImplementedError(f"Unsupported unary {node.op}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id.upper()
        func = DSL_FUNCTIONS.get(func_name)
        if func is None:
            raise NotImplementedError(f"Unsupported function {func_name}")
        args = [_eval_ast(arg, env) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Name):
        key = node.id.upper()
        if key not in env:
            raise NotImplementedError(f"Unknown variable {key}")
        return env[key]
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)):
            return ConstantOp(float(value))
        raise NotImplementedError(f"Unsupported constant {value}")
    raise NotImplementedError(f"Unsupported AST node {ast.dump(node)}")


def _power_expr(base, exponent):
    base_expr = _ensure_expr(base)
    exp_value = _extract_constant(exponent)
    if exp_value is None:
        exponent_expr = _ensure_expr(exponent)
        return _signed_power(base_expr, exponent_expr)
    if abs(exp_value) < 1e-12:
        return ConstantOp(1.0)
    if abs(exp_value - 1.0) < 1e-12:
        return base_expr
    if float(exp_value).is_integer():
        n = int(round(exp_value))
        if n < 0:
            raise NotImplementedError("Negative exponents are not supported.")
        result = base_expr
        for _ in range(n - 1):
            result = OpMul(result, base_expr)
        return result
    exponent_expr = ConstantOp(float(exp_value))
    return _signed_power(base_expr, exponent_expr)


def _ensure_expr(value):
    if isinstance(value, (int, float)):
        return ConstantOp(float(value))
    return value


def _extract_constant(value):
    if isinstance(value, ConstantOp):
        return float(value.attrs.get("value", 0.0))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value):
    if isinstance(value, ConstantOp):
        return int(value.attrs.get("value", 0))
    return int(value)


def _to_float(value):
    if isinstance(value, ConstantOp):
        return float(value.attrs.get("value", 0.0))
    return float(value)


def _safe_div(numerator, denominator, eps: float = 1e-7):
    return OpDiv(_ensure_expr(numerator), _ensure_expr(denominator))


def _signed_power(base_expr, exponent_expr):
    base_expr = _ensure_expr(base_expr)
    exponent_expr = _ensure_expr(exponent_expr)
    abs_base = OpAbs(base_expr)
    safe_base = OpMax(abs_base, ConstantOp(1e-6))
    magnitude = OpExp(OpMul(exponent_expr, OpLog(safe_base)))
    if OpSign is not None:
        sign = OpSign(base_expr)
    else:
        sign = Select(
            OpGreaterThan(base_expr, ConstantOp(0.0)),
            ConstantOp(1.0),
            ConstantOp(-1.0),
        )
    return OpMul(sign, magnitude)


def _build_ts_inputs(mkt_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp], List[str]]:
    if not isinstance(mkt_df.index, pd.MultiIndex) or mkt_df.index.nlevels != 2:
        raise ValueError("market_data 必须使用 (date, symbol) MultiIndex")

    dates = sorted(mkt_df.index.get_level_values(0).unique())
    symbols = sorted(mkt_df.index.get_level_values(1).unique())
    inputs: Dict[str, np.ndarray] = {}

    for famaf, alias in FIELD_MAP.items():
        col = _find_column(mkt_df, famaf)
        if col is None:
            col = _find_column(mkt_df, alias)
        if col is None:
            continue
        slice_df = mkt_df[col].unstack(level=1)
        slice_df = slice_df.reindex(index=dates, columns=symbols)
        arr = slice_df.to_numpy(dtype=np.float32)
        inputs[alias] = np.ascontiguousarray(arr)

    missing = [field for field in FIELD_MAP.values() if field not in inputs]
    if missing:
        raise ValueError(f"KunQuant 后端缺少字段: {missing}")
    return inputs, dates, symbols


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    target = name.lower()
    for col in df.columns:
        if col.lower() == target:
            return col
    return None


def _build_env(inputs: Dict[str, Input]) -> Dict[str, Input]:
    env = {key.upper(): inputs[key] for key in FIELD_MAP.values()}
    close = env["CLOSE"]
    volume = env["VOLUME"]
    amount = env["AMOUNT"]

    # RET = close / lag(close,1) - 1
    lag_close = BackRef(close, 1)
    env["RET"] = OpSub(OpDiv(close, lag_close), ConstantOp(1.0))

    # VWAP = amount / (volume + 1e-7)
    env["VWAP"] = OpDiv(amount, AddConst(volume, 1e-7))
    return env
