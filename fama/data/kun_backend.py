"""KunQuant 后端：解析符号化表达式并批量计算因子。"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, ConstantOp, Input, Output, Rank
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops.CompOp import WindowedAvg, WindowedCorrelation, WindowedStddev
from KunQuant.ops.ElewiseOp import (Abs as OpAbs, AddConst, Div as OpDiv, Mul as OpMul,
                                    Sign as OpSign, Sub as OpSub)
from KunQuant.ops.MiscOp import BackRef
from KunQuant.runner import KunRunner as kr


try:
    from KunQuant.predefined import Alpha101 as alpha101_module
except Exception:  # pragma: no cover
    alpha101_module = None

FIELD_MAP: Dict[str, str] = {
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
    "AMOUNT": "amount",
}

DSL_FUNCTIONS = {}
_ALPHA_TOKEN = re.compile(r"^alpha\d{3}$", re.IGNORECASE)


def _register_ops():
    DSL_FUNCTIONS.update(
        {
            "RANK": lambda x: Rank(x),
            "DELTA": lambda x, n: OpSub(x, BackRef(x, int(n))),
            "TS_MEAN": lambda x, n: WindowedAvg(x, int(n)),
            "TS_STDDEV": lambda x, n: WindowedStddev(x, int(n)),
            "CORREL": lambda x, y, n: WindowedCorrelation(x, int(n), y),
            "SIGN": lambda x: OpSign(x),
            "ABS": lambda x: OpAbs(x),
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
    num_stocks = first.shape[0]
    length = first.shape[1]
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
        stacked[expr] = df.stack(dropna=False)

    result = pd.concat(stacked, axis=1)
    result.index.names = ["date", "symbol"]
    return result.sort_index(), fallback_exprs


def _compile_expression(expr: str, env: Dict[str, Input]):
    token = expr.strip().lower()
    if _ALPHA_TOKEN.fullmatch(token):
        return _compile_alpha_token(token, env)
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, env)


def _compile_alpha_token(token: str, env: Dict[str, Input]):
    if alpha101_module is None:
        raise NotImplementedError("KunQuant Alpha101 模块不可用。")
    builder_cls = getattr(alpha101_module, "AllData", None)
    if builder_cls is None:
        raise NotImplementedError("KunQuant Alpha101 缺少 AllData 定义。")
    func = getattr(alpha101_module, token, None)
    if func is None:
        raise NotImplementedError(f"未知的 Alpha101 因子: {token}")

    data = builder_cls(
        open=env["OPEN"],
        close=env.get("CLOSE"),
        high=env.get("HIGH"),
        low=env.get("LOW"),
        volume=env.get("VOLUME"),
        amount=env.get("AMOUNT"),
        vwap=env.get("VWAP"),
    )
    return func(data)


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
            return OpDiv(left, right)
        if isinstance(node.op, ast.Pow):
            return left ** right
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
        inputs[alias] = slice_df.to_numpy(dtype=np.float32).T

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
