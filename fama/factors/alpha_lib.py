"""README “Seed Alpha Library” 章节中提到的符号化 Alpha 库。"""

from __future__ import annotations

import ast
import importlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

_ALPHA_TOKEN = re.compile(r"^alpha\d{3}$", re.IGNORECASE)


def parse_symbolic_expression(expr: str) -> dict:
    """将 Alpha 风格表达式解析为结构化字典。

    Args:
        expr: README “Prompt Template & LLM Integration” 中提到的符号表达式。

    Returns:
        供后续验证使用的语法树描述。
    """

    tree = ast.parse(expr, mode="eval")
    functions = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
    )
    variables = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCTIONS
        }
    )
    return {
        "expression": expr,
        "functions": functions,
        "variables": variables,
    }


def list_seed_alphas() -> list[str]:
    """返回 CSS 阶段所需的初始 Alpha 表达式。"""

    return _default_seed_expressions().copy()


_ALPHA101_CACHE: list[str] | None = None


def list_alpha101_tokens() -> list[str]:
    """从 KunQuant 内置的 Alpha101 模块列出可用的符号名称。"""

    global _ALPHA101_CACHE
    if _ALPHA101_CACHE is not None:
        return _ALPHA101_CACHE.copy()
    try:
        module = importlib.import_module("KunQuant.predefined.Alpha101")
    except Exception:
        _ALPHA101_CACHE = []
        return []

    tokens: list[str] = []
    for name, obj in vars(module).items():
        if callable(obj) and _ALPHA_TOKEN.fullmatch(name.lower()):
            tokens.append(name.lower())
    _ALPHA101_CACHE = sorted(tokens)
    return _ALPHA101_CACHE.copy()


def validate_alpha_syntax(
    expr: str,
    allowed_variables: Iterable[str] | None = None,
    *,
    allowed_ops: Optional[Iterable[str]] = None,
) -> bool:
    """在表达式进入 CSS 前做轻量语法校验。

    Args:
        expr: 由 LLM 产生的 Alpha 字符串。

    Returns:
        布尔值，表示表达式是否只包含被支持的语法节点。
    """

    stripped = expr.strip()
    if _ALPHA_TOKEN.fullmatch(stripped):
        return True

    allowed = set(_BASE_VARIABLES)
    if allowed_variables:
        allowed.update(var.upper() for var in allowed_variables)
    allowed_ops_set = {op.upper() for op in allowed_ops} if allowed_ops else None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    called_ops = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.Call, ast.Num, ast.Constant)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_ops.add(node.func.id.upper())
            continue
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCTIONS and node.id not in allowed:
                return False
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
            continue
        return False

    if allowed_ops_set is not None:
        unknown = called_ops - allowed_ops_set
        if unknown:
            return False

    return True


def evaluate_expression(expr: str, context: dict[str, "pd.Series"]) -> "pd.Series":
    """安全地执行 Alpha101 风格的表达式。

    Args:
        expr: 引用 OHLCV 及辅助函数的符号表达式。
        context: 变量名到 pandas Series 的映射，索引为 ``(date, symbol)``。

    Returns:
        含有计算结果的 pandas Series。
    """

    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, context)


def _eval_node(node: ast.AST, context: dict[str, "pd.Series"]) -> Any:
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        return _apply_operator(node.op, left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary operator: {node.op}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not supported.")
        args = [_eval_node(arg, context) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown variable '{node.id}' in expression.")
        return context[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def _apply_operator(op: ast.operator, left: Any, right: Any) -> Any:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.Pow):
        return left ** right
    raise ValueError(f"不支持的运算符: {op}")


def _rank(series: "pd.Series") -> "pd.Series":
    return series.groupby(level=0).rank(pct=True)


def _delta(series: "pd.Series", periods: int) -> "pd.Series":
    return series.groupby(level=1).diff(periods)


def _delay(series: "pd.Series", periods: int) -> "pd.Series":
    return series.groupby(level=1).shift(periods)


def _ts_mean(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).mean()
    )


def _ts_sum(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).sum()
    )


def _ts_stddev(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=2).std()
    )


def _ts_min(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).min()
    )


def _ts_max(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).max()
    )


def _ts_product(series: "pd.Series", window: int) -> "pd.Series":
    def _product(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        return float(np.nanprod(values))

    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).apply(_product, raw=True)
    )


def _ts_argmax(series: "pd.Series", window: int) -> "pd.Series":
    def _argmax(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        return float(np.nanargmax(values))

    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).apply(_argmax, raw=True)
    )


def _ts_argmin(series: "pd.Series", window: int) -> "pd.Series":
    def _argmin(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return np.nan
        return float(np.nanargmin(values))

    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).apply(_argmin, raw=True)
    )


def _ts_rank(series: "pd.Series", window: int) -> "pd.Series":
    def _rank(values: "pd.Series") -> float:
        last = values.iloc[-1]
        if pd.isna(last):
            return np.nan
        ranks = values.rank(pct=True, method="average")
        return float(ranks.iloc[-1])

    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).apply(_rank, raw=False)
    )


def _correl(x: "pd.Series", y: "pd.Series", window: int) -> "pd.Series":
    joined = x.to_frame("x").join(y.to_frame("y"))
    grouped = joined.groupby(level=1, group_keys=False)
    return grouped.apply(lambda df: df["x"].rolling(window, min_periods=2).corr(df["y"]))


def _covar(x: "pd.Series", y: "pd.Series", window: int) -> "pd.Series":
    joined = x.to_frame("x").join(y.to_frame("y"))
    grouped = joined.groupby(level=1, group_keys=False)
    return grouped.apply(lambda df: df["x"].rolling(window, min_periods=2).cov(df["y"]))


def _z_score(series: "pd.Series") -> "pd.Series":
    mean = series.groupby(level=0).transform("mean")
    std = series.groupby(level=0).transform("std").replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _sign(series: "pd.Series") -> "pd.Series":
    return np.sign(series)


def _abs(series: "pd.Series") -> "pd.Series":
    return series.abs()


def _decay_linear(series: "pd.Series", window: int) -> "pd.Series":
    weights = np.arange(1, window + 1, dtype=float)

    def apply(values: np.ndarray) -> float:
        valid = ~np.isnan(values)
        if not valid.any():
            return 0.0
        w = weights[-len(values):][valid]
        return float(np.dot(values[valid], w) / w.sum()) if w.sum() else 0.0

    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).apply(apply, raw=True)
    )


def _scale(series: "pd.Series") -> "pd.Series":
    grouped = series.groupby(level=0)
    def normalize(s: "pd.Series") -> "pd.Series":
        denom = s.abs().sum()
        return s * 0 if denom == 0 else s / denom
    return grouped.transform(normalize)


def _broadcast_series(value: Any, index: "pd.Index") -> "pd.Series":
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _align_pair(left: Any, right: Any) -> tuple["pd.Series", "pd.Series"]:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        return left.align(right, join="outer")
    if isinstance(left, pd.Series):
        return left, _broadcast_series(right, left.index)
    if isinstance(right, pd.Series):
        return _broadcast_series(left, right.index), right
    raise ValueError("At least one operand must be a pandas Series.")


def _if(condition: "pd.Series", true_series: Any, false_series: Any) -> "pd.Series":
    cond = condition.astype(bool)
    index = cond.index
    true_aligned = _broadcast_series(true_series, index)
    false_aligned = _broadcast_series(false_series, index)
    return true_aligned.where(cond, false_aligned)


def _logical_and(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left.astype(bool) & aligned_right.astype(bool)).astype(float)


def _logical_or(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left.astype(bool) | aligned_right.astype(bool)).astype(float)


def _logical_not(series: Any) -> "pd.Series":
    series = series if isinstance(series, pd.Series) else pd.Series(series)
    return (~series.astype(bool)).astype(float)


def _gt(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left > aligned_right).astype(float)


def _ge(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left >= aligned_right).astype(float)


def _lt(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left < aligned_right).astype(float)


def _le(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left <= aligned_right).astype(float)


def _eq(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return (aligned_left == aligned_right).astype(float)


def _max_series(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return pd.concat([aligned_left, aligned_right], axis=1).max(axis=1)


def _min_series(left: Any, right: Any) -> "pd.Series":
    aligned_left, aligned_right = _align_pair(left, right)
    return pd.concat([aligned_left, aligned_right], axis=1).min(axis=1)


def _log(series: "pd.Series") -> "pd.Series":
    return np.log(series.replace(0, np.nan))


def _exp(series: "pd.Series") -> "pd.Series":
    return np.exp(series)


def _replace_nan_inf(series: "pd.Series", value: float = 0.0) -> "pd.Series":
    return series.replace([np.inf, -np.inf], np.nan).fillna(value)


__all__ = [
    "parse_symbolic_expression",
    "list_seed_alphas",
    "list_alpha101_tokens",
    "validate_alpha_syntax",
    "evaluate_expression",
]


_ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "RANK": _rank,
    "DELTA": _delta,
    "DELAY": _delay,
    "TS_MEAN": _ts_mean,
    "TS_SUM": _ts_sum,
    "TS_STDDEV": _ts_stddev,
    "TS_MIN": _ts_min,
    "TS_MAX": _ts_max,
    "TS_PRODUCT": _ts_product,
    "TS_ARGMAX": _ts_argmax,
    "TS_ARGMIN": _ts_argmin,
    "TS_RANK": _ts_rank,
    "CORREL": _correl,
    "COVAR": _covar,
    "Z_SCORE": _z_score,
    "SIGN": _sign,
    "ABS": _abs,
    "DECAY_LINEAR": _decay_linear,
    "SCALE": _scale,
    "IF": _if,
    "AND": _logical_and,
    "OR": _logical_or,
    "NOT": _logical_not,
    "GT": _gt,
    "GE": _ge,
    "LT": _lt,
    "LE": _le,
    "EQ": _eq,
    "MAX": _max_series,
    "MIN": _min_series,
    "LOG": _log,
    "EXP": _exp,
    "REPLACE_NAN_INF": _replace_nan_inf,
}

_BASE_VARIABLES = {
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
    "RET",
    "VWAP",
}


def _default_seed_expressions() -> list[str]:
    base = [
        "RANK(CLOSE - OPEN)",
        "DELTA(CLOSE, 3)",
        "TS_MEAN(RET, 5)",
        "RANK(TS_STDDEV(RET, 10))",
        "RANK(CORREL(CLOSE, VOLUME, 5))",
        "RANK(VWAP - CLOSE)",
        "DELTA(RANK(CLOSE), 5)",
        "RANK(CLOSE)",
        "RANK(CLOSE - LOW)",
        "RANK(HIGH - CLOSE)",
        "TS_STDDEV(CLOSE - OPEN, 7)",
        "TS_MEAN(RET * VOLUME, 5)",
        "RANK(CORREL(RET, VWAP, 10))",
        "RANK(TS_STDDEV(VOLUME, 15))",
        "DELTA(RANK(VOLUME), 4)",
    ]
    exprs = base.copy()
    for w in range(3, 23):
        exprs.append(f"RANK(TS_MEAN(CLOSE - OPEN, {w}))")
    for lag in range(2, 22):
        exprs.append(f"DELTA(RANK(CLOSE - LOW), {lag})")
    for w in range(5, 25):
        exprs.append(f"RANK(TS_STDDEV(VOLUME, {w}))")
    for w in range(6, 26):
        exprs.append(f"RANK(CORREL(CLOSE, VOLUME, {w}))")
    for w in range(4, 14):
        exprs.append(f"RANK(TS_MEAN(RET * VOLUME, {w}))")
    for lag in range(3, 13):
        exprs.append(f"RANK(DELTA(VWAP - CLOSE, {lag}))")
    return exprs[:101]
