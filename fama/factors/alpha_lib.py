"""README “Seed Alpha Library” 章节中提到的符号化 Alpha 库。"""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


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
    """返回 CSS 阶段所需的初始 Alpha 表达式。

    Returns:
        可作为 CSS 初始化输入的一组表达式字符串。
    """

    return [
        "RANK(CLOSE - OPEN)",
        "DELTA(CLOSE, 3)",
        "TS_MEAN(RET, 5)",
        "RANK(TS_STDDEV(RET, 10))",
        "RANK(CORREL(CLOSE, VOLUME, 5))",
        "RANK(VWAP - CLOSE)",
        "DELTA(RANK(CLOSE), 5)",
        "RANK(Z_SCORE(CLOSE))",
    ]


def validate_alpha_syntax(expr: str) -> bool:
    """在表达式进入 CSS 前做轻量语法校验。

    Args:
        expr: 由 LLM 产生的 Alpha 字符串。

    Returns:
        布尔值，表示表达式是否只包含被支持的语法节点。
    """

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.Call, ast.Num, ast.Constant)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCTIONS and node.id not in _ALLOWED_VARIABLES:
                return False
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
            continue
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


def _ts_mean(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=1).mean()
    )


def _ts_stddev(series: "pd.Series", window: int) -> "pd.Series":
    return series.groupby(level=1, group_keys=False).apply(
        lambda s: s.rolling(window, min_periods=2).std()
    )


def _correl(x: "pd.Series", y: "pd.Series", window: int) -> "pd.Series":
    joined = x.to_frame("x").join(y.to_frame("y"))
    grouped = joined.groupby(level=1, group_keys=False)
    return grouped.apply(lambda df: df["x"].rolling(window, min_periods=2).corr(df["y"]))


def _z_score(series: "pd.Series") -> "pd.Series":
    mean = series.groupby(level=0).transform("mean")
    std = series.groupby(level=0).transform("std").replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _sign(series: "pd.Series") -> "pd.Series":
    return np.sign(series)


def _abs(series: "pd.Series") -> "pd.Series":
    return series.abs()


__all__ = [
    "parse_symbolic_expression",
    "list_seed_alphas",
    "validate_alpha_syntax",
    "evaluate_expression",
]


_ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "RANK": _rank,
    "DELTA": _delta,
    "TS_MEAN": _ts_mean,
    "TS_STDDEV": _ts_stddev,
    "CORREL": _correl,
    "Z_SCORE": _z_score,
    "SIGN": _sign,
    "ABS": _abs,
}

_ALLOWED_VARIABLES = {
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
    "RET",
    "VWAP",
}
