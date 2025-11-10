"""README 中介绍的 CoE（经验链）基础操作。"""

from __future__ import annotations

import math

import pandas as pd

from fama.data.factor_space import FactorSet


def build_initial_coe(factors: "FactorSet", scores: "pd.Series" | None = None) -> list[list[str]]:
    """根据 CSS 选出的因子创建 CoE 初始链。

    Args:
        factors: CSS 阶段汇总得到的 FactorSet。
        scores: README 中提到的评分序列，可为空。

    Returns:
        依得分排序的表达式链列表。
    """

    if not factors.factors:
        return []

    expressions = [factor.expression for factor in factors.factors]
    factor_names = [factor.name or f"factor_{idx}" for idx, factor in enumerate(factors.factors)]
    if scores is None or scores.empty:
        values = pd.Series(
            data=[1 - idx / max(1, len(expressions)) for idx in range(len(expressions))],
            index=factor_names,
        )
    else:
        values = scores.reindex(factor_names).fillna(scores.mean())

    ranked = sorted(zip(expressions, values.tolist()), key=lambda pair: pair[1], reverse=True)
    chunk_size = max(1, math.ceil(len(ranked) / 3))
    chains: list[list[str]] = []
    for start in range(0, len(ranked), chunk_size):
        chunk = [expr for expr, _ in ranked[start : start + chunk_size]]
        if chunk:
            chains.append(chunk)
    return chains


def match_coe(sample_expr: str, coe_chains: list[list[str]]) -> list[str]:
    """将 CSS 样本表达式匹配到最合适的经验链。"""

    if not coe_chains:
        return []

    sample_tokens = _tokenize(sample_expr)
    best_chain = coe_chains[0]
    best_score = -1.0
    for chain in coe_chains:
        chain_tokens = set().union(*(_tokenize(expr) for expr in chain))
        if not chain_tokens:
            continue
        intersection = len(sample_tokens & chain_tokens)
        union = len(sample_tokens | chain_tokens)
        score = intersection / union if union else 0.0
        if score > best_score:
            best_score = score
            best_chain = chain
    return best_chain


def extend_or_split_coe(coe_chain: list[str], new_expr: str) -> list[list[str]]:
    """根据新表达式对经验链进行扩展或拆分。

    Args:
        coe_chain: :func:`match_coe` 返回的链。
        new_expr: LLM 生成的候选表达式。

    Returns:
        更新后的链集合，供后续提示使用。
    """

    if not coe_chain:
        return [[new_expr]]

    if new_expr in coe_chain:
        return [coe_chain]

    extended = coe_chain + [new_expr]
    max_chain_len = 5
    if len(extended) <= max_chain_len:
        return [extended]

    midpoint = len(extended) // 2
    return [extended[:midpoint], extended[midpoint:]]


def _tokenize(expr: str) -> set[str]:
    cleaned = (
        expr.replace("(", " ")
        .replace(")", " ")
        .replace("-", " ")
        .replace("+", " ")
        .replace("*", " ")
        .replace("/", " ")
    )
    return {token.upper() for token in cleaned.split() if token}
