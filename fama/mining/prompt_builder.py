"""README “LLM Prompting” 部分所述的提示词工具。"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from fama.factors.opcards import render_cards

OPS_REGEX = re.compile(r"\b(RANK|DELTA|TS_MEAN|TS_STDDEV|CORREL|Z_SCORE|SIGN|ABS)\b")


def _extract_ops(exprs: Iterable[str]) -> list[str]:
    ops: set[str] = set()
    for expr in exprs:
        for match in OPS_REGEX.finditer(str(expr)):
            ops.add(match.group(1))
    return sorted(ops)


def _checksum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def build_prompt(
    css_examples: list[str],
    coe_path: list[str],
    constraints: dict,
    *,
    available_fields: list[str],
) -> str:
    """拼装 LLM 挖掘阶段所需的结构化提示词。"""

    whitelist = set(constraints.get("operator_whitelist", []))
    context_ops = _extract_ops(list(css_examples) + list(coe_path))
    ops = sorted((set(context_ops) & whitelist) if whitelist else set(context_ops))
    cards = render_cards(ops)
    checksum = _checksum(cards)
    css_block = "\n".join(f"- {expr}" for expr in css_examples) or "- (none)"
    coe_block = "\n".join(f"- {entry}" for entry in coe_path) or "- (none)"
    fields_block = ", ".join(available_fields)
    max_new = constraints.get("max_new_factors", 5)

    prompt = f"""
OPS-CHECKSUM: {checksum}
仅可使用 Operator Specification 中列出的算子，且窗口参数需为正整数。禁止输出任何未列出的算子。

{cards or '(no operators detected—fallback to whitelist)'}

# CSS exemplars
{css_block}

# Chain-of-Experience (ICL hints)
{coe_block}

# Guardrails
- 输出数量：{max_new} 条，一行一个表达式，纯文本、勿加反引号。
- 仅允许使用以下字段（大写）：{fields_block}
- 严格使用上述算子；未知算子视为违规。
- 表达式长度 <= 80 字符，嵌套层级 <= 2。
""".strip()

    return prompt


def parse_llm_output(text: str) -> list[str]:
    """将 LLM 的原始输出解析成候选因子表达式。"""

    expressions: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if ":" in candidate:
            _, candidate = candidate.split(":", 1)
        candidate = candidate.strip().strip("`")
        if candidate:
            expressions.append(candidate)
    return expressions
