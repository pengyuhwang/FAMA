"""README “LLM Prompting” 部分所述的提示词工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_TEMPLATE = """# FAMA Alpha Prompt
CSS exemplars:
{css_examples}
Chain-of-Experience:
{coe_path}
Available fields:
{available_fields}
Constraints:
{constraints}
"""


def build_prompt(
    css_examples: list[str],
    coe_path: list[str],
    constraints: dict,
    available_fields: list[str] | None = None,
) -> str:
    """拼装 LLM 挖掘阶段所需的结构化提示词。

    Args:
        css_examples: CSS 挑选出的示例。
        coe_path: 当前经验链路径。
        constraints: 配置表中定义的各种约束。
        available_fields: 允许 LLM 使用的字段列表。

    Returns:
        可直接发送给 LLM 的提示词文本。
    """

    template_path = constraints.get("instructions_path")
    template = DEFAULT_TEMPLATE
    if template_path:
        path = Path(template_path)
        if path.exists():
            template = path.read_text(encoding="utf-8")
    css_block = "\n".join(f"- {expr}" for expr in css_examples) if css_examples else "- N/A"
    coe_block = " -> ".join(coe_path) if coe_path else "N/A"
    fields_block = ", ".join(available_fields) if available_fields else "N/A"
    constraint_payload = {k: v for k, v in constraints.items() if k != "instructions_path"}
    constraints_block = json.dumps(constraint_payload, ensure_ascii=False, indent=2)
    return template.format(
        css_examples=css_block,
        coe_path=coe_block,
        available_fields=fields_block,
        constraints=constraints_block,
    )


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
