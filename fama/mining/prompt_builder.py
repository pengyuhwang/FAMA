"""README “LLM Prompting” 部分所述的提示词工具。"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from fama.factors.opcards import render_cards

OPS_REGEX = re.compile(
    r"\b(RANK|DELTA|DELAY|TS_MEAN|TS_SUM|TS_STDDEV|TS_MIN|TS_MAX|TS_PRODUCT|TS_ARGMAX|TS_ARGMIN|TS_RANK|CORREL|COVAR|Z_SCORE|SIGN|ABS|DECAY_LINEAR|SCALE|IF|GT|GE|LT|LE|EQ|REPLACE_NAN_INF)\b"
)


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

    prompt = f"""OPS-CHECKSUM: {checksum}
    你是一名量化研究员，请用中文完成任务，并严格遵守以下要求：

    [总体目标]
    1. 在现有因子库的基础上，生成 **与当前因子库整体相关性更低、但预期 RankIC 更优** 的新阿尔法因子。
    2. 新因子应在“结构和经济含义”上体现**新的风格或机制**，而不是对示例因子做轻微扰动（如只改窗口长度、简单替换 HIGH/LOW 等）或线性组合。

    [算子与表达式约束]
    - 仅可使用下述允许算子与字段构造表达式；
    - 严禁使用未在算子说明中出现的算子或函数。

    [允许算子说明]
    {cards or '(no operators detected—fallback to whitelist)'}

    [如何利用 CSS 示例]
    - 下方 CSS 示例来自 **不同簇**，它们之间两两相关性较低，是当前因子库中“跨风格、低相关”的代表性强因子。
    - 你可以从中学习：
      - 使用了哪些算子（如秩、相关、滑窗统计、量价组合等）；
      - 它们是如何刻画不同市场行为/风格的。
    - 但必须避免：
      - 直接复刻这些表达式；
      - 只对其做参数级的小改动；
      - 简单线性组合多个 CSS 示例因子。
    - 目标：在这些“低相关强风格”的启发下，构造**在风格上有差异、与当前因子库整体相关性更低**的新因子。

    # CSS 示例（跨簇低相关的代表性因子）
    {css_block}

    [如何利用 CoE 经验链]
    - 下方 CoE 为同一簇内按 **RankIC 由弱到强排序** 的因子演化链：
      - 链头部：相对较弱/早期版本；
      - 链尾部：同簇内 RankIC 更强、结构更成熟的版本。
    - 你需要：
      - 观察链中因子是如何从“简单”演化到“复杂”以提升 RankIC 的（例如加入条件过滤、引入波动率或成交量、增加滞后维度、使用更加稳健的归一化等）；
      - 在此基础上进行进一步**结构创新**，提出有望在该簇中取得更高 RankIC 的新变体。
    - 同时要注意：
      - 新因子不能只是链尾因子的微小改写；
      - 在全局上也应与当前因子库保持较低相关性。

    # Chain-of-Experience（同簇内按 RankIC 由弱到强的经验链）
    {coe_block}

    [输出格式与字段约束]
    - 仅允许使用以下字段（必须大写）：{fields_block}
    - 你必须输出一个 **JSON 数组**，共 {max_new} 个元素；
    - 每个元素必须严格为以下格式（字段名固定，均为字符串）：
    {{
      "expression": "<合法 DSL 表达式，满足算子/字段/长度/嵌套约束>",
      "explanation": "<一句中文经济学解释，仅一句，简洁说明该因子的经济含义>"
    }}
    - JSON 外 **不得包含任何额外文本、注释、编号、说明或反引号**；
    - 不要输出自然语言解释、指南或多余字段，只输出 JSON 数组本身。

    请基于以上 CSS 示例与 CoE 经验链信息，在控制与现有因子库相关性较低的前提下，发挥你对金融量化因子的理解，尽可能提升新因子的预期 RankIC，直接输出符合要求的 JSON。""".strip()

    return prompt


def parse_llm_output(text: str) -> list[dict]:
    """将 LLM 的输出解析成包含 expression / explanation 的列表。"""

    import json

    cleaned = text.strip()
    if not cleaned:
        return []

    try:
        data = json.loads(cleaned)
        parsed: list[dict] = []
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                expr = item.get("expression")
                expl = item.get("explanation")
                if not expr:
                    continue
                parsed.append({"expression": str(expr).strip(), "explanation": expl.strip() if isinstance(expl, str) else None})
            if parsed:
                return parsed
    except Exception:
        pass

    # 兼容旧格式：每行一个表达式
    parsed: list[dict] = []
    for line in cleaned.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if ":" in candidate:
            _, candidate = candidate.split(":", 1)
        candidate = candidate.strip().strip("`")
        if candidate:
            parsed.append({"expression": candidate, "explanation": None})
    return parsed
