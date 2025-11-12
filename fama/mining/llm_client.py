"""负责向 LLM 请求新因子的客户端适配层。"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc]

from fama.mining.prompt_builder import parse_llm_output


def request_new_factors(
    prompt: str,
    provider: str,
    model: str,
    api_key: str | None = None,
    temperature: float | None = None,
    thinking: str | None = None,
    allowed_fields: list[str] | None = None,
    logger=None,
) -> list[str]:
    """将提示词发送给配置的 LLM 服务端。

    Args:
        prompt: 编排器构造的提示词。
        provider: defaults.yaml 中的服务商标识，目前支持 ``openai``。
        model: 目标模型名称。
        api_key: API 密钥，缺失时将使用回退逻辑。
        temperature: 采样温度，会直接传递给 OpenAI 接口。
        thinking: reasoning/“thinking”力度设置。
        allowed_fields: 允许引用的字段列表，用于 fallback 生成时保障安全。
        logger: 可选日志记录器，用于输出原始响应。

    Returns:
        因子表达式列表；没有密钥或 SDK 不可用时退回确定性样本。
    """

    if provider.lower() != "openai":
        return _fallback_generation(prompt, allowed_fields, logger)
    if OpenAI is None:
        raise RuntimeError(
            "未安装 openai SDK，无法调用真实 LLM。请 `pip install openai` 或提供其他提供商。"
        )
    effective_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_key:
        return _fallback_generation(prompt, allowed_fields, logger)

    client = OpenAI(api_key=effective_key)
    # ✅ 使用 Chat Completions；不要传 reasoning 参数
    resp = client.chat.completions.create(
        model=model,  # 例如 "gpt-4o-mini"
        temperature=float(temperature or 0.2),
        messages=[
            {"role": "system", "content": "You are a quant researcher. Return one alpha per line."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    if logger is not None:
        logger.info("LLM raw output: %s", content)
    return parse_llm_output(content)


def _fallback_generation(prompt: str, allowed_fields: list[str] | None = None, logger=None) -> list[str]:
    """在无法访问真实 LLM 时生成可复现且受限于字段列表的伪造因子。"""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    seeds = [digest[i : i + 8] for i in range(0, len(digest), 8)][:3]
    fields = allowed_fields or ["CLOSE", "VWAP", "RET"]
    responses = []
    for i, seed in enumerate(seeds):
        field = fields[i % len(fields)]
        responses.append(f"FA{i+1}: RANK({field}) + 0.{seed[:3]} * DELTA({field}, {i + 1})")
    text = "\n".join(responses)
    if logger is not None:
        logger.info("Fallback raw output: %s", text)
    return parse_llm_output(text)
