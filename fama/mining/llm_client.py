"""负责向 LLM 请求新因子的客户端适配层。"""

from __future__ import annotations

import hashlib
from fama.mining.prompt_builder import parse_llm_output


def request_new_factors(prompt: str, provider: str, model: str, api_key: str | None = None) -> list[str]:
    """将提示词发送给配置的 LLM 服务端。

    Args:
        prompt: 编排器构造的提示词。
        provider: defaults.yaml 中的服务商标识。
        model: 目标模型名称。
        api_key: 可选的 API 密钥，若为空则使用内置的确定性回退逻辑。

    Returns:
        LLM 返回的因子表达式列表；没有密钥时会使用伪造响应以保持流程可执行。
    """

    if not api_key:
        return _fallback_generation(prompt)

    raise NotImplementedError(
        "LLM provider integration is intentionally left blank. Implement the "
        "provider call using your preferred SDK and return parse_llm_output(payload)."
    )


def _fallback_generation(prompt: str) -> list[str]:
    """在未提供 API Key 时，依据提示词生成可复现的伪造因子列表。"""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    seeds = [digest[i : i + 8] for i in range(0, len(digest), 8)][:3]
    responses = [f"FA{i+1}: RANK(CLOSE) + {seed[:3]} * DELTA(VWAP, {i + 1})" for i, seed in enumerate(seeds)]
    return parse_llm_output("\n".join(responses))
