#!/usr/bin/env python
"""Fill the explanation field of factor_cache entries by calling the existing LLM client."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from fama.data.factor_space import deserialize_factor_set, serialize_factor_set
from fama.utils.logging import get_logger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate economic explanations for factors in factor_cache.")
    parser.add_argument("--config", default="fama/config/defaults.yaml", help="Path to configuration YAML.")
    parser.add_argument(
        "--factor-cache",
        default="/Users/hpy/PycharmProjects/FAMA/data/factor_cache",
        help="Optional factor cache path (defaults to config paths.factor_cache).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key override (falls back to llm.api_key or environment).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate explanations even if present.")
    args = parser.parse_args()

    from fama.utils.io import read_yaml

    load_dotenv()
    cfg = read_yaml(args.config)
    cache_path = Path(args.factor_cache or cfg["paths"]["factor_cache"]).resolve()
    fs = deserialize_factor_set(str(cache_path))
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "openai")
    model = llm_cfg.get("model", "gpt-4o")
    temperature = llm_cfg.get("temperature", 0)
    api_key_env = llm_cfg.get("api_key_env", "LLM_API_KEY")
    api_key = args.api_key or llm_cfg.get("api_key") or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    logger = get_logger(__name__)

    if provider.lower() != "openai":
        raise ValueError("generate_factor_explanations 目前仅支持 provider=openai。")
    if OpenAI is None:
        raise RuntimeError("未安装 openai SDK，请 `pip install openai` 再运行该脚本。")
    if not api_key:
        raise RuntimeError(
            "未提供 API Key。请在 .env 中设置 OPENAI_API_KEY 或配置 llm.api_key/api_key_env，"
            "或者通过 --api-key 显式指定。"
        )

    client = OpenAI(api_key=api_key)

    updated = False
    for factor in fs.factors:
        if factor.explanation and not args.overwrite:
            continue
        prompt = (
            "你是一名量化研究员。请用中文简洁说明以下因子的经济学含义，只输出一句话，不要编号、JSON 或多余说明。\n"
            f"因子表达式: {factor.expression}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=float(temperature or 0.2),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名量化研究员，请用中文回答，对因子的经济学含义给出精炼的一句话。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover
            logger.error("LLM 调用失败：%s", exc)
            raise

        content = (resp.choices[0].message.content or "").strip()
        factor.explanation = content.splitlines()[0].strip()
        updated = True
        logger.info("Explanation generated for %s", factor.name)

    if updated:
        serialize_factor_set(fs, str(cache_path))
        logger.info("Factor cache updated with explanations.")
    else:
        logger.info("No explanations updated (all factors already have explanations?).")


if __name__ == "__main__":
    main()
