"""提示词构建模块的测试。"""

from __future__ import annotations

from fama.mining import prompt_builder


def test_build_prompt_signature(tmp_path):
    """提示词构建器应正确替换 CSS/CoE 内容。"""

    template_path = tmp_path / "template.txt"
    template_path.write_text("placeholder", encoding="utf-8")
    prompt = prompt_builder.build_prompt(
        ["RANK(CLOSE)"],
        ["RANK(CLOSE)", "DELTA(CLOSE, 3)"],
        {"instructions_path": str(template_path), "max_new_factors": 3, "operator_whitelist": ["RANK", "DELTA"], "deny_fields": []},
        available_fields=["CLOSE", "RET"],
    )
    assert "OPS-CHECKSUM" in prompt
    assert "RANK(CLOSE)" in prompt
    assert "仅允许使用以下字段" in prompt


def test_parse_llm_output_signature():
    """确认冒号分隔的响应可解析为表达式列表。"""

    text = "FA1: RANK(CLOSE)\nFA2: DELTA(CLOSE, 3)"
    expressions = prompt_builder.parse_llm_output(text)
    assert expressions == ["RANK(CLOSE)", "DELTA(CLOSE, 3)"]
