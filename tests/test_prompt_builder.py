"""提示词构建模块的测试。"""

from __future__ import annotations

from fama.mining import prompt_builder


def test_build_prompt_signature(tmp_path):
    """提示词构建器应正确替换 CSS/CoE 内容。"""

    template_path = tmp_path / "template.txt"
    template_path.write_text("CSS:\n{css_examples}\nCoE:{coe_path}\nConstraints:{constraints}\n", encoding="utf-8")
    prompt = prompt_builder.build_prompt(
        ["RANK(CLOSE)"],
        ["RANK(CLOSE)", "DELTA(CLOSE, 3)"],
        {"instructions_path": str(template_path), "max_new_factors": 3},
    )
    assert "RANK(CLOSE)" in prompt
    assert "max_new_factors" in prompt


def test_parse_llm_output_signature():
    """确认冒号分隔的响应可解析为表达式列表。"""

    text = "FA1: RANK(CLOSE)\nFA2: DELTA(CLOSE, 3)"
    expressions = prompt_builder.parse_llm_output(text)
    assert expressions == ["RANK(CLOSE)", "DELTA(CLOSE, 3)"]
