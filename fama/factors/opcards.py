"""LLM 提示中可用的算子语义卡片。"""

from __future__ import annotations

OP_CARDS: dict[str, str] = {
    "RANK": "RANK(x): 按日期对所有标的做截面排名，输出范围 [0,1]，并列取平均。",
    "DELTA": "DELTA(x, n): 每个标的的时间差分 x_t - x_{t-n}，n 为正整数，历史不足返回 NA。",
    "TS_MEAN": "TS_MEAN(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列均值，历史不足返回 NA。",
    "TS_STDDEV": "TS_STDDEV(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列标准差 (总体)，历史不足返回 NA。",
    "CORREL": "CORREL(x, y, n): 每个标的最近 n 个 bar 的皮尔逊相关系数，范围 [-1, 1]，历史不足返回 NA。",
    "Z_SCORE": "Z_SCORE(x): 每日截面 z-score，(x - 当日均值) / 当日标准差；若标准差为 0 则输出 0。",
    "SIGN": "SIGN(x): 元素级符号函数，输出 {-1, 0, 1}。",
    "ABS": "ABS(x): 元素级绝对值。",
}


def render_cards(ops: list[str]) -> str:
    """将算子白名单渲染为多段文本。"""

    lines: list[str] = []
    for op in ops:
        card = OP_CARDS.get(op)
        if not card:
            continue
        lines.append(f"### {op}\n{card}")
    return "\n\n".join(lines)
