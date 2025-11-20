"""LLM 提示中可用的算子语义卡片。"""

from __future__ import annotations

OP_CARDS: dict[str, str] = {
    "RANK": "RANK(x): 按日期对所有标的做截面排名，输出范围 [0,1]，并列取平均。",
    "DELTA": "DELTA(x, n): 每个标的的时间差分 x_t - x_{t-n}，n 为正整数，历史不足返回 NA。",
    "DELAY": "DELAY(x, n): 将序列向后平移 n 个 bar，用于引用过去的取值。",
    "TS_MEAN": "TS_MEAN(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列均值，历史不足返回 NA。",
    "TS_SUM": "TS_SUM(x, n): 每个标的最近 n 个 bar 的滚动求和。",
    "TS_STDDEV": "TS_STDDEV(x, n): 每个标的最近 n 个 bar (含当期) 的时间序列标准差 (总体)，历史不足返回 NA。",
    "TS_MIN": "TS_MIN(x, n): 最近 n 个 bar 的滚动最小值。",
    "TS_MAX": "TS_MAX(x, n): 最近 n 个 bar 的滚动最大值。",
    "TS_PRODUCT": "TS_PRODUCT(x, n): 最近 n 个 bar 的连乘结果（适合累积收益）。",
    "TS_ARGMAX": "TS_ARGMAX(x, n): 最近 n 个 bar 内最大值所在的相对位置（0 为窗口起点）。",
    "TS_ARGMIN": "TS_ARGMIN(x, n): 最近 n 个 bar 内最小值所在的相对位置。",
    "TS_RANK": "TS_RANK(x, n): 当前值在最近 n 个 bar 中的排序百分位。",
    "CORREL": "CORREL(x, y, n): 每个标的最近 n 个 bar 的皮尔逊相关系数，范围 [-1, 1]，历史不足返回 NA。",
    "COVAR": "COVAR(x, y, n): 最近 n 个 bar 的协方差，用于衡量联合波动。",
    "Z_SCORE": "Z_SCORE(x): 每日截面 z-score，(x - 当日均值) / 当日标准差；若标准差为 0 则输出 0。",
    "SIGN": "SIGN(x): 元素级符号函数，输出 {-1, 0, 1}。",
    "ABS": "ABS(x): 元素级绝对值。",
    "DECAY_LINEAR": "DECAY_LINEAR(x, n): 线性加权的滚动平均，越近的 bar 权重越大。",
    "SCALE": "SCALE(x): 当日截面标准化，使 ∑|x| = 1，用于消除量纲。",
    "IF": "IF(cond, a, b): 条件选择，cond≠0 取 a，否则取 b。",
    "GT": "GT(x, y): 大于比较，返回 {0,1}。",
    "GE": "GE(x, y): 大于等于比较，返回 {0,1}。",
    "LT": "LT(x, y): 小于比较，返回 {0,1}。",
    "LE": "LE(x, y): 小于等于比较，返回 {0,1}。",
    "EQ": "EQ(x, y): 相等比较，返回 {0,1}。",
    "REPLACE_NAN_INF": "REPLACE_NAN_INF(x, v): 将 x 中的 NaN/Inf 替换为常数 v。",
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
