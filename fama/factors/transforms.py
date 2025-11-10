"""README “Transforms & Normalization” 中使用的因子变换函数。"""

from __future__ import annotations

import pandas as pd


def normalize_factor_series(s: "pd.Series") -> "pd.Series":
    """按时间截面进行 Z-score 归一化（参见 README “Data & Factor Inputs”）。

    Args:
        s: 待归一化的因子信号。

    Returns:
        各时间截面独立标准化后的结果。
    """

    grouped = s.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, float("nan"))
    normalized = (s - mean) / std
    return normalized.fillna(0.0)


def rank_transform(s: "pd.Series") -> "pd.Series":
    """将值转换为截面排名百分位（README “Data & Factor Inputs”）。

    Args:
        s: 需要排名的序列。

    Returns:
        适合 CSS 使用的百分位排名。
    """

    return s.groupby(level=0).rank(pct=True)


def lag_shift(s: "pd.Series", periods: int) -> "pd.Series":
    """按照 README 建议对因子进行 ``periods`` 期的滞后处理。

    Args:
        s: 因子序列。
        periods: 滞后期数。

    Returns:
        滞后后的序列，可用于预测实验。
    """

    return s.groupby(level=1).shift(periods)
