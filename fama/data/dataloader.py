"""与 README 数据接入章节对应的数据加载工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fama.factors.alpha_lib import evaluate_expression, list_seed_alphas


def load_market_data(path: str) -> "pd.DataFrame":
    """从 ``path`` 加载结构化行情数据。

    Args:
        path: README “Data & Factor Inputs” 中提到的文件路径。

    Returns:
        以 ``(date, symbol)`` 为 MultiIndex、包含 OHLCV 列的 pandas DataFrame。
        若文件缺失，则自动模拟 Alpha101 风格的数据以便端到端运行。
    """

    data_path = Path(path)
    if data_path.exists():
        if data_path.suffix == ".parquet":
            frame = pd.read_parquet(data_path)
        else:
            frame = pd.read_csv(data_path)
        return _normalize_loaded_frame(frame)

    return _simulate_market_data()


def compute_weighted_price(df: "pd.DataFrame") -> "pd.Series":
    """计算流动性加权价格序列。

    Args:
        df: README “Data & Factor Inputs” 中定义的输入数据。

    Returns:
        可用于 CSS 统计的加权价格序列（参见 README “CSS & Diversity”）。
    """

    traded_value = df["close"] * df["volume"]
    total_volume = df["volume"].groupby(level=0).transform("sum").replace(0, np.nan)
    vw_price = traded_value / total_volume
    return vw_price.fillna(df["close"])


def compute_factor_values(df: "pd.DataFrame", formulas: list[str]) -> "pd.DataFrame":
    """根据符号表达式生成因子矩阵。

    Args:
        df: 满足 README “Data & Factor Inputs” 规范的归一化行情数据。
        formulas: 由编排器提供的符号表达式列表（README “单次流程”）。

    Returns:
        行索引为 ``(date, symbol)``、列为每个表达式的 DataFrame。
    """

    if not formulas:
        formulas = list_seed_alphas()

    context = _build_evaluation_context(df)
    factor_columns = {}
    for idx, formula in enumerate(formulas):
        label = formula.strip() or f"factor_{idx}"
        factor_columns[label] = evaluate_expression(formula, context)

    factor_df = pd.DataFrame(factor_columns).sort_index().fillna(0.0)
    return factor_df


def zscore_normalize(s: "pd.Series") -> "pd.Series":
    """对 Series 做 Z-score 标准化。

    Args:
        s: README “Data & Factor Inputs” 所述的原始序列。

    Returns:
        保持原 MultiIndex 的标准化结果。
    """

    grouped = s.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    normalized = (s - mean) / std
    return normalized.fillna(0.0)


def _normalize_loaded_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    """将用户数据重建为项目期望的 MultiIndex OHLCV 结构。"""

    if {"date", "symbol"}.issubset(frame.columns):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index(["date", "symbol"]).sort_index()
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("输入行情数据必须使用 (date, symbol) 为索引。")
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"缺少必要的 OHLCV 列: {missing}")
    return frame[list(sorted(required_cols))]


def _simulate_market_data(
    symbols: list[str] | None = None,
    num_days: int = 180,
    seed: int = 7,
) -> "pd.DataFrame":
    """生成可复现实验的合成 OHLCV 数据。"""

    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = [f"SYM{i}" for i in range(1, 6)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=num_days, freq="B")
    records = []
    for symbol in symbols:
        price = 50 + rng.normal(0, 1)
        for date in dates:
            drift = rng.normal(0.0005, 0.01)
            shock = rng.normal(0, 1)
            open_price = max(1.0, price * (1 + drift + shock * 0.001))
            close_price = max(1.0, open_price * (1 + rng.normal(0, 0.01)))
            high_price = max(open_price, close_price) * (1 + abs(rng.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(abs(rng.normal(1e6, 2e5)))
            price = close_price
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
    frame = pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
    return frame


def _build_evaluation_context(df: "pd.DataFrame") -> dict[str, "pd.Series"]:
    """构造 Alpha101 风格表达式所需的基础序列上下文。"""

    context = {
        "OPEN": df["open"],
        "HIGH": df["high"],
        "LOW": df["low"],
        "CLOSE": df["close"],
        "VOLUME": df["volume"],
    }
    group = df["close"].groupby(level=1)
    context["RET"] = group.pct_change().fillna(0.0)
    context["VWAP"] = compute_weighted_price(df)
    return context
