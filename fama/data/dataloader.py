"""与 README 数据接入章节对应的数据加载与特征发现工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from fama.factors.alpha_lib import evaluate_expression, list_seed_alphas

DATE_CANDIDATES = {"date", "trade_date", "calc_date", "datetime", "time"}
SYMBOL_CANDIDATES = {"symbol", "ticker", "secid", "sec_code", "asset", "code", "unique_id"}


def load_market_data(path: str) -> "pd.DataFrame":
    """从 ``path`` 加载结构化行情数据。

    Args:
        path: README “Data & Factor Inputs” 中提到的文件路径。

    Returns:
        以 ``(date, symbol)`` 为 MultiIndex、包含所有可用指标的 pandas DataFrame。
        若文件缺失，则自动模拟 Alpha101 风格的数据以便端到端运行。
    """

    data_path = Path(path)
    if data_path.exists():
        frame = _read_table(data_path)
        return _normalize_loaded_frame(frame)

    return _simulate_market_data()


def available_factor_inputs(df: "pd.DataFrame") -> list[str]:
    """列出可供表达式引用的指标名称（全部转为大写）。"""

    _, keys = _build_evaluation_context(df)
    return sorted(keys)


def compute_weighted_price(
    df: "pd.DataFrame",
    price_column: str | None = None,
    volume_column: str | None = None,
) -> "pd.Series":
    """计算流动性加权价格序列（VWAP）。

    Args:
        df: README “Data & Factor Inputs” 中定义的输入数据。
        price_column: 可选的价格列名，默认自动匹配 ``close``。
        volume_column: 可选的成交量列名，默认自动匹配 ``volume``。

    Returns:
        可用于 CSS 统计的加权价格序列。
    """

    price_column = price_column or _match_column_name(df.columns, {"close"})
    volume_column = volume_column or _match_column_name(df.columns, {"volume"})
    if not price_column or not volume_column:
        raise KeyError("计算 VWAP 需要同时存在 close 与 volume 列。")

    traded_value = df[price_column] * df[volume_column]
    total_volume = df[volume_column].groupby(level=0).transform("sum").replace(0, np.nan)
    vw_price = traded_value / total_volume
    return vw_price.fillna(df[price_column])


def compute_factor_values(df: "pd.DataFrame", formulas: list[str]) -> "pd.DataFrame":
    """根据符号表达式生成因子矩阵。

    Args:
        df: 满足 README “Data & Factor Inputs” 规范的行情数据。
        formulas: 由编排器提供的符号表达式列表。

    Returns:
        行索引为 ``(date, symbol)``、列为每个表达式的 DataFrame。
    """

    if not formulas:
        formulas = list_seed_alphas()

    context, _ = _build_evaluation_context(df)
    factor_columns = {}
    for idx, formula in enumerate(formulas):
        label = formula.strip() or f"factor_{idx}"
        try:
            factor_columns[label] = evaluate_expression(formula, context)
        except Exception as exc:  # pragma: no cover - 实际表达式视数据而定
            raise ValueError(f"因子表达式 {label} 计算失败: {exc}") from exc

    factor_df = pd.DataFrame(factor_columns).sort_index().fillna(0.0)
    return factor_df


def zscore_normalize(s: "pd.Series") -> "pd.Series":
    """对 Series 做 Z-score 标准化。"""

    grouped = s.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    normalized = (s - mean) / std
    return normalized.fillna(0.0)


def _read_table(path: Path) -> "pd.DataFrame":
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - 环境缺少引擎时才触发
            raise ImportError(
                "读取 Parquet 需要 pyarrow 或 fastparquet。请安装其中之一后重试。"
            ) from exc
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"暂不支持的文件格式: {path.suffix}")


def _normalize_loaded_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    """将用户数据重建为项目期望的 (date, symbol) MultiIndex 结构。"""

    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels >= 2:
        return frame.sort_index()

    date_col = _match_column_name(frame.columns, DATE_CANDIDATES)
    symbol_col = _match_column_name(frame.columns, SYMBOL_CANDIDATES)
    if not date_col or not symbol_col:
        raise ValueError("请确保数据表包含日期与标的列，例如 date/symbol。")

    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index([date_col, symbol_col]).sort_index()
    frame.index.names = ["date", "symbol"]
    frame = frame.drop(columns=[col for col in [date_col, symbol_col] if col in frame.columns], errors="ignore")
    return frame


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
    frame.index.names = ["date", "symbol"]
    return frame


def _build_evaluation_context(df: "pd.DataFrame") -> tuple[dict[str, "pd.Series"], set[str]]:
    """构造表达式所需的基础序列上下文，并返回所有可用键集合。"""

    context: dict[str, pd.Series] = {}
    available: set[str] = set()

    for column in df.columns:
        series = df[column]
        if ptypes.is_numeric_dtype(series):
            key = column.upper()
            context[key] = series
            available.add(key)

    close_col = _match_column_name(df.columns, {"close"})
    if close_col:
        close_series = df[close_col]
        ret = close_series.groupby(level=1).pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
        context["RET"] = ret
        available.add("RET")

        volume_col = _match_column_name(df.columns, {"volume"})
        if volume_col:
            try:
                vwap = compute_weighted_price(df, close_col, volume_col)
                context["VWAP"] = vwap
                available.add("VWAP")
            except KeyError:
                pass

    return context, available


def _match_column_name(columns: Iterable[str], candidates: Sequence[str] | set[str]) -> str | None:
    lower_candidates = {c.lower() for c in candidates}
    for col in columns:
        if col.lower() in lower_candidates:
            return col
    return None
