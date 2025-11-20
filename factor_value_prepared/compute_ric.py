#!/usr/bin/env python
"""Compute per-asset RankIC metrics from precomputed factor values."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from RankIC.efficientCalculation import EfficientCalculator
from fama.data.dataloader import load_market_data


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-asset RIC from factor CSV.")
    parser.add_argument(
        "--config",
        default="fama/config/defaults.yaml",
        type=Path,
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--input",
        default=Path("factor_value_prepared/factor_values.csv"),
        type=Path,
        help="Input CSV produced by compute_factor_values.py.",
    )
    parser.add_argument(
        "--output",
        default=Path("factor_value_prepared/factor_ric.csv"),
        type=Path,
        help="Destination CSV file (will be overwritten).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional override for RIC start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional override for RIC end date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.input.exists():
        raise FileNotFoundError(f"Factor CSV not found: {args.input}")

    factor_df = pd.read_csv(args.input)
    required_cols = {"time", "unique_id", "factor_tag", "value"}
    missing = required_cols - set(factor_df.columns)
    if missing:
        raise ValueError(f"Factor CSV missing required columns: {sorted(missing)}")
    factor_df["time"] = pd.to_datetime(factor_df["time"])

    coe_cfg = cfg.get("coe", {})
    start_date = args.start or coe_cfg.get("ric_start_date")
    end_date = args.end or coe_cfg.get("ric_end_date")
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    if start_ts is not None:
        factor_df = factor_df[factor_df["time"] >= start_ts]
    if end_ts is not None:
        factor_df = factor_df[factor_df["time"] <= end_ts]

    if factor_df.empty:
        raise ValueError("Filtered factor data is empty; nothing to compute.")

    market = load_market_data(cfg["paths"]["market_data"])
    close_wide = market["close"].unstack(level=1).sort_index()
    close_wide = close_wide.ffill().bfill()
    returns_wide = close_wide.pct_change(1).shift(-1)
    if start_ts is not None:
        returns_wide = returns_wide[returns_wide.index >= start_ts]
    if end_ts is not None:
        returns_wide = returns_wide[returns_wide.index <= end_ts]

    calc = EfficientCalculator()
    min_samples = 10
    ric_records: list[dict] = []

    grouped = factor_df.groupby(["unique_id", "factor_tag"])
    for (asset_id, factor_tag), group in grouped:
        if asset_id not in returns_wide.columns:
            continue
        factor_series = group.sort_values("time").set_index("time")["value"]
        returns_series = returns_wide[asset_id]
        common_dates = factor_series.index.intersection(returns_series.index)
        if not len(common_dates):
            continue
        factor_aligned = factor_series.loc[common_dates]
        returns_aligned = returns_series.loc[common_dates]
        valid_mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]
        if len(factor_clean) < min_samples:
            continue
        if factor_clean.nunique() <= 1 or returns_clean.nunique() <= 1:
            continue
        ric = calc.efficent_cal_ric(factor_clean.values, returns_clean.values)
        if pd.isna(ric):
            continue
        ric_records.append(
            {
                "unique_id": asset_id,
                "factor_tag": factor_tag,
                "ric": float(ric),
                "sample_count": len(factor_clean),
                "start_date": factor_clean.index.min(),
                "end_date": factor_clean.index.max(),
            }
        )

    ric_df = pd.DataFrame(ric_records)
    if ric_df.empty:
        raise ValueError("No valid RIC results computed.")
    ric_df["abs_ric"] = ric_df["ric"].abs()
    ric_df.sort_values("abs_ric", ascending=False, inplace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ric_df.to_csv(args.output, index=False)
    print(f"Saved {len(ric_df)} RIC rows to {args.output}.")


if __name__ == "__main__":
    main()
