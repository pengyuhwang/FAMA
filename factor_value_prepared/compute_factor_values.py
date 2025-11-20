#!/usr/bin/env python
"""Compute all cached factor expressions into a long CSV table."""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import pandas as pd

from fama.data.dataloader import load_market_data, compute_factor_values
from fama.data.factor_space import deserialize_factor_set


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def resolve_factor_repo(cfg: dict) -> Path:
    cache_path = Path(cfg["paths"]["factor_cache"])
    if cache_path.is_dir():
        return cache_path / "factors.yaml"
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute factor values to CSV.")
    parser.add_argument(
        "--config",
        default="fama/config/defaults.yaml",
        type=Path,
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--output",
        default=Path("factor_value_prepared/factor_values.csv"),
        type=Path,
        help="Destination CSV file (will be overwritten).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_path = resolve_factor_repo(cfg)
    if not repo_path.exists():
        raise FileNotFoundError(f"Factor cache file not found: {repo_path}")

    factor_set = deserialize_factor_set(str(repo_path))
    expressions = [factor.expression for factor in factor_set.factors]
    if not expressions:
        raise ValueError("No expressions found in factor cache.")

    market = load_market_data(cfg["paths"]["market_data"])
    frame = compute_factor_values(market, expressions, cfg=cfg)
    if frame.empty:
        raise ValueError("Factor computation returned an empty frame.")

    stacked = frame.stack(dropna=False).rename("value").reset_index()
    column_map = {}
    for candidate in ("date", "time"):
        if candidate in stacked.columns:
            column_map[candidate] = "time"
            break
    for candidate in ("symbol", "unique_id"):
        if candidate in stacked.columns:
            column_map[candidate] = "unique_id"
            break
    for candidate in ("level_2", "factor"):
        if candidate in stacked.columns:
            column_map[candidate] = "factor_tag"
            break
    stacked = stacked.rename(columns=column_map)
    required_cols = ["time", "unique_id", "factor_tag", "value"]
    missing = [col for col in required_cols if col not in stacked.columns]
    if missing:
        raise ValueError(f"Unable to map stacked columns to required names: {missing}")
    stacked = stacked[required_cols]
    expr_to_name = {factor.expression: factor.name for factor in factor_set.factors}
    stacked["factor_tag"] = stacked["factor_tag"].map(expr_to_name).fillna(stacked["factor_tag"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stacked.to_csv(args.output, index=False)
    print(f"Saved factor values to {args.output} ({len(stacked)} rows).")


if __name__ == "__main__":
    main()
