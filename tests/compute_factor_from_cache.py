"""Helper script to compute any cached factor expression against the market data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from fama.data.dataloader import compute_factor_values, load_market_data
from fama.utils.io import read_yaml, write_yaml


def _load_factor_expression(cache_path: Path, factor_name: str | None) -> tuple[str, str]:
    factors = read_yaml(str(cache_path)) or []
    if not factors:
        raise ValueError(f"No factors found in cache: {cache_path}")
    if factor_name is None:
        entry = factors[0]
    else:
        matched = next((item for item in factors if item.get("name") == factor_name), None)
        if matched is None:
            available = ", ".join(item.get("name", "?") for item in factors[:20])
            raise ValueError(f"Factor {factor_name} not found in cache. Available head: {available}")
        entry = matched
    name = entry.get("name")
    expr = entry.get("expression")
    if not name or not expr:
        raise ValueError("Factor entry must contain both 'name' and 'expression'.")
    return name, expr


def _persist_series(df, factor_name: str, output_path: Path | None) -> None:
    df = df.rename(columns={"date": "time", "symbol": "unique_id"})
    df["factor_tag"] = factor_name
    df = df[["time", "unique_id", "factor_tag", "value"]]
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote factor values to {output_path}")
    else:
        print(df.head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a factor from factor_cache using project data/config.")
    parser.add_argument("--config", default="fama/config/defaults.yaml", help="Path to the YAML config.")
    parser.add_argument("--factor-name", default=None, help="Factor name from factor_cache; defaults to first entry.")
    parser.add_argument(
        "--factor-cache",
        default=None,
        help="Optional override for factor cache path; defaults to config paths.factor_cache.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save computed values as CSV.")
    args = parser.parse_args()

    cfg: Dict[str, Any] = read_yaml(args.config)
    cache_path = Path(args.factor_cache or cfg["paths"]["factor_cache"]).resolve()
    market_path = Path(cfg["paths"]["market_data"]).resolve()

    factor_name, expression = _load_factor_expression(cache_path, args.factor_name)
    market_df = load_market_data(str(market_path))
    factor_df = compute_factor_values(market_df, [expression], cfg)
    if expression not in factor_df.columns:
        raise RuntimeError(f"Expression {expression} produced no data.")

    output = Path(args.output).resolve() if args.output else None
    series = factor_df[expression].rename("value").reset_index()
    _persist_series(series, factor_name, output)


if __name__ == "__main__":
    main()
