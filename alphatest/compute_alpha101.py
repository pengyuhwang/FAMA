#!/usr/bin/env python
"""Compute KunQuant Alpha101 factors directly into CSV files."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.runner import KunRunner as kr
from KunQuant.predefined import Alpha101 as alpha101

warnings.filterwarnings(
    "ignore",
    message="The previous implementation of stack is deprecated",
    category=FutureWarning,
)


def load_market_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" not in df.columns or "unique_id" not in df.columns:
        raise ValueError("Parquet must contain 'time' and 'unique_id' columns.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["time", "unique_id"]).sort_index()
    return df


def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "vwap" not in df.columns:
        vwap = df["amount"] / df["volume"].replace(0, np.nan)
        df = df.assign(vwap=vwap)
    return df


def build_ts_inputs(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[pd.Timestamp], list[str]]:
    dates = sorted(df.index.get_level_values(0).unique())
    symbols = sorted(df.index.get_level_values(1).unique())

    def to_matrix(column: str) -> np.ndarray:
        pivot = df[column].unstack(level=1).reindex(index=dates, columns=symbols)
        return pivot.to_numpy(dtype=np.float32).T  # shape (symbols, time)

    inputs = {
        "open": to_matrix("open"),
        "high": to_matrix("high"),
        "low": to_matrix("low"),
        "close": to_matrix("close"),
        "volume": to_matrix("volume"),
        "amount": to_matrix("amount"),
        "vwap": to_matrix("vwap"),
    }
    return inputs, dates, symbols


def run_graph(inputs_np: dict[str, np.ndarray], dates, symbols) -> dict[str, pd.Series]:
    builder = Builder()
    with builder:
        inp = {key: Input(key) for key in inputs_np.keys()}
        data = alpha101.AllData(
            open=inp["open"],
            close=inp["close"],
            high=inp["high"],
            low=inp["low"],
            volume=inp["volume"],
            amount=inp["amount"],
            vwap=inp["vwap"],
        )
        funcs = [
            (name, getattr(alpha101, name))
            for name in dir(alpha101)
            if name.startswith("alpha") and callable(getattr(alpha101, name))
        ]
        compiled = []
        counter = 0
        for name, func in funcs:
            try:
                ir = func(data)
            except Exception as exc:
                print(f"Skipping {name}: {exc}")
                continue
            counter += 1
            Output(ir, f"f_{counter}")
            compiled.append(name)
    if not compiled:
        return {}

    func = Function(builder.ops)
    lib = cfake.compileit(
        [("alpha101_graph", func, KunCompilerConfig(input_layout="TS", output_layout="TS"))],
        "alpha101_graph_lib",
        cfake.CppCompilerConfig(),
    )
    module = lib.getModule("alpha101_graph")
    executor = kr.createMultiThreadExecutor(max(1, 4))
    first = next(iter(inputs_np.values()))
    num_stocks = first.shape[0]
    length = first.shape[1]
    out = kr.runGraph(
        executor,
        module,
        inputs_np,
        0,
        length,
        {},
        True,
        num_stocks=num_stocks,
    )

    num_dates = len(dates)
    num_symbols = len(symbols)
    stacked: dict[str, pd.Series] = {}
    for idx, name in enumerate(compiled, 1):
        matrix = np.asarray(out[f"f_{idx}"])
        if matrix.shape == (num_symbols, num_dates):
            values = matrix.T
        elif matrix.shape == (num_dates, num_symbols):
            values = matrix
        else:
            raise ValueError(
                f"Unexpected shape {matrix.shape} for {name}; "
                f"expected {(num_symbols, num_dates)} or {(num_dates, num_symbols)}"
            )
        df = pd.DataFrame(values, index=dates, columns=symbols)
        stacked[name] = df.stack(dropna=False)
    return stacked


def write_factor_csv(series_dict: dict[str, pd.Series], output_dir: Path) -> None:
    if not series_dict:
        print("No factors to write.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for name, series in series_dict.items():
        df = series.rename("value").reset_index()
        time_col, asset_col = df.columns[:2]
        df = df.rename(columns={time_col: "time", asset_col: "unique_id"})
        df["factor_tag"] = name
        combined.append(df[["time", "unique_id", "factor_tag", "value"]])
    result = pd.concat(combined, ignore_index=True)
    out_path = output_dir / "alpha101_all.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved {len(series_dict)} factors -> {out_path}")


def main():
    data_path = "/Users/hpy/PycharmProjects/FAMA/data/fof_price_updating.parquet"
    df = ensure_fields(load_market_data(data_path))
    inputs_np, dates, symbols = build_ts_inputs(df)
    series_dict = run_graph(inputs_np, dates, symbols)
    write_factor_csv(series_dict, Path("alphatest/factor_values"))


if __name__ == "__main__":
    main()
