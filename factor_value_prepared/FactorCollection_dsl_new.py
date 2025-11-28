#!/usr/bin/env python
"""因子值 DSL 计算（新版 KunQuant 后端）。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from factor_value_prepared.FactorCollection import FactorCollection
from fama.data.factor_space import deserialize_factor_set
from fama.data.kun_backend_new import compute_factor_values_kunquant_new
from fama.factors.alpha101_extractor import extract_alpha101_expressions
from fama.utils.io import ensure_dir


class FactorCollectionDSLNew(FactorCollection):
    def __init__(
        self,
        config_path: str | Path = "/Users/hpy/PycharmProjects/FAMA/fama/config/defaults.yaml",
        factor_cache_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        with self.config_path.open("r", encoding="utf-8") as fp:
            self.cfg = yaml.safe_load(fp)

        cache_path = (
            Path(factor_cache_path)
            if factor_cache_path
            else Path(self.cfg["paths"].get("factor_cache", "/Users/hpy/PycharmProjects/FAMA/data/factor_cache_new/factors.yaml"))
        )
        self.factor_cache_path = cache_path
        if not self.factor_cache_path.exists():
            self._regenerate_factor_cache()

    def update_dsl_factors(
        self,
        output_path: str | Path | None = None,
        threads: int | None = None,
    ) -> Path:
        factor_set = deserialize_factor_set(str(self.factor_cache_path))
        if not factor_set.factors:
            raise ValueError("factor_cache_new 为空，无法计算 DSL 因子。")
        expressions = [factor.expression for factor in factor_set.factors]

        market_df = self._build_market_frame()
        compute_cfg = self.cfg.get("compute", {})
        kun_threads = threads if threads is not None else int(compute_cfg.get("threads", 4))
        layout = str(compute_cfg.get("layout", "TS"))
        kun_df, fallback = compute_factor_values_kunquant_new(
            market_df,
            expressions,
            threads=kun_threads,
            layout=layout,
        )
        if fallback:
            print(f"[DSL-New] KunQuant 无法解析 {len(fallback)} 个表达式，回退 Python 解释器。")
            from fama.data.dataloader import _compute_factor_values_python

            fallback_df = _compute_factor_values_python(market_df, fallback)
            kun_df = pd.concat([kun_df, fallback_df], axis=1).sort_index()

        stacked = kun_df.stack(future_stack=True).rename("value").reset_index()
        stacked = stacked.rename(columns={"date": "time", "symbol": "unique_id", "level_2": "factor_tag"})
        expr_to_name = {factor.expression: factor.name for factor in factor_set.factors}
        stacked["factor_tag"] = stacked["factor_tag"].map(expr_to_name).fillna(stacked["factor_tag"])
        stacked = stacked[["time", "unique_id", "factor_tag", "value"]]
        stacked = stacked.drop_duplicates(subset=["time", "unique_id", "factor_tag"])

        save_path = Path(output_path) if output_path else (self.factor_dir / "dsl_factors_new.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".csv":
            stacked.to_csv(save_path, index=False)
        else:
            stacked.to_parquet(save_path, index=False)
        print(f"[DSL-New] 因子计算完成，写入 {save_path}（{len(stacked)} 行）。")
        return save_path

    def _build_market_frame(self) -> pd.DataFrame:
        required = {"time", "unique_id", "open", "high", "low", "close", "volume", "amount"}
        missing = required - set(self.native_price.columns)
        if missing:
            raise ValueError(f"行情数据缺少列: {missing}")

        native = deepcopy(self.native_price)
        frames = []
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for uid, frame in native.groupby("unique_id", sort=False):
            df = (
                frame.set_index("time")[cols]
                .reindex(self.working_days)
                .ffill()
                .bfill()
            )
            df["unique_id"] = uid
            frames.append(df.reset_index().rename(columns={"index": "time"}))
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.set_index(["time", "unique_id"]).sort_index()
        merged.index.names = ["date", "symbol"]
        return merged

    def _regenerate_factor_cache(self) -> None:
        """使用 KunQuant 提供的 Alpha101 定义重新导出 factor cache。"""

        expressions = extract_alpha101_expressions()
        payload = [
            {
                "name": expr.name,
                "expression": expr.expression,
                "explanation": expr.explanation,
            }
            for expr in expressions
        ]
        ensure_dir(str(self.factor_cache_path.parent))
        with self.factor_cache_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    collector = FactorCollectionDSLNew()
    collector.update_dsl_factors()
