#!/usr/bin/env python
"""扩展 FactorCollection，使用 DSL + KunQuant 计算因子。"""

from __future__ import annotations

import yaml
from pathlib import Path
import pandas as pd

from alphatest.FactorCollection import FactorCollection
from fama.data.factor_space import deserialize_factor_set
from fama.data.kun_backend_new import compute_factor_values_kunquant_new
from copy import deepcopy



class FactorCollectionDSL(FactorCollection):
    """在原有 FactorCollection 基础上增加 DSL 因子计算能力。"""

    def __init__(
        self,
        config_path: str | Path = "/Users/hpy/PycharmProjects/FAMA/fama/config/defaults.yaml",
        target_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        if target_ids:
            self.available_unique_ids = target_ids

    def update_dsl_factors(
        self,
        output_path: str | Path | None = None,
        threads: int | None = None,
    ) -> Path:
        """解析 factor_cache 中的 DSL，调用 KunQuant 计算并写出因子文件。"""

        cfg = self._load_config(self.config_path)
        factor_repo = self._resolve_factor_repo(cfg)
        factor_set = deserialize_factor_set(str(factor_repo))
        if not factor_set.factors:
            raise ValueError("factor_cache 为空，无法计算 DSL 因子。")
        expressions = [factor.expression for factor in factor_set.factors]

        market_df = self._build_market_frame()
        kun_threads = threads if threads is not None else int(cfg.get("compute", {}).get("threads", 4))
        kun_df, fallback = compute_factor_values_kunquant_new(
            market_df,
            expressions,
            threads=kun_threads,
            layout=str(cfg.get("compute", {}).get("layout", "TS")),
        )
        if fallback:
            print(f"[DSL] KunQuant 无法解析 {len(fallback)} 个表达式，将改用 Python 解释器。")
            from fama.data.dataloader import _compute_factor_values_python

            fallback_df = _compute_factor_values_python(market_df, fallback)
            kun_df = pd.concat([kun_df, fallback_df], axis=1).sort_index()

        stacked = kun_df.stack(dropna=False).rename("value").reset_index()
        stacked = stacked.rename(columns={"date": "time", "symbol": "unique_id", "level_2": "factor_tag"})
        expr_to_name = {factor.expression: factor.name for factor in factor_set.factors}
        stacked["factor_tag"] = stacked["factor_tag"].map(expr_to_name).fillna(stacked["factor_tag"])
        stacked = stacked[["time", "unique_id", "factor_tag", "value"]]

        save_path = Path(output_path) if output_path else (self.factor_dir / "dsl_factors.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".csv":
            stacked.to_csv(save_path, index=False)
        else:
            stacked.to_parquet(save_path, index=False)
        print(f"[DSL] 因子计算完成，写入 {save_path}（{len(stacked)} 行）。")
        return save_path

    def _load_config(self, config_path: Path) -> dict:
        with config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    def _resolve_factor_repo(self, cfg: dict) -> Path:
        repo = Path(cfg["paths"]["factor_cache"])
        if repo.is_dir():
            repo = repo / "factors.yaml"
        if not repo.exists():
            raise FileNotFoundError(f"未找到 factor cache: {repo}")
        return repo

    def _build_market_frame(self) -> pd.DataFrame:
        required = {"time", "unique_id", "open", "high", "low", "close", "volume", "amount"}
        missing = required - set(self.native_price.columns)
        if missing:
            raise ValueError(f"行情数据缺少列: {missing}")

        # target_ids = self.available_unique_ids or self.native_price["unique_id"].unique().tolist()
        # native = self.native_price[self.native_price["unique_id"].isin(target_ids)].copy()
        native = deepcopy(self.native_price)

        frames = []
        for uid, frame in native.groupby("unique_id", sort=False):
            df = (
                frame.set_index("time")[["open", "high", "low", "close", "volume", "amount"]]
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


if __name__ == "__main__":
    factor_collection = FactorCollectionDSL()
    factor_collection.update_dsl_factors()
