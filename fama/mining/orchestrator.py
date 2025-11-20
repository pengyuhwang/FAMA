"""负责在单次运行中协调 CSS、CoE 以及提示词构建的编排器。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd

import numpy as np
from dotenv import load_dotenv

from fama.css.cluster import cluster_factors_kmeans, select_cross_samples
from fama.data.dataloader import (
    available_factor_inputs,
    compute_factor_values,
    load_market_data,
)
from fama.data.factor_space import Factor, FactorSet, deserialize_factor_set, serialize_factor_set
from fama.factors.alpha101_extractor import extract_alpha101_expressions
from fama.factors.alpha_lib import list_alpha101_tokens, list_seed_alphas, validate_alpha_syntax
from fama.mining import prompt_builder
from fama.mining.llm_client import request_new_factors
from fama.coe.manager import CoEManager
from fama.utils.io import ensure_dir
from fama.utils.logging import get_logger
from fama.utils.timers import Timer


class PromptOrchestrator:
    """根据 README 描述协调 CSS、CoE 以及 LLM 调用。"""

    def __init__(self, cfg: dict):
        """存储 CSS/CoE/提示词相关配置。

        Args:
            cfg: defaults.yaml 及其覆盖项合并后的配置字典。
        """

        self.cfg = cfg
        load_dotenv()
        self.logger = get_logger(__name__)
        self.market_data = load_market_data(cfg["paths"]["market_data"])
        self.llm_cfg = cfg.get("llm", {})
        self.coe_cfg = cfg.get("coe", {})
        deny_fields = {field.upper() for field in self.llm_cfg.get("deny_fields", [])}
        self.available_fields = available_factor_inputs(self.market_data)
        if not self.available_fields:
            raise ValueError("未检测到可用的数值字段，无法构建因子。")
        self.prompt_allowed_fields = sorted([f for f in self.available_fields if f not in deny_fields]) or self.available_fields
        self.allowed_variables = set(self.prompt_allowed_fields)
        self.allowed_ops = set(op.upper() for op in self.llm_cfg.get("operator_whitelist", []))
        self.logger.info("可用字段: %s", ", ".join(self.available_fields))
        if deny_fields:
            self.logger.info("生效字段（过滤 deny 列表后）: %s", ", ".join(self.prompt_allowed_fields))
        self.factor_repo = self._resolve_factor_repo()
        self.factor_output_dir = Path(self.cfg["paths"].get("factor_outputs", "./data/factor_values"))
        ensure_dir(str(self.factor_output_dir))
        self.factor_set = self._load_factor_set()
        self._sanitize_factor_set()
        self.factor_frame = compute_factor_values(
            self.market_data,
            [factor.expression for factor in self.factor_set.factors],
            cfg=self.cfg,
        )
        self.coe_manager = CoEManager()
        self.coe_manager.attach_logger(self.logger)
        self.forward_returns = self._compute_forward_returns(self.market_data)
        self.coe_manager.set_forward_returns(self.forward_returns)
        benchmark_assets = self.coe_cfg.get("benchmark_assets") or []
        self.coe_manager.benchmark_assets = benchmark_assets
        max_depth = self.coe_cfg.get("max_depth")
        if max_depth is not None:
            self.coe_manager.max_depth = max_depth
        min_rankic = self.coe_cfg.get("min_rankic")
        if min_rankic is not None:
            self.coe_manager.min_rankic = min_rankic
        prompt_chains = self.coe_cfg.get("prompt_chains")
        if prompt_chains is not None:
            self.coe_manager.prompt_chains = prompt_chains
        prompt_expr_chars = self.coe_cfg.get("prompt_expr_chars")
        if prompt_expr_chars is not None:
            self.coe_manager.prompt_expr_chars = prompt_expr_chars
        ric_start = self.coe_cfg.get("ric_start_date")
        ric_end = self.coe_cfg.get("ric_end_date")
        if ric_start:
            self.coe_manager.ric_start = pd.to_datetime(ric_start)
        if ric_end:
            self.coe_manager.ric_end = pd.to_datetime(ric_end)
        ric_path = Path(self.cfg["paths"].get("factor_ric", "factor_value_prepared/factor_ric.csv"))
        if ric_path.exists():
            try:
                ric_df = pd.read_csv(ric_path)
                self.coe_manager.set_precomputed_ric(ric_df)
                self.logger.info("Loaded precomputed RIC from %s", ric_path)
            except Exception as exc:
                self.logger.warning("Failed to load precomputed RIC (%s); will compute on the fly.", exc)
        else:
            self.logger.info("Precomputed RIC file %s not found; will compute on the fly.", ric_path)

    def run(self, use_css: bool = True, use_coe: bool = True) -> list[str]:
        """依据 CSS/CoE 开关执行一次挖掘流程。"""

        self.logger.info("Starting PromptOrchestrator run | CSS=%s | CoE=%s", use_css, use_coe)
        css_examples = self.prepare_css_context(self.factor_set) if use_css else []
        coe_examples = self.prepare_coe_context(self.factor_set) if use_coe else []
        prompt = self.build_prompt(css_examples, coe_examples)
        self.logger.info("LLM prompt payload:\n%s", prompt)
        expressions = self.call_llm(prompt)
        if expressions:
            self._update_factor_set(expressions)
        return expressions

    def prepare_css_context(self, factors: Optional["FactorSet"] = None) -> list[str]:
        """按照 README “CSS Context Assembly” 章节准备示例。"""

        factors = factors or self.factor_set
        if not factors.factors or self.factor_frame.empty:
            return []

        matrix = self.factor_frame.to_numpy(dtype=float)
        clusters, centers, norm_matrix = cluster_factors_kmeans(matrix, self.cfg.get("k", 8))
        if not clusters:
            self.logger.warning("CSS clustering produced no clusters; falling back to sequential order.")
            clusters = [list(range(len(factors.factors)))]
            norm_matrix = matrix
            centers = np.array([matrix.mean(axis=1)]) if matrix.size else np.zeros((0, matrix.shape[0]))
        cluster_sizes = [len(cluster) for cluster in clusters]
        self.logger.info("CSS formed %d clusters | sizes=%s", len(clusters), cluster_sizes)
        css_cfg = self.cfg.get("css", {})
        n_select = css_cfg.get("n_select", 16)
        seed = css_cfg.get("seed")
        self.logger.info("CSS selecting %d diversified context samples", n_select)
        selections = select_cross_samples(clusters, n_select, seed=seed)
        self.logger.info("CSS selected factor indices: %s", selections)
        self.coe_manager.rebuild_from_clusters(self.factor_set, self.factor_frame, clusters)
        css_examples = []
        selected_pairs: list[str] = []
        for idx in selections:
            if idx < len(factors.factors):
                factor = factors.factors[idx]
                if factor.explanation:
                    css_examples.append(f"{factor.expression}  # 说明: {factor.explanation}")
                else:
                    css_examples.append(factor.expression)
                selected_pairs.append(f"{factor.name}: {factor.expression}")
        preview = ", ".join(css_examples[:5])
        self.logger.info("CSS exemplar preview: %s", preview if preview else "None")
        if selected_pairs:
            self.logger.info("CSS selected factors: %s", " | ".join(selected_pairs))
        return css_examples

    def prepare_coe_context(self, factors: Optional["FactorSet"] = None) -> list[str]:
        """根据 README “CoE Context Assembly” 章节构造经验链。"""

        if not self.coe_manager.chains:
            matrix = self.factor_frame.to_numpy(dtype=float)
            clusters, _, _ = cluster_factors_kmeans(matrix, self.cfg.get("k", 8))
            self.coe_manager.rebuild_from_clusters(self.factor_set, self.factor_frame, clusters)

        coe_lines = self.coe_manager.format_top_chains()
        if coe_lines:
            self.logger.info("CoE formatted %d chains for prompt.", len(coe_lines))
        return coe_lines

    def build_prompt(
        self,
        css_examples: list[str],
        coe_examples: list[str],
    ) -> str:
        """将约束注入 prompt_builder 并返回提示词。"""

        constraints = self.llm_cfg.copy()
        return prompt_builder.build_prompt(
            css_examples,
            coe_examples,
            constraints,
            available_fields=self.prompt_allowed_fields,
        )

    def call_llm(self, prompt: str) -> list[str]:
        """使用环境变量中的凭据调用 LLM 客户端。"""

        llm_cfg = self.llm_cfg
        api_key_env = llm_cfg.get("api_key_env", "LLM_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            api_key = llm_cfg.get("api_key")
        provider = llm_cfg.get("provider", "mock")
        model = llm_cfg.get("model", "mock")
        temperature = llm_cfg.get("temperature")
        thinking = llm_cfg.get("thinking")
        if not api_key:
            self.logger.info(
                "Environment variable %s not set; using deterministic fallback LLM output.",
                api_key_env,
            )
        with Timer("llm_call"):
            return request_new_factors(
                prompt,
                provider,
                model,
                api_key=api_key,
                temperature=temperature,
                thinking=thinking,
                allowed_fields=self.prompt_allowed_fields,
                logger=self.logger,
            )

    def _resolve_factor_repo(self) -> Path:
        path = Path(self.cfg["paths"].get("factor_cache", "./data/factor_cache"))
        if path.is_dir():
            ensure_dir(str(path))
            return path / "factors.yaml"
        ensure_dir(str(path.parent))
        return path

    def _load_factor_set(self) -> FactorSet:
        if self.factor_repo.exists():
            return deserialize_factor_set(str(self.factor_repo))
        try:
            expressions = extract_alpha101_expressions()
            factors = [Factor(name=item.name, expression=item.expression) for item in expressions]
        except Exception:
            seed_exprs = self._bootstrap_seed_expressions()
            factors = [
                Factor(name=f"seed_{idx+1}", expression=expr)
                for idx, expr in enumerate(seed_exprs)
            ]
        factor_set = FactorSet(factors)
        serialize_factor_set(factor_set, str(self.factor_repo))
        return factor_set

    def _update_factor_set(self, expressions: list[dict]) -> None:
        start_idx = len(self.factor_set.factors)
        accepted_factors: list[Factor] = []
        for offset, item in enumerate(expressions):
            expr = (item.get("expression") if isinstance(item, dict) else None) or ""
            expr = expr.strip()
            if not validate_alpha_syntax(
                expr,
                self.allowed_variables,
                allowed_ops=self.allowed_ops,
            ):
                self.logger.warning("Skipping invalid expression: %s", expr)
                continue
            name = f"LLM_Factor{start_idx + offset + 1}"
            explanation = None
            if isinstance(item, dict):
                expl = item.get("explanation")
                if isinstance(expl, str):
                    explanation = expl.strip() or None
            factor_obj = Factor(name=name, expression=expr, explanation=explanation)
            self.factor_set.factors.append(factor_obj)
            accepted_factors.append(factor_obj)
        serialize_factor_set(self.factor_set, str(self.factor_repo))
        self.factor_frame = compute_factor_values(
            self.market_data,
            [factor.expression for factor in self.factor_set.factors],
            cfg=self.cfg,
        )
        if accepted_factors:
            self._persist_factor_series(accepted_factors)

    def _bootstrap_seed_expressions(self) -> list[str]:
        candidates = list_alpha101_tokens() or list_seed_alphas()
        seeds = [
            expr
            for expr in candidates
            if validate_alpha_syntax(expr, self.allowed_variables, allowed_ops=self.allowed_ops)
        ]
        if seeds:
            return seeds
        fallback = [f"RANK({field})" for field in self.available_fields[:5]]
        return fallback or ["RANK(VWAP)"]

    def _sanitize_factor_set(self) -> None:
        valid = [
            factor
            for factor in self.factor_set.factors
            if validate_alpha_syntax(
                factor.expression,
                self.allowed_variables,
                allowed_ops=None,
            )
        ]
        if len(valid) != len(self.factor_set.factors):
            self.logger.warning("检测到不兼容的因子，已自动清理。")
            self.factor_set = FactorSet(valid or [
                Factor(name=f"seed_{idx+1}", expression=expr)
                for idx, expr in enumerate(self._bootstrap_seed_expressions())
            ])
            serialize_factor_set(self.factor_set, str(self.factor_repo))

    def _persist_factor_series(self, factors: list[Factor]) -> None:
        if not factors:
            return
        expressions = [factor.expression for factor in factors]
        try:
            df = compute_factor_values(self.market_data, expressions, cfg=self.cfg)
        except Exception as exc:
            self.logger.warning("因子数值计算失败，未写入文件：%s", exc)
            return
        if df.empty:
            self.logger.info("因子计算结果为空，跳过持久化。")
            return
        rows = []
        for factor in factors:
            expr = factor.expression
            if expr not in df.columns:
                self.logger.warning("表达式 %s 未生成数据，跳过写入。", expr)
                continue
            series = df[expr].rename("value").reset_index()
            series = series.rename(columns={"date": "time", "symbol": "unique_id"})
            series["factor_tag"] = factor.name
            rows.append(series)
        if not rows:
            return
        merged = pd.concat(rows, ignore_index=True)[["time", "unique_id", "factor_tag", "value"]]
        out_path = self.factor_output_dir / "llm_factors.csv"
        ensure_dir(str(out_path.parent))
        if out_path.exists():
            merged_prev = pd.read_csv(out_path)
            merged = pd.concat([merged_prev, merged], ignore_index=True)
        merged.sort_values(["time", "unique_id", "factor_tag"], inplace=True)
        merged.to_csv(out_path, index=False)
        self.logger.info("LLM 因子写入 %s（新增 %d 条）。", out_path, len(rows))

    def _compute_forward_returns(self, market_data: pd.DataFrame) -> pd.Series | None:
        if "close" not in market_data.columns:
            self.logger.warning("市场数据缺少 close 列，无法计算 RankIC。")
            return None
        close_series = market_data["close"]
        returns = close_series.groupby(level=1).pct_change().shift(-1)
        returns.name = "forward_return"
        return returns
