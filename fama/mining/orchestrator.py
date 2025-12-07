"""负责在单次运行中协调 CSS、CoE 以及提示词构建的编排器。"""

from __future__ import annotations

import os
import re
import runpy
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
from fama.factors.alpha_lib import (
    list_alpha101_tokens,
    list_seed_alphas,
    validate_alpha_syntax,
    validate_alpha_syntax_strict,
)
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
        self.project_root = Path(__file__).resolve().parents[2]
        self.logger = get_logger(__name__)
        self.market_data = load_market_data(cfg["paths"]["market_data"])
        self.llm_cfg = cfg.get("llm", {})
        self.coe_cfg = cfg.get("coe", {})
        deny_fields = {field.upper() for field in self.llm_cfg.get("deny_fields", [])}
        self.available_fields = available_factor_inputs(self.market_data)
        # Ensure AMOUNT is kept if present in data but missed by dtype checks
        if "AMOUNT" not in self.available_fields and any(c.lower() == "amount" for c in self.market_data.columns):
            self.available_fields.append("AMOUNT")
            self.available_fields.sort()
        if not self.available_fields:
            raise ValueError("未检测到可用的数值字段，无法构建因子。")
        prompt_fields = sorted([f for f in self.available_fields if f not in deny_fields]) or self.available_fields
        if "AMOUNT" not in prompt_fields and "AMOUNT" in self.available_fields:
            prompt_fields.append("AMOUNT")
            prompt_fields.sort()
        self.prompt_allowed_fields = prompt_fields
        self.allowed_variables = set(self.prompt_allowed_fields)
        self.allowed_ops = set(op.upper() for op in self.llm_cfg.get("operator_whitelist", []))
        self.logger.info("可用字段: %s", ", ".join(self.available_fields))
        if deny_fields:
            self.logger.info("生效字段（过滤 deny 列表后）: %s", ", ".join(self.prompt_allowed_fields))
        self.factor_repo = self._resolve_factor_repo()
        self.llm_factor_repo = self._resolve_llm_factor_repo()
        self.factor_output_dir = Path(self.cfg["paths"].get("factor_outputs", "./data/factor_values"))
        ensure_dir(str(self.factor_output_dir))
        self.factor_set = self._load_factor_set()
        self.llm_factor_set = self._load_llm_factor_set()
        self._sanitize_factor_sets()
        self.base_factor_count = len(self.factor_set.factors)
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
        metrics_dir_cfg = self.cfg["paths"].get("factor_metrics_dir")
        default_metrics = self.project_root / "factor_value_prepared" / "data" / "factors"
        self.factor_metrics_dir = self._resolve_project_path(Path(metrics_dir_cfg)) if metrics_dir_cfg else default_metrics
        ric_cfg_path = self.cfg["paths"].get("factor_ric")
        default_ric_path = self.factor_metrics_dir / "factor_ric.csv"
        self.ric_output_path = (
            self._resolve_project_path(Path(ric_cfg_path)) if ric_cfg_path else default_ric_path
        )
        self._load_precomputed_ric_files(log_missing=True)

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
        # 根据预计算 RIC 构建得分向量（按 factor 顺序对齐）
        ric_scores = None
        precomputed_ric = getattr(self.coe_manager, "_precomputed_ric", None)
        if precomputed_ric is not None:
            ric_map = (
                precomputed_ric.groupby("factor_tag")["ric"].max()
                if "factor_tag" in precomputed_ric.columns
                else None
            )
            if ric_map is not None:
                ric_scores = [ric_map.get(factor.name) for factor in factors.factors]

        self.logger.info("CSS selecting %d diversified context samples", n_select)
        selections = select_cross_samples(clusters, n_select, seed=seed, ric_scores=ric_scores)
        self.logger.info("CSS selected factor indices: %s", selections)
        if clusters:
            self._execute_factor_metrics_scripts()
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
            if clusters:
                self._execute_factor_metrics_scripts()
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
                base_url=llm_cfg.get("base_url"),
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

    def _resolve_llm_factor_repo(self) -> Path:
        path = Path(self.cfg["paths"].get("llm_factor_cache", "./data/factor_cache_new/LLM_factors.yaml"))
        if path.is_dir():
            ensure_dir(str(path))
            return path / "LLM_factors.yaml"
        ensure_dir(str(path.parent))
        return path

    def _resolve_project_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _execute_factor_metrics_scripts(self) -> None:
        modules = [
            ("factor_value_prepared.compute_factor_values", "计算因子值"),
            ("factor_value_prepared.compute_ric", "计算 RankIC"),
        ]
        for module, desc in modules:
            self.logger.info("运行 %s（%s）...", module, desc)
            try:
                runpy.run_module(module, run_name="__main__")
            except Exception:
                self.logger.exception("执行 %s 失败，终止 CoE 构建。", module)
                raise
        if not self._load_precomputed_ric_files(log_missing=False):
            self.logger.warning("脚本执行完成但未找到最新的 RIC 文件，CoE 将无法使用预计算分数。")

    def _load_precomputed_ric_files(self, log_missing: bool = True) -> bool:
        candidates: list[Path] = []
        if self.ric_output_path:
            candidates.append(self.ric_output_path)
        default_candidate = self.factor_metrics_dir / "factor_ric.csv"
        if not candidates or default_candidate not in candidates:
            candidates.append(default_candidate)
        loaded = False
        for path in candidates:
            if not path.exists():
                continue
            try:
                ric_df = pd.read_csv(path)
            except Exception as exc:
                self.logger.warning("读取预计算 RIC 失败（%s）：%s", path, exc)
                continue
            self.coe_manager.set_precomputed_ric(ric_df)
            self.logger.info("已加载预计算 RIC：%s", path)
            loaded = True
            break
        if not loaded:
            if log_missing:
                joined = ", ".join(str(p) for p in candidates)
                self.logger.info("未找到预计算 RIC 文件，尝试路径：%s", joined)
            self.coe_manager.set_precomputed_ric(None)
        return loaded

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

    def _load_llm_factor_set(self) -> FactorSet:
        if self.llm_factor_repo.exists():
            return deserialize_factor_set(str(self.llm_factor_repo))
        ensure_dir(str(self.llm_factor_repo.parent))
        serialize_factor_set(FactorSet([]), str(self.llm_factor_repo))
        return FactorSet([])

    def _update_factor_set(self, expressions: list[dict]) -> None:
        pattern = re.compile(r"LLM_Factor(\d+)$")
        existing_suffixes = {
            int(match.group(1))
            for factor in self.llm_factor_set.factors
            for match in [pattern.match(factor.name)]
            if match
        }
        counter = max(existing_suffixes, default=self.base_factor_count)
        used_suffixes = set(existing_suffixes)
        accepted_factors: list[Factor] = []
        for item in expressions:
            expr = (item.get("expression") if isinstance(item, dict) else None) or ""
            expr = expr.strip()
            ok, reason = validate_alpha_syntax_strict(
                expr,
                self.allowed_variables,
                allowed_ops=self.allowed_ops,
            )
            if not ok:
                self.logger.warning("Skipping invalid expression: %s，原因: %s", expr, reason)
                continue
            counter += 1
            while counter in used_suffixes:
                counter += 1
            used_suffixes.add(counter)
            name = f"LLM_Factor{counter}"
            explanation = None
            if isinstance(item, dict):
                expl = item.get("explanation")
                if isinstance(expl, str):
                    explanation = expl.strip() or None
            factor_obj = Factor(name=name, expression=expr, explanation=explanation)
            self.llm_factor_set.factors.append(factor_obj)
            accepted_factors.append(factor_obj)
        if accepted_factors:
            serialize_factor_set(self.llm_factor_set, str(self.llm_factor_repo))
        # self.factor_frame = compute_factor_values(
        #     self.market_data,
        #     [factor.expression for factor in self.factor_set.factors],
        #     cfg=self.cfg,
        # )
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

    def _sanitize_factor_sets(self) -> None:
        def _sanitize(factors: list[Factor]) -> list[Factor]:
            cleaned = []
            for factor in factors:
                ok, reason = validate_alpha_syntax_strict(
                    factor.expression,
                    self.allowed_variables,
                    allowed_ops=None,
                )
                if ok:
                    cleaned.append(factor)
                else:
                    self.logger.warning("移除不兼容表达式: %s，原因: %s", factor.expression, reason)
            return cleaned

        base_valid = _sanitize(self.factor_set.factors)
        if len(base_valid) != len(self.factor_set.factors):
            self.logger.warning("基础因子库含有不兼容表达式，已自动清理。")
        if not base_valid:
            base_valid = [
                Factor(name=f"seed_{idx+1}", expression=expr)
                for idx, expr in enumerate(self._bootstrap_seed_expressions())
            ]
        self.factor_set = FactorSet(base_valid)
        serialize_factor_set(self.factor_set, str(self.factor_repo))
        self.base_factor_count = len(self.factor_set.factors)

        llm_valid = _sanitize(self.llm_factor_set.factors)
        if len(llm_valid) != len(self.llm_factor_set.factors):
            self.logger.warning("LLM 因子库含有不兼容表达式，已自动清理。")
        self.llm_factor_set = FactorSet(llm_valid)
        serialize_factor_set(self.llm_factor_set, str(self.llm_factor_repo))

    def _persist_factor_series(self, factors: list[Factor]) -> None:
        # 已禁用 CSV 持久化，避免重复存储。如需启用，请恢复原实现。
        return

    def _compute_forward_returns(self, market_data: pd.DataFrame) -> pd.Series | None:
        if "close" not in market_data.columns:
            self.logger.warning("市场数据缺少 close 列，无法计算 RankIC。")
            return None
        close_series = market_data["close"]
        returns = close_series.groupby(level=1).pct_change().shift(-1)
        returns.name = "forward_return"
        return returns
