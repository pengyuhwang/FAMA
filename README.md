# FAMA Prompt-Oriented Skeleton

A Python 3.10 scaffold for **single-run** factor discovery that mirrors the Factor-Augmented Mining Architecture (FAMA). The goal is to let a Large Language Model (LLM) synthesize new alpha expressions from an existing library using:

1. **Cross-Sample Selection (CSS)** – cluster existing factors (low correlation) and surface diverse exemplars.
2. **Chain-of-Experience (CoE)** – maintain experience chains that LLM prompts can extend or split.
3. **Prompt Builder + LLM Client** – build an operator-card-driven prompt and call OpenAI (or a deterministic fallback) under strict field/operator constraints.

When `paths.market_data` points to `data/fof_price_updating.parquet`, the orchestrator直接读取该 Parquet（自动识别 `time`、`unique_id` 等列），否则退回到内置的 OHLCV 模拟器。

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt  # KunQuant / JIT 工具链
python -m fama.cli mine --config fama/config/defaults.yaml
python -m fama.cli mine --config fama/config/defaults.yaml --skip-css   # CoE only
python -m fama.cli mine --config fama/config/defaults.yaml --skip-coe   # CSS only
```
The CLI prints newly generated expressions. Pass `--output ./artifacts/run.yaml` to persist them as YAML.

## Architecture
| Stage | Location | Description |
| ----- | -------- | ----------- |
| Data / Compute | `fama/data/dataloader.py`, `fama/data/kun_backend.py` | `load_market_data` 规整 `(date, symbol)` MultiIndex；`available_factor_inputs` 抽取可用字段；`compute_factor_values` 优先走 KunQuant（TS 布局、多线程执行），KunQuant 失败或关闭时回退到 Python 解释器。 |
| CSS | `fama/css/cluster.py` | `cluster_factors_kmeans` (KMeans clustering) + `select_cross_samples` (pick the factor closest to each cluster centroid). Hyperparameters `k` and `css.n_select` control the number of clusters and the count of chosen representatives. Inputs are normalized via `fama/factors/transforms.py`. |
| CoE | `fama/coe/chain.py` | `build_initial_coe`, `match_coe`, and `extend_or_split_coe` manage experience chains using heuristic scores derived from factor magnitudes. |
| Prompting | `fama/mining/prompt_builder.py` | Reads `prompts/alpha_prompt_template.txt`, fills placeholders `{css_examples}`, `{coe_path}`, `{constraints}`, and parses LLM responses. |
| Orchestrator | `fama/mining/orchestrator.py` | Loads configs, market data, FactorSet cache, runs CSS/CoE if enabled, builds prompts, calls the LLM client, validates responses, and extends the factor library. |
| LLM Client | `fama/mining/llm_client.py` | Replace `_fallback_generation` with your provider call. Until you do, a deterministic pseudo-response keeps tests/CLI working. |
| CLI | `fama/cli.py` | One subcommand `mine` (with `--skip-css`, `--skip-coe`, `--output`). Uses `PromptOrchestrator` under the hood. |
| Persistence | `fama/data/factor_space.py`, `fama/utils/io.py` | FactorSet serialized to YAML (`paths.factor_cache`). Config/outputs also rely on YAML helpers. |

## Data & Factors
- **Production Parquet**：`data/fof_price_updating.parquet`（或自定义路径）会被自动映射成 `(date, symbol)` MultiIndex；若文件缺失，则退回到内置的 OHLCV 模拟器。`pyarrow` 已列在依赖里。
- **Derived Series**：当 `close`/`volume` 存在时自动注入 `RET`、`VWAP`；字段可通过 `llm.deny_fields` 做黑名单控制。
- **KunQuant Backend**：`fama/data/kun_backend.py` 将 `(T,N)` 布局输入喂给 KunQuant JIT，批量执行 Alpha；`compute.use_kunquant=false` 时自动回退到 Python 解释器。
- **Expression DSL / Operators**：白名单覆盖 `RANK/DELTA/TS_MEAN/TS_STDDEV/CORREL/Z_SCORE/SIGN/ABS`，并由语义卡片 + 解析层双重限制，确保 LLM 不会输出未知算子或字段。
- **Seed Library**：项目随发行同步解析 KunQuant `predefined.Alpha101` 中可用的符号（当前 82 条，`alpha001`~`alpha101` 之间的子集），并写入 FactorCache；这些表达式运行时直接走 KunQuant 预置实现。

## Prompt & LLM Integration
1. Populate `.env` from `.env.example`（或设置 `llm.api_key_env` 对应的变量）。
2. `prompt_builder.build_prompt` 会自动：
   - 抽取 CSS/CoE 中出现的算子，并与 `llm.operator_whitelist` 取交集；
   - 渲染 Operator Cards，生成 checksum，并要求 LLM 在首行输出 `OPS-CHECKSUM: xxx`；
   - 注入允许字段列表（`available_fields - deny_fields`），并加上“只用这些字段/算子、输出 N 条”等 Guardrail。
3. 解析阶段再次使用同一白名单和字段集合做语法校验，违规表达式会被丢弃。
4. `fama/mining/llm_client.py` 使用官方 `OpenAI` Chat Completions (`client = OpenAI(...); client.chat.completions.create(...)`)，终端会打印原始输出；若缺少 API Key，则返回受限于字段白名单的确定性伪造结果。

## Configuration Cheatsheet (`fama/config/defaults.yaml`)
| Key | Meaning |
| --- | ------- |
| `k`, `css.n_select`, `u` | CSS 聚类数、每轮保留的簇心代表数、CoE 拆分阈值。 |
| `css.*`, `coe.*` | 其他 CSS/CoE 超参。 |
| `run.use_css`, `run.use_coe` | CLI 未显式指定时的默认开关。 |
| `paths.market_data` | CSV/Parquet 路径；缺失时回退到模拟数据。 |
| `paths.factor_cache` | 初始 FactorSet YAML（启动时解析 KunQuant Alpha101 预置表达式并写入）。 |
| `paths.factor_outputs` | 计算后的因子值保存目录（每个表达式单独一个文件）。 |
| `paths.prompts_dir` / `paths.output_dir` | 模板/产物路径。 |
| `llm.*` | Provider、模型、温度、reasoning、算子白名单、字段黑名单、API key 等。 |
| `compute.use_kunquant` | 是否启用 KunQuant 后端；关闭则强制使用 Python 解释器。 |
| `compute.threads` / `compute.layout` | KunQuant 执行器线程数、输入/输出布局（默认 TS）。 |

## Extending the Scaffold
- **Real Data**: Drop your OHLCV dataset, ensure it has `date` & `symbol` columns, and update `paths.market_data`.
- **More DSL Features**: Expand `_ALLOWED_FUNCTIONS` / `_ALLOWED_VARIABLES` in `alpha_lib.py`.
- **Better Scoring**: Replace the heuristic CoE scores with RankIC/RankICIR once you add evaluation modules.
- **LLM Integration**: In `request_new_factors`, swap `_fallback_generation` for actual API calls (OpenAI, Anthropic, Azure, etc.). Continue to reuse `parse_llm_output` for formatting.
- **Persistence & Experiments**: Use `serialize_factor_set` outputs as artifacts and version them however you prefer.

## Tests
`pytest` exercises the primary modules using the simulator and fallback LLM:
```bash
pytest
```

## Roadmap
- Wire RankIC/RankICIR computation + backtest metrics inside a new `eval/` package.
- Flesh out multi-round mining and early stopping logic.
- Integrate experiment tracking / dashboards.
- Replace heuristic CoE management with production logic (graph DB, metadata stores, etc.).
