# FAMA Prompt-Oriented Skeleton

A Python 3.10 scaffold for **single-run** factor discovery that mirrors the Factor-Augmented Mining Architecture (FAMA). The goal is to let a Large Language Model (LLM) synthesize new alpha expressions from an existing library using:

1. **Cross-Sample Selection (CSS)** – cluster existing factors (low correlation) and surface diverse exemplars.
2. **Chain-of-Experience (CoE)** – maintain experience chains that LLM prompts can extend or split.
3. **Prompt Builder + LLM Client** – render a configurable template, then invoke your preferred provider. A deterministic fallback keeps the pipeline executable until you wire a real API.

Synthetic OHLCV data (Alpha101-ish) is generated automatically so the entire stack can run without external datasets.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m fama.cli mine --config fama/config/defaults.yaml
python -m fama.cli mine --config fama/config/defaults.yaml --skip-css   # CoE only
python -m fama.cli mine --config fama/config/defaults.yaml --skip-coe   # CSS only
```
The CLI prints newly generated expressions. Pass `--output ./artifacts/run.yaml` to persist them as YAML.

## Architecture
| Stage | Location | Description |
| ----- | -------- | ----------- |
| Data Simulation | `fama/data/dataloader.py` | `load_market_data` returns a `(date, symbol)` indexed OHLCV frame, simulating data if files are absent. `compute_factor_values` evaluates Alpha101-style expressions through a safe AST interpreter. |
| CSS | `fama/css/cluster.py` | `cluster_factors_kmeans` (KMeans clustering) + `select_cross_samples` (diverse representatives). Inputs are normalized via `fama/factors/transforms.py`. |
| CoE | `fama/coe/chain.py` | `build_initial_coe`, `match_coe`, and `extend_or_split_coe` manage experience chains using heuristic scores derived from factor magnitudes. |
| Prompting | `fama/mining/prompt_builder.py` | Reads `prompts/alpha_prompt_template.txt`, fills placeholders `{css_examples}`, `{coe_path}`, `{constraints}`, and parses LLM responses. |
| Orchestrator | `fama/mining/orchestrator.py` | Loads configs, market data, FactorSet cache, runs CSS/CoE if enabled, builds prompts, calls the LLM client, validates responses, and extends the factor library. |
| LLM Client | `fama/mining/llm_client.py` | Replace `_fallback_generation` with your provider call. Until you do, a deterministic pseudo-response keeps tests/CLI working. |
| CLI | `fama/cli.py` | One subcommand `mine` (with `--skip-css`, `--skip-coe`, `--output`). Uses `PromptOrchestrator` under the hood. |
| Persistence | `fama/data/factor_space.py`, `fama/utils/io.py` | FactorSet serialized to YAML (`paths.factor_cache`). Config/outputs also rely on YAML helpers. |

## Data & Factors
- **Synthetic OHLCV**: Generated if `paths.market_data` does not exist. Columns: `open`, `high`, `low`, `close`, `volume`, multi-indexed by `(date, symbol)`.
- **Derived Series**: `RET` (symbol-level returns) and `VWAP` (weighted price) are injected into the expression context.
- **Expression DSL**: Supported functions include `RANK`, `DELTA`, `TS_MEAN`, `TS_STDDEV`, `CORREL`, `Z_SCORE`, `SIGN`, `ABS`. Variables: `OPEN/HIGH/LOW/CLOSE/VOLUME/RET/VWAP`. Extend `fama/factors/alpha_lib.py` if your DSL requires more.
- **Seed Library**: `list_seed_alphas` provides canonical Alpha101-style expressions to bootstrap CSS/CoE when no cache exists.

## Prompt & LLM Integration
1. Populate `.env` from `.env.example` (default key: `LLM_API_KEY`).
2. Customize `prompts/alpha_prompt_template.txt` or supply another template via `llm.instructions_path`.
3. `PromptOrchestrator.build_prompt` collects:
   - CSS exemplars (if enabled) joined as bullet points.
   - CoE path (if enabled) as an ordered chain.
   - Constraints payload (`llm.*` config minus `instructions_path`).
4. `request_new_factors` currently falls back to deterministic “fake” responses when no API key is found. Replace the `NotImplementedError` branch with a real SDK call, then feed the raw response through `parse_llm_output`.

### Example Template Snippet
```text
# FAMA Alpha Prompt Template
- CSS exemplars:
{css_examples}
- Chain-of-Experience path:
{coe_path}
- Guardrail constraints:
{constraints}
```

## Configuration Cheatsheet (`fama/config/defaults.yaml`)
| Key | Meaning |
| --- | ------- |
| `k`, `l`, `u` | CSS cluster count, CSS selection count, CoE split threshold guard. |
| `css.*`, `coe.*` | Extra knobs controlling sample counts, depth, edge thresholds. |
| `run.use_css`, `run.use_coe` | Defaults when CLI flags are omitted. |
| `paths.market_data` | CSV/Parquet path; triggers simulator if missing. |
| `paths.factor_cache` | File (or directory) storing serialized FactorSets. |
| `paths.prompts_dir` | Used by prompt builder to find templates. |
| `paths.output_dir` | Where CLI writes YAML results when `--output` is provided. |
| `llm.*` | Provider, model, temperature, max factors, template path, env var name. |

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
