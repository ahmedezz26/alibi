# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project: Alibi — agent-trace triage. Treats a trace as a time series and windowed judge-model calls as noisy sensor readings, then applies state-estimation techniques (forward filtering, CUSUM drift detection, backward/RTS smoothing) to localize the step where an agent went wrong.

**Current state (2026-09-20): research is done, the product is being built.** V2 + Jev is the adopted method; V3-V9 all failed their pre-registered gates. The product surface is `alibi.localize.diagnose()`, reached through `alibi diagnose`, the `alibi-mcp` MCP server and the Claude Code plugin in `plugin/`. Building and testing need no paid calls.

**Start every session by reading `docs/research-log.md`.** It holds every measurement, the pre-registered pass bars, data locations and the next action. The project thesis, architecture rationale, data strategy and open risks are in `docs/PROJECT_BRIEF.md`; read it before any architectural decision. This file only holds what must survive every session.

## Critical rules — do not skip

- **Paid APIs:** never run a paid model without explicit approval and a cost estimate; `make_judge` refuses non-free models unless `ALIBI_ALLOW_PAID_MODELS=1`. For frontier-model references, use **published paper numbers**, don't run them.
- **Go/no-go:** step 6 **passed on 2026-09-19** (see `docs/research-log.md`), so steps 7–10 are unblocked, but agree each one with the user first. Step 6 is validation of the backward-smoothing pass with recorded precision/recall. It started on AgentRx (73 annotated failures in the public `microsoft/AgentRx` release) and continues on AgenTracer's long traces; status is in `docs/research-log.md`. See `docs/PROJECT_BRIEF.md` §5 for the build order.
- Tune thresholds on the AgenTracer **train** split only; report on **test**. Don't tune prompts on test or pilot traces. Don't diagnose misses on test either: find patterns on train.
- **Locked held-out sets:** `docs/heldout_agentracer_test_ids.json` (126, used), `docs/heldout_trajerrbench_swe56_ids.json` (used) and `docs/heldout_longrca_ids.json` (LongRCA; its SWE-90 part is used, the rest unused). Run each once, for a final verdict, against a pass bar agreed in writing beforehand. Never inspect its misses or tune on it.
- Free OpenRouter endpoints may log prompts. Use them only on public benchmarks, **never** on customer or production traces.
- Don't commit or push unless asked.

## Build / test / lint

Use `uv` for everything (Python 3.12, src layout under `src/alibi/`).

```bash
uv sync                                              # install deps (incl. dev group)
uv add <pkg>            / uv add --dev <pkg>         # add a dependency
uv run pytest                                        # full suite (95 tests, all offline fakes)
uv run pytest tests/test_chunking.py::test_empty_trace   # single test
uv run ruff check . && uv run ruff format .          # lint + format
uv run alibi score examples/sample_trace.json        # full pipeline on one trace
uv run alibi eval --benchmark agentracer --split test --min-tokens 10000 --mode windowed --workers 8 --out runs/X.jsonl
```

`alibi eval` (alias `eval-agentrx`): `--benchmark agentrx|agentracer|trajerrbench|longrca`, `--mode hybrid|whole|windowed`, `--subset`, `--split`, `--min-tokens`, `--sample N`, `--limit N`, `--no-state`. Output is resumable JSONL in `runs/` (git-ignored); errored rows are retried on resume. Datasets live in git-ignored `data/` (AgentRx and TRAIL are gated and need `HF_TOKEN`; TRAIL must not be reshared).

## Architecture (pipeline, per `docs/PROJECT_BRIEF.md` §3)

**Hybrid routing:**
- **Short traces** (≤ `ALIBI_JUDGE_RELIABLE_TOKENS`): one whole-trace judge call.
- **Long traces:** `TraceSource` → `chunk_trace` (overlapping windows of `ALIBI_JUDGE_WINDOW_TOKENS`) → `forward_pass` (the judge carries a compressed running summary between chunks) → CUSUM (`drift.py`) over the per-chunk anomaly scores → `backward_pass` (RTS-style: re-examines chunks N..0 in parallel with hindsight). The anchor is the first alarm, else the top chunk, else the last chunk when the run is known to have failed.
- **Why routing exists:** single calls collapse past ~10K tokens.
- **The bet:** the backward pass is the primary bet. The estimation layer is the moat, not the judge model.

**Code map (`src/alibi/`):**
- **Core:** `types.py` (`Step`, `Chunk`, `Question`, `Annotation`), `config.py` (all env/settings), `chunking.py`, `forward.py`, `drift.py`, `backward.py`.
- **Adapters:**
  - `sources/`: the `TraceSource` protocol, plus `langsmith`, `jsonfile`, `agentrx`, `agentracer`, `trajerrbench` and `longrca` adapters.
  - `judges/`: the `Judge` protocol, `OpenRouterJudge` and `TypeSafeJudge` (Jev via `langchain-typesafe`). `judges/__init__.py:make_judge` is the only place that imports a concrete backend.
- **Entry points:** `evaluate.py` (benchmark harness) and `cli.py`.

## Coding conventions

- Judge model access always goes through the pluggable `Judge` interface (`docs/PROJECT_BRIEF.md` §3.3). Never import a specific judge-model implementation directly anywhere else in the codebase.
- Trace ingestion always goes through a `TraceSource` adapter (`docs/PROJECT_BRIEF.md` §3.1). Don't hardcode LangSmith-specific (or dataset-specific) calls outside the adapter.
- Free-form LLM judges run through OpenRouter (`OpenRouterJudge`, `openai` SDK with `base_url="https://openrouter.ai/api/v1"`, strict `json_schema`). The one exception, approved by the user on 2026-09-18, is TypeSafe's Jev, which runs through `TypeSafeJudge` (`langchain-typesafe`). Model IDs, window sizes and thresholds come only from `config.py`; never hardcode them elsewhere.
- Log cost/latency per call (`judge.calls`) and per run, so backend swaps stay comparable (§7).
- Storage during prototyping is local JSON/JSONL/parquet, with no database (§6).

## Environment / Tooling

Env vars live in a git-ignored `.env` (template: `.env.example`):
- `OPENROUTER_API_KEY`, `LANGSMITH_API_KEY`, and `HF_TOKEN` (gated datasets).
- `ALIBI_JUDGE_MODEL`: default **`deepseek/deepseek-v4-flash-0731:free`** (free). The fallback free model is `nvidia/nemotron-3-super-120b-a12b:free`.
- `ALIBI_ALLOW_PAID_MODELS`: the spend guard, default off (see Critical rules).
- `ALIBI_JUDGE_REASONING`: `off|low|medium|high`, empty = model default. Use `off`: DeepSeek is ~80× faster (4.8s vs ~400s per call) at similar accuracy.
- `ALIBI_JUDGE_MAX_RETRIES`: default 8. Free models have per-minute limits and **1000 requests/day** (check `GET /api/v1/key` → `free_model_daily_requests`).
- `ALIBI_JUDGE_RELIABLE_TOKENS`: the routing threshold, default 10000 (measured for both nano and DeepSeek on AgentRx). It's a property of the judge: re-measure when the model changes.
- `ALIBI_JUDGE_WINDOW_TOKENS`: the window size for long traces; defaults to the reliable length. Never hardcode a window size.
- `ALIBI_JUDGE_BACKEND`: `openrouter` (default) or `typesafe` (Jev, paid, max input ~32K tokens). `TYPESAFE_API_KEY`, `ALIBI_TYPESAFE_MODEL` (default `jev-latest`). Jev can only answer Noul/Score/Choice, so it uses the **typed pipeline** (`alibi eval --pipeline typed --mode windowed`): a memory card of probabilities instead of a written summary, per-step Nouls in the look-back, and pick = P(chunk)×P(step).

MCP servers (LangSmith, Hugging Face, Context7, alphaxiv) are configured in `.mcp.json` at the repo root; approve them on first use. GitHub MCP needs a personal access token, so it isn't in `.mcp.json`. Add it locally once a real remote exists:
```bash
claude mcp add -s user --transport http github https://api.githubcopilot.com/mcp -H "Authorization: Bearer YOUR_PAT"
```

## Progress Log

Keep the detailed state in **`docs/research-log.md`**: update its TL;DR, results and next steps, and append dated entries to its §8. Keep only a one-line pointer here.

- 2026-09-18: steps 1–5 and the backward pass are built. AgentRx: the pipeline loses to a single call on short traces. AgenTracer: the 30-trace pilot win did not hold. On 120 long test traces the pipeline **loses** (±3 48% vs 56% for a single call, p=0.23). Step 6 is a no-go as built; the next step is diagnosis on train. Details are in `docs/research-log.md`.
- 2026-09-19: **Step 6 PASSED** on the locked held-out set (126 unseen long AgenTracer test traces), by the pre-agreed bar: the method + Jev (typed pipeline) got exact 33% vs 7% for a DeepSeek single call (p<0.0001), ±3 60% vs 48%, at 9 s/trace. The held-out set is now used. Steps 7+ are unblocked. Details are in `docs/research-log.md`.
- 2026-09-19: **Real-failure test PASSED** on the locked TrajErrBench SWE-Bench Pro set (56 long real coding failures): V2 (method + Jev: gap card + checks + `step_plus_check` + `earliest_near_best`) got exact 16% vs 0% for a DeepSeek single call (p=0.004), ±3 21% vs 2%. The finalist round (V3/V3b) is not adopted. Details are in `docs/research-log.md`.
- 2026-09-19: V3–V8 (finalists, sensors, blame questions, DeepSeek picker, router + commit points, look back from the end) all failed to beat V2 on dev. **LongRCA held-out (90 real SWE failures, median 120K tokens): V2 exact 7.8% vs DeepSeek whole-trace 0% (p=0.016), but below the published 13.2%, so it FAILED the pre-agreed bar.** Details are in `docs/research-log.md`.
- 2026-09-20: **shipped.** V2 packaged as `alibi diagnose`, the `alibi-mcp` server and a Claude Code plugin, all through `alibi.localize.diagnose()` behind a 50K-token gate; repo public at https://github.com/ahmedezz26/alibi (MIT). Details are in `docs/research-log.md`.

## Glossary

See `docs/PROJECT_BRIEF.md` §9 for domain terms (trace, chunk, judge model, CUSUM, backward/RTS smoothing).
