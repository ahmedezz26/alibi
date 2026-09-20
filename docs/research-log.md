<img src="assets/alibi-icon.svg" alt="" width="56" align="left" hspace="12">

# Alibi research log

_Run artifacts (`runs/*.jsonl` and the one-off scoring scripts referenced below) are kept
locally and are not published: they embed benchmark traces whose licences differ. The numbers
here are the outputs of those runs._

_Every measurement taken while building Alibi, including the ones that failed. Pass bars were written down before each run; held-out sets were run once and never tuned on. The *why* behind the project lives in `docs/PROJECT_BRIEF.md`._

---

## 1. TL;DR — where we are

- **Final state: V2 + Jev is the adopted method and is shipped** (`alibi diagnose`, the `alibi-mcp` server, the Claude Code plugin). Step 6, the go/no-go validation, passed on 2026-09-19; V3-V9 all failed their pre-registered gates and were dropped. The entries below are in the order they happened, so earlier ones record what was true at the time.
- **AgentRx (short traces):** the windowed pipeline **loses** to a single whole-trace call. Almost every AgentRx trace is short enough for one call.
- **Key finding:** single calls collapse on traces **over ~10K tokens** (true for both gpt-5-nano and DeepSeek). So Alibi now uses **hybrid routing**: one call for short traces, the windowed pipeline for long ones.
- **AgenTracer long-trace pilot (30 traces, >10K tokens):** looked like a pipeline win (±3 16/30 vs 11/30, p=0.30), **but it did not hold up.**
- **AgenTracer long-120 run (2026-09-18, 120 evenly spaced test traces >10K, pilot included):** the pipeline **loses** to the single call. ±3: **57/120 (48%) vs 67/120 (56%)**, sign p=0.23. On the 90 new traces: 41 vs 56. The single call did **not** collapse on this data (56% ±3 against 42% random). The pipeline is near random. The gap is widest on the longest half (>16.5K tokens): 39% vs 53%.
- **STEP 6 PASSED (2026-09-19) on the locked held-out set, by the pass bar agreed beforehand.** Final version: typed pipeline + Jev, rule `chapter_x_step`, 10K windows. On the 126 never-seen long AgenTracer test traces vs a DeepSeek whole-trace call: **exact 42 (33%) vs 9 (7%), p<0.0001** (bar a ✓); **±3 75 vs 61** (bar b ✓); ±1 54 vs 29 (p=0.002). Agentic exact 42/104 vs 6; coding 0/20 vs 3 (label quirk, see §8). Speed 9.0 s/trace (5 calls) vs 4.8. AgentRx cross-check (23 real failures, too small to decide): exact 1 vs 1, ±3 6 vs 4.
- **The held-out set is now USED. Don't reuse it as a clean test.** Next steps (7+) are unblocked; see §7.
- **REAL-FAILURE TEST PASSED (2026-09-19)** on the locked TrajErrBench SWE-Bench Pro set (56 long real coding failures, median 57K tokens), by the pre-agreed bar. V2 (method + Jev, gap card + checks + `step_plus_check` + `earliest_near_best`) vs a DeepSeek whole-trace call: **exact 9/56 (16%) vs 0 (p=0.004); ±3 12 vs 1 (p=0.003)**. Speed 35 s/trace (16 calls, $0.46 total) vs 8 s. The TrajErrBench paper (ZH edition, all 86) reports direct frontier prompting at ~17% exact and TrajDebug at 24%. The SWE-56 set is now USED.
- **Judge = free `deepseek/deepseek-v4-flash-0731:free` with thinking OFF.** Paid models are off by default (see §6).

## 2. Hard rules (do not violate)

1. **Paid APIs:** never run a paid model without an explicit go-ahead for that specific run, plus a cost estimate. The code enforces this: `make_judge` refuses non-`:free` models unless `ALIBI_ALLOW_PAID_MODELS=1`.
2. **Frontier references come from papers, not runs.** GPT-5 numbers are quoted from the paper rather than re-measured here.
3. **Don't build build-order steps 7–10** (fault-injection harness, multi-resolution, backend swap, design partners) until step 6 passes. That's the `CLAUDE.md` critical rule.
4. **Never hardcode window sizes or model IDs.** They come from config (`src/alibi/config.py`). The window size is per-judge; the 2K used on AgentRx was experiment-specific only.
5. **Tune thresholds on the AgenTracer TRAIN split only.** Report on TEST. Don't tune prompts on pilot/test traces.
6. **Free endpoints may log prompts.** They're fine for public benchmarks and **never for customer/production traces** (those need redaction, brief §4).
7. Don't commit or push without being asked.

## 3. What's built (code map, `src/alibi/`)

| Module | What it does |
|---|---|
| `config.py` | **The only place settings are read** (`.env` via python-dotenv). See §6 for the env vars. |
| `types.py` | `Step`, `Chunk`, `Question` (answer types: boolean/integer/number/string), and `Annotation` (gold `critical_step`, 0-based). |
| `sources/` | `TraceSource` protocol. `langsmith.py` (key from Settings; `list_runs` is deprecated after 2027-01-31), `jsonfile.py`, `agentrx.py`, `agentracer.py`. |
| `chunking.py` | `chunk_trace` builds overlapping, token-budgeted windows of whole steps (approx 4 chars/token). `trace_tokens`, `render_step` (`[step i]` = 0-based position). |
| `judges/` | `Judge` protocol. `OpenRouterJudge` sends strict `json_schema`, logs `CallRecord` (cost/latency) in `judge.calls`, and passes the reasoning setting via `extra_body`. `make_judge` is the only constructor; it holds the paid-model guard and the retry setting. |
| `forward.py` | `forward_pass` scores each chunk with a judge-written running summary (`state_summary`, ≤150 tokens; `ChunkScore.state_in` is kept for auditing). `propagate_state=False` is the stateless ablation. |
| `drift.py` | One-sided upper CUSUM. `detect_drift` returns `ChangePoint(alarm, onset)`. Defaults k=0.2, h=0.5, baseline=0 are **uncalibrated placeholders**. |
| `backward.py` | RTS-style backward pass. The anchor is the first CUSUM alarm, else the argmax chunk if ≥0.5, else (with `known_failed`) the **last chunk** with a generic "run failed" description ("outcome" anchor). Walks chunks N..0 **in parallel** (`workers=4`) asking for `precursor_score` plus a best-guess step (always required; out-of-window steps are clamped and marked invalid). Picks by (valid, score, earliest). The prompt says root causes are the AGENT's own erroneous steps, not correct tool outputs. |
| `evaluate.py` | Benchmark harness. `EvalConfig.mode` = `hybrid` (default: one call if trace ≤ `route_threshold`, else windowed) \| `whole` \| `windowed`. Resumable JSONL (retries errored rows). `summarize` reports precision/recall at ±0/1/3 and a uniform-random baseline. |
| `cli.py` | `alibi score <trace>` (single trace, full pipeline output), `alibi eval` (alias `eval-agentrx`), `alibi download-agentrx`. |

**Typed pipeline for Jev (2026-09-19):** `judges/typesafe.py` (`TypeSafeJudge`: noul→p, score→[0,1], choice→dict; cost = input tokens × price), `forward.forward_pass_typed` (health Score + 4 warning-sign Nouls + suspect-step Choice; the memory card carries the numbers), `backward.backward_pass_typed` (only chunks ≤ the anchor; problem card = suspect step text + raised signs, or the final steps for the outcome anchor; a chapter Noul with true/false criteria + a Noul per step; pick = max P(chunk)×P(step), top-3 in `ranked`). `EvalConfig.pipeline` / `--pipeline typed` (windowed only).

Tests: 58 pass (`uv run pytest`), ruff clean. All tests use fakes, with no network.

**Eval CLI:**
```bash
uv run alibi eval --benchmark agentrx|agentracer [--subset ...] [--split test|train] \
  [--min-tokens 10000] [--sample N] [--limit N] --mode hybrid|whole|windowed \
  [--max-tokens W --overlap-tokens O --route-threshold T] [--no-state] --workers 8 --out runs/X.jsonl
```
Window defaults to `ALIBI_JUDGE_WINDOW_TOKENS`, overlap to window/8, threshold to `ALIBI_JUDGE_RELIABLE_TOKENS`. Output rows record `route`, `anchor`, `judge_model`, `judge_reasoning`, cost, latency and precursors. **Runs before this note don't record `judge_reasoning`**: files named `*_nothink*` and `agentracer_pilot_*` used thinking OFF; `agentrx_wholetrace_deepseek.jsonl` used the model default (thinking ON).

## 4. Data (all in git-ignored `data/`)

| Dataset | Status | Notes |
|---|---|---|
| **AgentRx** (`microsoft/AgentRx`, HF, gated, CC-BY-4.0) | `data/agentrx/`, `alibi download-agentrx` | **73** annotated failures (29 τ-retail + 44 Magentic-One), **not 115** (the paper *ran* 115 τ-bench tasks and 39 failed). Gold = the `step_number` of `root_cause_failure_id` (1-based). **Mostly short:** 50/73 are ≤10K tokens and 64/73 fit in 28K. |
| **AgenTracer** v1.0.0 (GitHub release `bingreeky/AgenTracer` data-v1.0.0, 2026-08-27, sha256 verified; **no license stated**, cite arXiv 2509.03312) | `data/agentracer/agentracer-data-v1.0.0/` | 790 test + 3,208 train (agentic 600/1500, coding 127/1141, math 63/567). One `mistake_step` + `mistake_agent` per trace. **86% of agentic labels are injected faults** (exact ground truth). Agentic steps are 1-based and coding/math 0-based; the loader matches `history[i].step`. **246 test traces are >10K tokens** (max ~80K). 68/600 agentic are Chinese. The release differs from the paper's data. **This is the primary long-trace benchmark.** |
| **TRAIL** (`PatronusAI/TRAIL`, gated, MIT, **no resharing**) | `data/trail/` parquet, no adapter | **Dropped.** Its ~300K-token average is ~12× inflated by OTel LLM spans replaying the history. After dedup: GAIA median 4.8K, SWE median 21.7K, 76 traces >10K. It labels ALL 841 errors with no root cause, and its agents are early-2025. One labels row needs a tolerant JSON parse (trailing comma). |
| Who&When (`Kevin355/Who_and_When`) | not downloaded | 184 step-labelled failures, mostly short. An option for extra statistical power. |
| TrajErrBench (arXiv 2608.06346) | **not released yet** | The ideal match (486 critical-step labels, τ²-bench + SWE-Bench Pro). Watch for its release. Its method **TrajDebug ("multi-granularity history compression") is close prior art. Read it.** |
| Long-Horizon Trajectory Attribution (arXiv 2608.06909, GitHub `chenjing-2024/agent-trajectory-attribution`) | not used | Short traces (median 6–20 components); mostly security/refusal attribution. Worth borrowing: **report accuracy by attribution distance** (gold step → where the failure shows up). |

**No public labeled dataset has truly huge traces (>100K tokens after dedup).** Real huge traces will need production data or a custom harness.

## 5. Results so far

**AgentRx: gpt-5-nano (paid, done earlier)**

| Run (file in `runs/`) | Exact | ±1 | ±3 | $/trace | s/trace |
|---|---|---|---|---|---|
| Windowed v2, 2K/250 windows (`agentrx_2k.jsonl`) | 13.7% | 21.9% | 30.1% | 0.013 | 298 |
| Windowed v3, forced best guess (`agentrx_2k_v3.jsonl`; **29 Magentic traces errored on the spend cap**, incomplete) | τ 17.2% | — | τ 24.1% | 0.011 | 252 |
| **Whole-trace nano** (`agentrx_wholetrace.jsonl`) | **19.2%** | 26.0% | **35.6%** | **0.0016** | 22 |
| Random | 4.0% | 10.7% | 22.9% | | |

- v2 left 18/73 traces with no prediction (the judge said "no step" in every window when outcome-anchored). v3 forced a guess, but the forced guesses were mostly far misses, biased toward early steps by the earliest-on-tie rule.
- **By length (whole-trace nano):** 13/50 exact at ≤10K vs **1/23** at >10K. The windowed pipeline got 0/23 at >10K.

**AgentRx: DeepSeek V4 Flash (free)**

| Run | Exact | ±3 | s/call |
|---|---|---|---|
| Whole-trace, thinking **off** (`agentrx_wholetrace_deepseek_nothink.jsonl`, 73) | 17.8% | 43.8% | **4.8** |
| Whole-trace, thinking **on** (`agentrx_wholetrace_deepseek.jsonl`, 54/73, stopped by user) | 22.9% on 35 (vs off 17.1% on the same 35) | 51.4% (vs 40.0%) | ~400 |

With thinking off, it gets 11/50 exact at ≤10K vs 2/23 at >10K, so the **10K routing limit holds for DeepSeek too.**

**Paper references (auto-extracted from HTML; verify against the PDFs before citing):**
- **AgentRx paper** (arXiv 2602.02475, Table 7), GPT-5 whole-trajectory baseline (with taxonomy, told the run failed, 3-run mean):
  - τ-bench (39 traces): 29.9% exact / 41.0 ±1 / 53.8 ±3 / 68.4 ±5.
  - Magentic-One (same 44): 32.6 / 43.2 / 43.9 / 52.3.
  - Full AgentRx method: τ 42.7 exact, Magentic 36.4.
- **AgenTracer paper** (arXiv 2509.03312, Table 2), whole trajectory in one pass, step accuracy with/without ground truth (2025 models):
  - Agentic: Claude Sonnet 4 30.3/29.8, DeepSeek-R1 27.1/21.8, GPT-4.1 25.0/25.7, Gemini 2.5 Pro 18.6/17.0.
  - Code: 6–16%.
  - Math: 18–46%.
  - **Rough target on long traces: ~25% agentic, ~11–15% coding.** It was measured on different data (the paper's version, all lengths).

**AgenTracer long-trace pilot (the main result so far)**
- **Setup:** 30 test traces with >10K tokens, evenly sampled (25 agentic, 5 coding; mean ~20K tokens, ~22 steps). DeepSeek free, thinking off. Window 10K, overlap 1,250.
- **Files:** `runs/agentracer_pilot_{whole,windowed}.jsonl`.

| | Exact | ±1 | ±3 | calls/trace | s/trace |
|---|---|---|---|---|---|
| **Pipeline (windowed)** | **3** | **12** | **16 (53%)** | ~5 (2.8 windows) | ~37 |
| Whole-trace | 2 | 9 | 11 (37%) | 1 | ~5 |
| Random (exact) | ~1.9 | | | | |

- **By domain:** agentic ±3 14 vs 11; coding ±3 2 vs 0.
- **The backward pass matters:** 16/30 gold steps lie BEFORE the forward-flagged window. Anchors: 17 forward, 13 outcome.
- **Not significant:** ±3 discordant pairs 10 vs 5 (sign test p=0.30); ±1 7 vs 4 (p=0.55). Exact accuracy is near random for both.

**AgenTracer long-120 run (supersedes the pilot)**
- **Setup:** `--min-tokens 10000 --sample 120` on test (stride 2.05, which contains all 30 pilot traces; 100 agentic, 19 coding, 1 math). Same judge and settings as the pilot. Files: `runs/agentracer_long120_{whole,windowed}.jsonl`. 0 errors, ~540 free requests.

| | Exact | ±1 | ±3 | calls/trace |
|---|---|---|---|---|
| Pipeline (windowed) | 14 (12%) | 34 (28%) | 57 (48%) | 4.9 |
| **Whole-trace** | 13 (11%) | **40 (33%)** | **67 (56%)** | 1 |
| Random | 7% | 20% | 42% | |

- **Discordant pairs at ±3:** 23 only-pipeline vs 33 only-whole (p=0.23). **Agentic:** 50 vs 61 (p=0.15). **Coding:** 6 vs 5.
- **By length (median 16.5K tokens):** ≤16.5K 56% vs 59%; >16.5K **39% vs 53%**. The pipeline gets worse on longer traces, which is the opposite of the thesis.
- 52.5% of gold steps come before the forward-flagged chunk, but the backward pass mostly fails to recover them.
- **Where the look-back fails (offline analysis, no API calls):** the true step sits in the alarm chapter in 48/120 traces, earlier in 63, later in 9. The look-back ranks the true chapter top (including ties) in **73/120** (vs ~1/2.7 by chance), with a tie at the top in 30. **Inside the true chapter, the step pick is at chance:** exact 22 vs 18 random, ±3 79 vs 83 random. **The bottleneck is recognizing the wrong step, not finding the chapter.**
- **Takeaway:** the pilot win was noise (regression to the mean). DeepSeek V4 Flash whole-trace holds up past 10K on AgenTracer, so the "single calls collapse past 10K" finding (from AgentRx) doesn't transfer to this data or judge.

**Sample trace** (`examples/sample_trace.json`, flight booking with an error at step 2): stateless forward misses it, stateful catches it, backward localizes step 2 after the prompt fix. n=1, anecdotal.

## 6. Environment

- **`.env` (git-ignored):**
  - Keys: `OPENROUTER_API_KEY`, `LANGSMITH_API_KEY`, `HF_TOKEN` (has access to AgentRx and TRAIL).
  - Judge settings: `ALIBI_JUDGE_MODEL=deepseek/deepseek-v4-flash-0731:free`, `ALIBI_JUDGE_REASONING=off`.
- **All config vars, with defaults in `config.py`; template in `.env.example`:**
  - `ALIBI_JUDGE_MODEL`: default is the free DeepSeek.
  - `ALIBI_ALLOW_PAID_MODELS`: default 0 (the spend guard).
  - `ALIBI_JUDGE_REASONING`: `off|low|medium|high`, empty = model default.
  - `ALIBI_JUDGE_MAX_RETRIES`: default 8 (SDK backoff).
  - `ALIBI_JUDGE_RELIABLE_TOKENS`: default 10000, the routing threshold. Re-measure it per judge.
  - `ALIBI_JUDGE_WINDOW_TOKENS`: defaults to the reliable value.
- **Free-model limits:**
  - **1000 requests/day** (`GET https://openrouter.ai/api/v1/key` → `free_model_daily_requests`), plus a per-minute cap. 8–12 workers have been fine.
  - Free endpoints sometimes return **empty responses**. Those rows are recorded as errors and retried on resume.
  - The free DeepSeek endpoint is hosted by the third-party provider "OpenInference" (fp8). It can change or disappear. Fallback: `nvidia/nemotron-3-super-120b-a12b:free` (tested, works). OpenRouter has no free gpt-oss.
- **Free-quota state at this update:** 150 requests left on 2026-09-18 (resets daily).
- **`.mcp.json`** (LangSmith, Hugging Face, Context7, alphaxiv) is at the repo root.

## 7. Next steps (in order)

1. **[Done 2026-09-18] Long-120 comparison:** the pipeline loses (§5). The remaining 126 long test traces can still be run later for power, but it's unlikely to flip the sign.
2. **Diagnose the loss before spending more quota (on TRAIN, not test):**
   - Error analysis on the 33 traces the pipeline missed but the single call got: did the forward pass flag the wrong chunk? Did the backward pass pick a wrong earlier step (earliest-on-tie bias, precursor scores saturated at 1.0)?
   - Ablations: forward-only best step, backward without earliest-tie, and a "whole-trace + pipeline vote" ensemble.
   - Re-measure the routing threshold for DeepSeek on AgenTracer: the whole-trace call didn't collapse at 10–30K here.
3. If the pipeline wins significantly, calibrate on the **train** split: the routing threshold, window size, CUSUM k/h, and the earliest-on-tie rule (it caused an early-step bias in v3). Then maybe a small frontier-model reference, **only with an explicit go-ahead**.
4. Known weaknesses to address:
   - Precursor scores saturate at 1.0, so the tie-break decides too much.
   - Exact-step accuracy is near random, and the signal is in ±1/±3.
   - State compression loss has not been audited yet (`state_in` is logged).
5. Housekeeping:
   - `langsmith` `list_runs` → `client.runs.query` before 2027-01-31.
   - Read the TrajDebug paper (prior art).

## 8. Dated log (condensed)

- **2026-09-18:**
  - **Setup:** scaffolded with uv; steps 1–5 built; backward pass built; judge on OpenRouter.
  - **AgentRx (gpt-5-nano):** loaded (73 traces, not 115). Windowed runs v2/v3; whole-trace nano dominates on these short traces. Length finding (single calls collapse >10K) → hybrid routing.
  - **Mistakes to learn from:**
    - An unrequested GPT-5 run (~$0.43+, stopped at 14/73; `runs/agentrx_wholetrace_gpt5.jsonl` is **partial, not for accuracy**).
    - A spend-cap overrun that broke v3.
  - **Long-trace data:** TRAIL downloaded and dropped. AgenTracer chosen.
  - **Free judge:** switched to free DeepSeek, added the paid-model guard, retries, the reasoning toggle and the parallel backward pass. Measured that thinking off is 80× faster at similar accuracy.
  - **AgenTracer pilot:** first pipeline win (±3 16 vs 11, n=30, p=0.30).
  - **Long-120 run:** the pipeline **loses** (±3 48% vs 56%, p=0.23). The pilot win was noise. Step 6 is a no-go as built. 150 free requests were left at the end of the day.
- **2026-09-19, Jev setup:**
  - Installed `langchain-typesafe` 0.0.1a2 (beta warning; `load_dotenv()` must be given the repo `.env` path when run from elsewhere). The key works; the model is `jev-1.13.0`, with ~0.6–0.9s per call.
  - **Max input ≈ 32K real tokens** (a 400 error with `max_tokens_exceeded` at ~33K; 32,013 accepted). Our 4-chars/token estimate undercounts by ~3%. 10K chapters fit easily; whole traces >32K can't be sent at all.
  - Setup probes spent ~141K input tokens, **≈ $0.006**.
- **2026-09-19, Jev stage 1 (chapter test, TRAIN, paid, $0.055):**
  - 50 evenly spaced train traces >10K tokens (IDs in `runs/jev_stage1_ids.json`; script and results `runs/jev_stage1*.{py,jsonl}`).
  - Setup: 10K chapters; the failure card is the last 2 steps plus "run failed"; one Noul per step and a chapter Noul, on the gold chapter and one control chapter.
  - **Step pick inside the gold chapter: top-1 17/50 vs 7.0 random; top-2 22 vs 13; ±1 27 vs 16.8; ±3 37 vs 30.2.** The median gold rank is 3 of ~12. **No saturation:** 0 of 622 step probabilities >0.9, and no ties.
  - **Chapter Noul is weak:** gold > control in 32/48 (sign p=0.03), but mean p is 0.63 vs 0.58.
  - **Combined over 2 chapters** (P(ch)×P(step)): top-1 9/48 vs 3.2 random.
  - **Verdict: GO.** Jev spots the step far better than chance, where DeepSeek was at chance (on test). The chapter signal needs work in stage 2.
- **2026-09-19, stage 2 built** (free, TDD, 16 new tests; 58 pass). Not yet run on real traces. **Next: stage 3**, the full typed pipeline on the same 50 train traces (`runs/jev_stage1_ids.json`), ~$0.15, not yet run. Compare the chapter pick vs stage 1 (32/48).
- **2026-09-19, stage 3: typed pipeline on the same 50 TRAIN traces (paid, $0.123):** `runs/jev_stage3_train50.jsonl`. Paired with free DeepSeek runs on the same 50 (`runs/ds_train50_{whole,windowed}.jsonl`):

  | 50 train traces | Exact | ±1 | ±3 |
  |---|---|---|---|
  | **Jev + method (typed)** | **12** | **22** | **31** |
  | DeepSeek + method (text) | 7 | 12 | 19 |
  | DeepSeek whole-trace | 5 | 10 | 23 |
  | Random | ~3 | ~8 | ~17 |

  - **Jev vs DeepSeek, same method:** ±1 13:3 (p=0.02), ±3 16:4 (p=0.01). **Jev method vs DeepSeek whole:** ±1 16:4 (p=0.01).
  - The chapter containing the pick holds gold 37/50. Gold came after the alarm chapter (never examined) in 1/50. Gold was in the top-3 in 23/50.
  - **Weak spots:** coding exact 2/21 (agentic 10/29). Confidence is only mildly informative (top p≥0.4: 13/26 within ±1 vs 9/24).
  - No thresholds were tuned (CUSUM defaults unchanged), so the test set stays clean. **Next: stage 4 on the 120 test traces, ~$0.30.**
- **2026-09-19, stage 4: typed pipeline on the 120 long TEST traces** (the same traces as the long-120 DeepSeek runs; $0.341 incl. reruns). `runs/jev_stage4_test120.jsonl`.

  | 120 test traces | Exact | ±1 | ±3 | s/trace | calls/trace |
  |---|---|---|---|---|---|
  | **Jev + method** | **39** | **48** | **68** | 6.9 | 5.2 |
  | DeepSeek + method | 14 | 34 | 57 | 34.3 | 4.9 |
  | DeepSeek whole-trace | 13 | 40 | 67 | 5.1 | 1 |
  | Random | ~8 | ~24 | ~50 | | |

  - Paired, Jev vs whole: exact 35:9 (p=0.0001), ±1 23:15 (p=0.26), ±3 23:22 (p=1.0). Jev vs DeepSeek + method: exact 32:7 (p=0.0001).
  - Agentic exact 38/100 vs 12; coding 0/19 vs 1. Long half exact 22 vs 5; short half 17 vs 8. Gold in the Jev top-3: 74/120.
  - **Bug found and fixed:** `render_step` uses `json.dumps` (ensure_ascii), so Chinese becomes `\uXXXX`. Measured on Jev at ~4.75 tokens per escape, but `approx_tokens` counted 1.5, so 4 traces overflowed Jev's 32K limit (400 `max_tokens_exceeded`). `approx_tokens` now counts 5 per escape (test added; 59 pass). **Only those 4 traces were rerun** (by ID; rerunning the CLI would change the `--min-tokens` selection). Every earlier run, DeepSeek included, fed escaped Chinese to the judge. **Follow-up:** render with `ensure_ascii=False` + a CJK-aware count, and rerun the affected traces for all methods.
  - Nothing was tuned on test. The CUSUM defaults are unchanged.
- **2026-09-19, overfitting guard ("are we overfitting?"):**
  - **Admission:** the 4 proposed fixes (prefer-earlier, symptom Noul, forward-informed chapter prior, coding off-by-one) came from analysing Jev's misses on the 120 **test** traces. That breaks the "diagnose on train only" rule. They stand only if the same patterns reappear on TRAIN, and they must never be scored on those 120 again as proof.
  - **Locked held-out set:** `docs/heldout_agentracer_test_ids.json`, the 126 long test traces (>10K, len//4 counter) never used (104 agentic, 20 coding, 2 math). Use it **once**, for a final verdict. Never inspect misses or tune on it. `alibi eval --ids FILE` runs an exact list (new, tested; 61 pass).
  - **Cross-dataset check:** `docs/agentrx_long_ids.json`, the 23 AgentRx traces >10K (real, not injected, failures). The DeepSeek whole-trace baseline already exists (`runs/agentrx_wholetrace_deepseek_nothink.jsonl`).
  - **Pass bar, fixed in writing on 2026-09-19, before any held-out run:** on the 126 held-out traces, the current Jev method vs a DeepSeek whole-trace call on the same traces: (a) more exact-step hits, paired sign test p<0.05; (b) ±3 hits not lower than whole-trace.
- **2026-09-19, decision:** improve the method first (on TRAIN only), then spend the held-out set once on the final version.
- **2026-09-19, test-derived patterns re-checked on TRAIN (stage 3 rows, free):**
  - "Picks a later step (symptom)": only 9/19 train misses, **not confirmed**, so prefer-earlier is dropped.
  - **Wrong chapter: 12/19 train misses, confirmed** (the main target).
  - **Coding +1 offset confirmed (5/21 at exactly +1 vs 2 exact) and explained:** in **637/1141 coding train traces the gold step is a context/prompt dump** ("<Agent> thinking: # Past Experience … # Current Plan"), not the agent's own output. Our questions ask for the agent's own mistake, so Jev picks the next step. In 2 inspected traces, the mistake described in `mistake_reason` doesn't appear anywhere in the trace text. **This is a label convention of the benchmark, not a method error. Don't shift predictions to fit it** (that would be overfitting); report coding at ±1 alongside exact.
- **2026-09-19, improvements built (free, TDD; 71 tests pass):**
  - **Readable non-ASCII:** `render_step` uses `ensure_ascii=False`. `approx_tokens` counts 1 token per non-ASCII char (measured on Jev: 1.01 per Chinese char, vs ~4.75 per `\uXXXX` escape).
  - **All evidence saved:** rows carry `backward_evidence` (P(chunk) and every P(step)) plus `forward_answers`. `evaluate.rescore(row, rule)` re-picks offline with no calls.
  - **Fixed combination rules** (`backward.COMBINE_RULES`, `--combine-rule`): `chapter_x_step` (original), `step_only`, `rise_prior` (weight 0.5 + the health jump in that chunk), `forward_suspect` (average of the look-back step Noul and the forward suspect Choice). No learning; picking one is a discrete choice made on TRAIN.
  - **Train sets:** `docs/train_first50_ids.json` (the stage 1/3 traces) and `docs/train_fresh50_ids.json` (disjoint; 28 agentic, 21 coding, 1 math).
  - **Rule-choice procedure (fixed before running):** one run of the new version on the 100 train traces, then all 4 rules rescored offline. Choose the rule with the most exact hits over all 100 (tie-break ±3). Replace `chapter_x_step` only if the new rule is ≥ it on **both** halves (first 50 and fresh 50); otherwise keep `chapter_x_step`. Then the held-out run happens once, with that rule.
- **2026-09-19, rule choice on 100 TRAIN traces (v2 = readable non-ASCII + digit-aware counter + oversized-step clipping; spent ≈ $0.20):** `runs/jev_v2_train_{first50,fresh50}.jsonl`, 0 errors after fixes.

  | rule | first50 exact/±1/±3 | fresh50 exact/±1/±3 | exact total |
  |---|---|---|---|
  | **chapter_x_step** | **13**/23/31 | 11/18/25 | **24** |
  | step_only | 11/19/27 | 12/18/25 | 23 |
  | rise_prior | 11/21/30 | 10/18/25 | 21 |
  | forward_suspect | 11/21/30 | 9/16/21 | 20 |

  - **By the pre-registered procedure, `chapter_x_step` stays.** None of the forward-informed chapter rules helped, so that "fix" is not supported. v2 vs v1 on first50: exact 13 vs 12 (same).
  - **More robustness fixes, found on train:** agentic/train/896 has one ~1M-char numeric tool output (~265K tokens). `chunk_trace` now clips oversized steps (head + tail + "[N chars omitted]"). `approx_tokens` counts 1 token per digit (measured: numeric output ≈ 1 real token/char; the old estimate was 4× low). 73 tests pass.
  - **Total Jev spend so far ≈ $0.74.**
  - **Final version for held-out:** v2 code, `--combine-rule chapter_x_step`, window 10K. Next: the held-out run (~$0.35 Jev + free DeepSeek whole baseline on the same 126) and the AgentRx cross-check (~$0.06).
- **2026-09-19, FINAL held-out run (Jev $0.249 + AgentRx $0.061).** Files: `runs/heldout126_{jev,ds_whole}.jsonl`, `runs/agentrx23_{jev,ds_whole}.jsonl`; the DeepSeek baselines were rerun with the v2 rendering, so both sides read the same text.

  | 126 held-out | Exact | ±1 | ±3 | s/trace |
  |---|---|---|---|---|
  | **method + Jev** | **42 (33%)** | **54** | **75 (60%)** | 9.0 |
  | DeepSeek whole-trace | 9 (7%) | 29 | 61 (48%) | 4.8 |
  | random | ~8.7 | | | |

  - Discordant pairs: exact 38:5 (p<0.0001), ±1 44:19 (p=0.002), ±3 36:22 (p=0.09). **Pass bar met: (a) and (b).**
  - The stage 4 result replicated on unseen data (33% exact both times).
  - AgentRx 23: exact 1/1, ±1 4/2, ±3 6/4. Inconclusive (n=23, both low).
- **2026-09-19, AgentRx real-failure diagnosis (AgentRx is now a DEV set, not a clean test):**
  - In 17/17 misses Jev rated the true step lower (~0.2) than a later step (0.3–0.5), usually in the right chapter. 17/23 predictions are later than gold; gold is mostly early (median 20% into the run).
  - The alarm fires in chunk 1 in 19/23 traces (uninformative).
  - **Cause:** real mistakes are *quiet*. Examples: step 3 "search family blogs" when the user asked for TripAdvisor; asking FileSurfer to "listen to" an mp3. Each looks fine alone and is wrong only relative to the request or a tool's abilities. Per-step Nouls judged in isolation favour later, visibly troubled steps. AgenTracer's injected faults are locally visible, which hid this.
- **Data:** Who&When (HF `Kevin355/Who_and_When`, no licence stated, 184 traces) **overlaps AgentRx heavily** (80 shared task IDs; AgentRx Magentic-One ≈ relabelled Who&When), so it is not a fresh test. **TrajErrBench is released** (`THU-KEG/TrajDebug`, MIT for annotations; upstream terms for trajectories), in `data/trajdebug/` (git-ignored, sparse clone of `data/`). Adapter `sources/trajerrbench.py`, `--benchmark trajerrbench`. EN edition: **SWE-Bench Pro 86 traces, median 51K tokens, 78 >32K, labels always on the agent's own message**; τ²-bench 400, median 6K. The paper evaluates the ZH edition.
  - **Splits (seed 20260919):** `docs/trajerrbench_swe_dev30_ids.json` (dev), **`docs/heldout_trajerrbench_swe56_ids.json` (LOCKED real-failure held-out)**, `docs/trajerrbench_tau2_dev50_ids.json` (dev).
- **Method changes for quiet mistakes (free, TDD, 83 tests; all switchable, defaults = the old behaviour):**
  - `--card gap`: the look-back card shows the user's request + how the run ended + the forward symptom.
  - `--checks`: 4 per-step Nouls (`backward.STEP_CHECKS`: request / capability / unfounded / observation), only on agent-written steps (not system/tool). ≤204 questions per call on SWE.
  - Rule `step_plus_check`: P(chunk)×mean(P(step), max check).
  - `--pick earliest_near_best`: the earliest step with a score ≥0.8×best.
- **Version-choice procedure (fixed BEFORE running):**
  - Run v2 (symptom card, no checks) and v3 (gap card + checks) on the dev sets: SWE dev30 + τ² dev50 (real failures). Run v3 on the 100 AgenTracer train traces (v2 rows already exist).
  - From the v3 rows, rescore 4 variants: {chapter_x_step, step_plus_check} × {best, earliest_near_best}.
  - Adopt the v3 variant with the most exact hits on real-failure dev, **only if** it beats v2 on that dev **and** is not more than 3 exact hits below v2 on AgenTracer train 100 (24). Otherwise keep v2.
  - Then one run on the locked SWE-56 held-out vs a DeepSeek whole-trace call, against a pass bar agreed beforehand.
- **2026-09-19, version choice run ($0.70: dev80 v2 $0.228 + v3 $0.251; AgenTracer100 v3 $0.218).**

  | version | real dev80 exact/±1/±3 | SWE30 exact/±3 | τ²50 exact/±3 | AgenTracer train100 exact/±3 |
  |---|---|---|---|---|
  | v2 (old) | 17/18/33 | 3/7 | 14/26 | 24/56 |
  | v3 chapter_x_step/best | 15/16/30 | 3/6 | 12/24 | 27/49 |
  | v3 chapter_x_step/earliest_near_best | 16/17/30 | 3/5 | 13/25 | 36/60 |
  | v3 step_plus_check/best | 15/15/24 | 3/6 | 12/18 | 20/46 |
  | **v3 step_plus_check/earliest_near_best** | **21/21/34** | 3/5 | 18/29 | 29/53 |

  - **By the pre-registered rule, adopt v3 = `--card gap --checks --combine-rule step_plus_check --pick earliest_near_best`** (21 > 17 on real dev; AgenTracer 29 ≥ 24−3).
  - **Not significant:** paired exact on real dev is 10:6 (p=0.45); AgenTracer 11:6 (p=0.33). Picking the best of 4 variants on dev is optimistic, so only the locked held-out can confirm it.
  - **Long real coding (SWE30) is unchanged and poor:** exact 3/30 for both (random ≈0.3), ±3 worse (5 vs 7). The gain comes from τ² (short customer-service traces).
  - Speed on SWE: ~40 s/trace, ~13 calls (long traces).
- **Naming:** **V2 = current adopted version** (gap card + checks + `step_plus_check` + `earliest_near_best`; `runs/teb_dev80_v3.jsonl` / `runs/jev_v3_train_*` hold its raw answers, rescored). **V3 = V2 + finalist round** (`alibi finalists`: shortlist the top-8 by `step_plus_check`, one Jev Choice over them side by side with the request + ending; candidates fair-share-clipped to one chapter's budget; tested; 88 pass).
- **2026-09-19, V3 vs V2 on the same dev traces (spent $0.052; total Jev ≈ $1.80):**

  | set | V2 exact/±1/±3 | V3 exact/±1/±3 | only-V3:only-V2 | gold in top-8 |
  |---|---|---|---|---|
  | SWE long coding (30) | 3/3/5 | 4/4/5 | 1:0 | 14 |
  | τ² customer service (50) | 18/18/29 | 13/13/24 | 4:9 | 46 |
  | real all (80) | **21**/21/34 | 17/17/29 | 5:9 (p=0.42) | 60 |
  | AgenTracer (100) | **29**/41/53 | 26/43/59 | 6:9 | 67 |

  - **V3 is not adopted** (it doesn't beat V2 on real dev). The side-by-side Choice picks the true step less often than V2's aggregated per-step evidence, even though it is in the shortlist 46/50 times on τ² (V3 13 vs V2 18).
  - Likely reasons: candidates are shown without their surrounding steps (e.g. the tool output that makes them wrong), and the Choice has no earliest-cause preference.
  - Long coding gains +1 (4 vs 3) with the gold in the top-8 for 14/30, so the ceiling is there but not reached. Mean finalist call 2.8s.
- **2026-09-19, one more idea before the final real-failure test. Pre-registered BEFORE building:**
  - **V3b** = finalists shown **with context** (the previous and next step around each marked candidate) + `earliest_near_best` applied to the finalist Choice probabilities.
  - **The only variant tried.** Adopt it over V2 only if exact > 21 on real dev80 **and** ≥ 26 on AgenTracer train100. Otherwise keep V2 and run the locked SWE-56 final test with V2.
- **2026-09-19, V3b result (spent $0.080; total Jev ≈ $1.88):** `runs/*v3b*.jsonl`

  | set | V2 | V3 | V3b (exact/±1/±3) |
  |---|---|---|---|
  | SWE long coding (30) | 3/3/5 | 4/4/5 | **5/5/7** |
  | τ² (50) | **18**/18/29 | 13/13/24 | 16/16/25 |
  | real all (80) | 21/21/34 | 17/17/29 | 21/21/32 |
  | AgenTracer (100) | 29/41/53 | 26/43/59 | 27/45/58 |

  - **Not adopted:** it ties V2 on real dev (21, and the rule required >21); paired 8:8.
  - V3b helps long coding a little (5 vs 3, p=0.5, n=30) and hurts τ² (16 vs 18). "Use finalists only on long traces" would be tailoring after seeing dev results. It stays a **hypothesis for fresh data** (e.g. step-7 synthetic long traces), not a change.
  - **Current best remains V2. Next: the locked SWE-56 final real-failure test with V2** (pass bar to agree first; a DeepSeek whole-trace baseline is free once the daily quota resets; some SWE traces reach ~125K tokens, so check the DeepSeek context).
- **Pass bar for the SWE-56 final real-failure test, fixed in writing on 2026-09-19, before running:**
  - V2 (`--card gap --checks --combine-rule step_plus_check --pick earliest_near_best`) vs a DeepSeek whole-trace call on the same 56 locked traces.
  - Passes if: (a) more exact hits, paired sign test p<0.05; (b) ±3 hits not lower.
  - Traces too long for DeepSeek count as whole-trace misses.
- **2026-09-19, FINAL real-failure test on the locked SWE-56** (Jev $0.461; DeepSeek whole free; DeepSeek free context ≈1M tokens, so all fit). Files: `runs/heldout_swe56_{v2,ds_whole}.jsonl`.

  | 56 held-out SWE-Bench Pro | Exact | ±1 | ±3 | s/trace |
  |---|---|---|---|---|
  | **V2 (method + Jev)** | **9 (16%)** | **9** | **12 (21%)** | 35.4 |
  | DeepSeek whole-trace | 0 | 1 | 1 | 8.1 |
  | random | ~0.5 | | ~3.5 | |

  - Paired: exact 9:0 (p=0.004), ±1 9:1 (p=0.02), ±3 12:1 (p=0.003). **Pass bar met.**
  - **Baseline sanity check:** the same rendering on both sides. DeepSeek's predictions cluster at the start (median 9% of the trace vs gold at 63%), and 3/56 are out of range. The single-call collapse on long traces is real, not a bug. The baseline is a free model; the paper's frontier direct-prompting reference is ~17% (ZH edition, different subset: not directly comparable).
  - V2 dev (SWE30) was 10% exact, held-out 16%, so no sign of dev overfitting.
  - **Total Jev spend ≈ $2.34.**
- **2026-09-19, V4 "Jev-native" sensor check** (TypeSafe guidance: narrow questions, only relevant context, JSON state with backtick paths, code composes). `runs/sensor_test.py` → `runs/sensor_v4_dev80.jsonl`.
  - Setup: per assistant step, a small JSON state (user_request ≤800, rules/tools ≤500, 2 before ≤500 each, step ≤1200, 2 after ≤500 each) and 6 local questions (severity Score + request / capability / unsupported / misread / consequence Nouls).
  - **Spent $0.246 (est. ~$0.21; +17%).** Calls average 0.74s (vs 1–3s for chunk calls).
  - **Primary metric (pre-set): gold rank among assistant steps by severity vs V2: no better.** SWE: median rank 10 vs 10 (random 27.5), top-1 2 vs 3, top-10 15 vs 15. τ²: median 4 vs 3, **top-1 5 vs 12**, top-3 21 vs 29. All 80: top-1 7 vs 15.
  - Exact pick: V4 severity + earliest 14 vs V2 21. **V4 is not adopted.**
  - **Jev does see the errors locally:** gold severity is 0.28 vs 0.07 typical (SWE) and 0.41 vs 0.18 (τ²). The best single cue is `unsupported` (τ² gold rank 2, SWE 9.5). But many other steps score high too, so the local sensor alone can't single out the root cause. V2's chunk context + gap card + estimation does better, especially on τ².
  - **Confound:** the τ² policy/rules were clipped to 500 tokens, so relevant rules may have been cut.
  - **Fresh data note:** τ² has 350 unused traces (only dev50 used), a clean place to validate any future rule chosen on dev. All 86 SWE traces are used.
- **2026-09-19, combining V2 + the V4 sensor. Checked free on the stored dev80 answers BEFORE spending on the 350 fresh τ² traces (~$0.67 estimated):**

  | fusion rule (exact hits) | SWE30 | τ²50 | all80 |
  |---|---|---|---|
  | V2 alone | 3 | 18 | **21** |
  | average rank (V2, unsupported) | 0 | 15 | 15 |
  | average rank (V2, severity) | 3 | 9 | 12 |
  | V2 × (1+unsupported), earliest_near_best | 4 | 15 | 19 |
  | V2 × (1+severity), earliest_near_best | 4 | 16 | 20 |

  - **No combination beats V2 on dev**, so the fresh-data validation run was NOT spent.
  - Conclusion: the per-step local sensor adds noise, not information, on top of V2. V2 remains the method.
- **2026-09-19, cascade A→B (V2 shortlists its top-K agent steps, the V4 sensor picks among them). Free on stored dev80:** every K (2/3/5/8) × cue (severity/unsupported) is worse than V2. Best all80: 14 vs 21. τ² with K=8 and severity: the gold is in the shortlist 48/50 but B picks it only 5 times (random within the list ≈ 6).
  - **Insight:** B ranks by *how wrong a step looks*, which favours visible symptoms over quiet root causes. V2's look-back + `earliest_near_best` encodes causality/time, which B lacks. **Not pursued; no spend.**
- **2026-09-19, V5: A picks the chapter, B picks the step with "quiet cause" questions. Paid, $0.165, dev80 only.** A = the chapter holding V2's pick (gold inside 65/80). B = Jev on every agent step in it, with the V2 gap card + 2 steps of context and 3 Nouls: blame ("the problem in the card started here"), reaction ("reacting to an earlier problem"), decision ("a choice later steps build on"). Script: `runs/blame_test.py`; results: `runs/blame_v5_chapter_dev80.jsonl`.
  - Pre-registered pass bar (> V2's 21/80 exact, primary = earliest_near_best over blame×(1−reaction)): **FAILED, 10/80.** Other reported rules: 9–13/80. SWE30: 5 vs 5 (same traces); τ²50: 5 vs 16 (±3: 14 vs 30).
  - Signal check (gold vs other steps in the chapter, mean P): blame 0.36 vs 0.29 (weak), reaction 0.35 vs 0.33 (none), **decision 0.83 vs 0.65 (the only real separation)**. On flat blame scores, earliest_near_best drifts early (τ² wrong-earlier 30 of 45 misses).
  - Conclusion: isolated per-step questions, even ones pointed at the known symptom, don't find the quiet cause better than V2's chapter look-back. `decision` is a hypothesis only; it was not tuned on dev80, and it needs fresh train data before any use. V2 remains the method.
- **2026-09-19, V6: Jev/V2 picks the chapter, free DeepSeek Flash reads that whole chapter (+ the V2 gap card) and names the root-cause step. Free, dev80 only.** Script: `runs/deepseek_b_test.py`; results: `runs/deepseek_b_v6_dev80.jsonl`. Pre-registered: an invalid pick (outside the chapter / not an agent step) falls back to V2; pass = > 21/80.
  - **FAILED: 18/80 vs 21** (SWE 4 vs 5, τ² 14 vs 16; ±3: 32 vs 38). 10 invalid picks (9 on SWE, mostly tool-output steps), 6.8 s/trace.
  - With the gold inside the chapter (65), DeepSeek's raw pick is right 16 times vs V2's 21. They agree on 22 traces (8 of them right). Only V2 right: 11; only DeepSeek right: 8, so they're partly complementary. A fusion rule was NOT fitted on dev80 (tuning risk).
  - Conclusion: a stronger free-text reader of one chapter still doesn't beat V2's per-step look-back with `earliest_near_best`. V2 remains the method.
- **2026-09-19, "follow the edit" (free, code only, no model; SWE dev30).** Part of the idea of routing by trace type (code / math / conversation). Script: `runs/follow_edit_check.py`. Two bugs were fixed after the first run: "first mention" matched the user's issue text, and Grok edits through bash weren't detected. The rules were not otherwise changed.
  - The chapter holding the agent's **first edit** contains the gold **19/30** vs V2's chapter 17/30. As a step: first edit is exact 5, ±3 8, identical to V2 (5, 8) at zero model cost. The two chapters agree 17/30; gold is in either one 22/30 (only V2's: 3, only first-edit's: 5). V2 step or first-edit step hits exactly 8/30.
  - First *mention* of the edited file is useless (2/30): agents name files early while exploring.
  - Too small to adopt (30 traces, +2); a possible code-route signal to pre-register and test on fresh coding traces.
- **2026-09-19, fresh-data check for the code route:** **0 unused real coding failures remain.** All 86 TrajErrBench SWE traces are used (dev30 + locked held-out 56), and the ZH edition is the same trajectories translated. AgenTracer coding doesn't fit: it's code generation, not repo edits, and has the prompt-dump label quirk. TRAIL SWE labels every error, not the root cause. AgentDebugBench is alfworld/gaia/webshop. τ²: 350 of 400 unused, so the conversation route CAN be tested on fresh data.
- **2026-09-19, naming:** **V7 = route by trace type before V2.** Code route: V2's look-back also gets the chapter of the agent's first code edit. Conversation route: step checks against the user request + policy (to design). Math: none yet. The router uses tool names, with a Jev Choice on the trace start as a fallback. V5 = Jev blame questions in A's chapter (failed); V6 = DeepSeek reads A's chapter (failed).
- **2026-09-19, new data found: LongRCA Bench** (arXiv 2608.15242, HF `CLoud5-real/longrca-bench`, public/not gated, updated 2026-09-02; **no licence stated**, so cite and don't reshare). 1,140 **real, non-injected** failed trajectories with human labels (`mistake_step` = the earliest decisive step, `mistake_agent`, `mistake_reason`), in the same `history` format as AgenTracer/Who&When. By source: SWE-bench Pro 128, Terminal-Bench 2 42, TravelPlanner 685, VitaBench 108, WebArena Verified 177. Very long: median 145 steps (max 728); the root is a median 48 steps before the end. Paper baselines: exact root step 13.2% (ECHO) and 24.1% (their RCTA). Human exact-step agreement is only 39.5%, so the labels are noisy. Multi-agent roles (e.g. DiagnostAgent → ActionAgent), so these are different runs from TrajErrBench's SWE traces. **Not yet downloaded.**
- **2026-09-19, LongRCA set up (free).** Downloaded to `data/longrca/` (git-ignored; `longrca-full.parquet` converted to `longrca-full.jsonl`). Adapter `sources/longrca.py` (terminal output → `tool` step), `--benchmark longrca --subset swe_bench_pro|terminal_bench_2|travelplanner|vitabench|webarena_verified|all`; 4 new tests, **95 pass**. Sizes (median tokens): SWE 122K, TB2 82K, TravelPlanner 50K, WebArena 37K, VitaBench 12K; essentially all >10K.
  - **Split locked before inspecting any label or content** (seed 20260919, per subset): `docs/longrca_dev_ids.json` (dev: SWE 38, TB2 12, Vita 33, TP 60, WA 40) and **`docs/heldout_longrca_ids.json` (LOCKED: SWE 90, TB2 30, Vita 75, TP 150, WA 100)**. The other 425 TravelPlanner and 37 WebArena traces are unassigned reserve.
- **2026-09-19, V7-code free checks on LongRCA DEV (SWE 38, TB2 12; no model).** Script: `runs/follow_edit_longrca.py`.
  - **First edit** (rule written before looking): SWE exact 3/38, ±3 5, gold in its chapter 13/38 (random agent step: 4.4). TB2: exact 0, chapter 6/12 (random 3.9). As a step guess it is weak.
  - **Dev diagnosis:** the gold sits before the first edit in 17 traces, at it in 3, after it in 18 (median offset 0, quartiles −13/+34). In **23/38 SWE traces the gold is a handoff step** (`DiagnostAgent (-> ActionAgent)`: the diagnostician telling the executor what to do); ActionAgent 11, JudgeAgent 2.
  - **First handoff = the gold in 16/38 SWE (42%) exact, ±3 18**; the last handoff 0; a random handoff ≈ 5.8. It is handoff #1 in 16 of the 23 handoff-gold cases. TB2: 1/12 (doesn't transfer). Paper reference (all domains): best 24.1% exact.
  - **Caution:** this is found on dev and exploits this benchmark's multi-agent harness. It may partly be an annotation convention ("earliest decisive" = the first committed diagnosis). The general idea behind it ("the first moment a diagnosis/plan is committed and handed to execution") is the "decision" lead again. Before any held-out use it must be written down as a rule.
- **2026-09-19, V7-code built (TDD; 100 tests pass).** `--commit` asks one extra Noul per agent step inside the look-back calls (`backward.COMMIT_QUESTION`: "the agent commits to a diagnosis or plan … that it or another agent then acts on"). It is stored as `p_commit` in `backward_evidence`. New fixed rule `commit_x_step_plus_check` = V2's `step_plus_check` × P(commit). Jev answers questions independently, so **one run yields V2 and V7 on the same calls** (V2 = `rescore(row, "step_plus_check", "earliest_near_best")`).
  - **Pre-registered for the dev run** (`docs/longrca_dev_code50_ids.json`, 50 traces): the primary V7 = `commit_x_step_plus_check` + `earliest_near_best`. **Go to held-out only if V7 exact > V2 exact on dev.** Also reported, not deciding: ±3, and V7 with `best`. Estimate ~$0.76 (calibrated: $0.13 per 1M trace tokens from SWE-56; 5.9M trace tokens + ~$0.02 for commit questions).
- **2026-09-19, V7 router built (TDD; 106 tests pass).** `src/alibi/routing.py:trace_kind(steps)`: plain code on the first 60 steps. `code` if ≥20% of agent steps run commands or edit files; `conversation` if ≥2 times the agent asks a question in plain text and gets a non-tool reply; else `other`. Every row now stores `trace_kind`. `evaluate.v7_rule(row)`: code → `commit_x_step_plus_check`, everything else → V2 (`step_plus_check`); the conversation branch is not designed yet. A Jev fallback wasn't needed.
  - **Router accuracy (free, all data):** coding 256/256 (TEB SWE, LRCA SWE + TB2); customer service 489/508 (τ² 387/400, Vita 102/108; misses → other = V2, safe); planning/web/agentic/math 1,524/1,525. AgenTracer "coding" 10/127: code written in chat with no shell/edits, so it gets V2 (acceptable).
- **2026-09-19, V7-code design run (paid; $0.568 vs est. $0.76; 0 errors; 33 s/trace; 15 chunks/trace on average).** `runs/longrca_dev_code50_v7.jsonl`; scorer `runs/v7_report.py`. Router: 50/50 code.
  - **Pre-registered go rule FAILED: V7 exact 9 vs V2 10** (±3: 14 vs 12, p=0.5). SWE38: 5 vs 6 (±3 9 vs 7); TB2 12: 4 vs 4. V7 with `best`: 7.
  - **Why:** the commit Noul saturates. Mean P(commit): gold 0.92, other handoffs 0.91, other agent steps 0.77. Jev calls most agent steps a "commit", so the product barely reorders V2. In 14/50 the gold is after the anchor chapter and so never examined (the look-back only goes backwards from the anchor).
  - **Positive:** **V2 on LongRCA's very long real coding failures (median 120K tokens): exact 10/50 = 20%, ±3 24%.** Paper reference, all domains: ECHO 13.2%, RCTA 24.1%. A code-only rule "first diagnostician→executor handoff" was 16/38 on SWE dev, but it is harness-specific and not part of the method.
  - V7-code (commit rule) is **not** taken to the held-out set. The router stays; the code branch = V2 for now.
- **2026-09-19, V8 = V2 whose look-back also checks chapters AFTER the alarm.** Built TDD: `backward_pass_typed(reach_after=…)`, `--reach-after N`, 107 tests pass. The router costs nothing (plain code), so it stays, dormant. Free check on dev code50: 12/50 golds lie after the alarm chapter: +1 chapter 5, +2 chapters 2, then scattered 6–11 chapters later.
  - **Pre-registered:** primary V8 = reach 2 + `step_plus_check` + `earliest_near_best`. **Go only if V8 exact > V2 exact (10/50)** on dev code50. Reach 1 and 3 are reported only. Add-on `runs/v8_reach.py` asks Jev only about the ≤3 chapters after the saved anchor (same prompts/questions as the pipeline). Estimate ~$0.08.
  - **Revised before any paid call:** why a fixed +2? The alarm over-fires on long coding traces: 0–16 CUSUM alarms per trace (calibrated on shorter AgenTracer). Options: A +2 / B health peak / C last alarm / D from the end (pure RTS) / E first health ≥ 0.5 / F reuse the forward suspects (free: 10 vs 10, no gain). **One fetch of every chapter after the alarm (~$0.28 est) scores A–E offline. PRIMARY = D (look back from the end); go only if D exact > 10/50.**
- **2026-09-19, V8 result (paid, $0.314 vs est. $0.28; 0 errors): FAILED, D 10/50 = V2 10/50.** A, B, C and E all 10/50 too (±3 all 12). It fetched 9.4 extra chapters per trace and changed 3 picks, none to the gold. The 12 golds only reachable after the alarm score only 0.12–0.52 of the top and rank 4th–149th of hundreds: **reach is not the bottleneck; Jev doesn't recognise these quiet causes.** V2 stays.
- **2026-09-19, PRE-REGISTERED final LongRCA measurement (fixed in writing before running):** V2 (typed, gap card, checks, `step_plus_check`, `earliest_near_best`, 10K windows) on the **locked `docs/heldout_longrca_swe90_ids.json` (90 SWE-bench Pro)** vs a free DeepSeek Flash whole-trace call (1M context; the held-out median is 120K tokens, max 368K). **Pass = (a) V2 exact > DeepSeek exact with sign-test p < 0.05, AND (b) V2 exact ≥ 13.2%** (ECHO, the paper's best earlier method; an imperfect comparison, since it covers all 5 domains). ±3 is reported. Held-out misses are never inspected.
  - Errors: a trace still erroring after retries counts as a miss for that method (fixed before results). Scorer: `runs/heldout_longrca_report.py`.
- **2026-09-19, FINAL LongRCA held-out verdict (locked SWE90, run once): FAIL by the pre-agreed bar.** V2 exact **7/90 (7.8%)** vs DeepSeek whole-trace 0/90: **bar (a) PASS** (7 vs 0, p=0.016; ±3 15 vs 0, p=0.0001). **Bar (b) FAIL: 7.8% < 13.2%** (ECHO; all-domain, imperfect comparison). ±3 16.7%. V2 $1.144 (est. $1.17), 48 s/trace, 17 chunks/trace, 0 errors. DeepSeek (free, 1M context) 14 s/trace, 0 errors; it collapses to step 0/1 on 27/90 (median trace 250 steps). The held-out set is now USED; misses are not inspected.
  - **Dev vs held-out:** SWE dev38 6/38 (16%) → held-out 7/90 (7.8%). The dev number was optimistic (small sample); the held-out is the honest estimate at this length (median 120K tokens).
- **2026-09-19, fair ECHO comparison, published numbers only (no re-implementation).** The LongRCA leaderboard (`github.com/longrca-bench/longrca-bench.github.io`, `data/leaderboard.json`) gives exact root-step counts per benchmark, all methods on DeepSeek-V4-Flash. **SWE-bench Pro (128): RCTA 49 (38.3%), ECHO 10 (7.8%), All-at-once 2 (1.6%), FALAT 3, Step-by-step 1, Binary search 1.** TB2 (42): RCTA 26.2%, ECHO 21.4%, All-at-once 7.1%.
  - **V2 on the same 128 SWE traces: 13/128 = 10.2%** (held-out 7/90 = 7.8% + dev 6/38). V2's config was frozen on TrajErrBench before LongRCA arrived; the LongRCA dev run only added independent V7 questions to the same calls, so V2 was never tuned on LongRCA and all 128 are fair. On the held-out 90 alone: 7.8%, the same as ECHO's 7.8% over 128.
  - **Verdict: V2 is on par with ECHO on SWE-bench Pro** (13 vs 10 of 128; counts only, no per-trace predictions, so no paired test, and the gap is within noise). It is far above all-at-once / step-by-step / binary search, which matches our own DeepSeek whole-trace 0/90, and **far below RCTA (38.3%)**. The earlier 13.2% bar compared V2's SWE score with ECHO's all-domain average (unfair); the pre-registered FAIL stands as recorded.
  - **Unpaired Fisher exact tests on the totals** (the leaderboard publishes no per-trace answers, so no paired test): V2 vs ECHO 13 vs 10 of 128, p=0.66 (tie); V2 vs RCTA 13 vs 49, p=1.8e-7. **RCTA** (Root-Cause Trajectory Attribution, the paper's training-free method): segment the trace (5-step overlap) → one call per segment for a summary and candidate steps → one outline call by subgoal → **handoff tracing**: compare each candidate's text with the retrieved earlier handoff instructions (m+2 calls). This matches our free dev finding that the gold is often the first diagnostician→executor handoff.
  - **V9 result (free, 0 errors): FAILED the gate.** LongRCA dev50 exact **8 vs V2 10** (±3 13 vs 12; only V9 5, only V2 7); TEB dev80 **19 vs 21** (±3 32 vs 38). The gold was among the candidates in 31/50 (20 of them a handoff), but DeepSeek picked it only 8 times, and a gold handoff only 5/20. Misses with the gold present went later 16 vs earlier 7: it still picks the executor's visible action, not the instruction that caused it. Not run on the 128 for the RCTA comparison (the gate failed, so the question of a second held-out use is moot). V2 stays.
- **2026-09-19, CORRECTION (found while writing up):** V5 (`runs/blame_test.py`), V6 (`runs/deepseek_b_test.py`) and V9-TEB (`runs/v9_instruction.py`) took "V2" on TrajErrBench dev80 from `runs/teb_dev80_v2.jsonl`. That is the OLD run (symptom card, no checks; `step_plus_check` rescoring there = SWE 5 + τ² 16 = 21). The real V2 answers are in `runs/teb_dev80_v3.jsonl` (gap card + checks: SWE 3 + τ² 18 = 21). **Verdicts unchanged** (same 21 total; V5 10, V6 18, V9 19 all < 21), but in those TEB tests the chapter/shortlist A came from the older run. The cascade/V4 fusion checks may have used the same file. The LongRCA tests (V7, V8, V9-LongRCA, held-out) used true V2 rows.
- **2026-09-19, write-up rewritten:** V2 only, the ADAS framing (forward filter, CUSUM FDI, RTS-style smoother, gating/association, onset vs alarm), origin story, Jev as a sensor with designed channels (question = channel, answer type = measurement type), datasets and results, and comparison of accuracy/speed/cost/model.
- **2026-09-19, RCTA method vs V9: line-by-line diff (free).** Source: arXiv 2608.15242 HTML, §5 + Appendix E (prompts and thresholds) + Appendix F (pseudocode), read verbatim (the alphaxiv MCP is not connected in this session). **No RCTA code is released**; the "run artifacts" with frozen prompts are not public, and only the leaderboard repo exists.
  | Stage | RCTA (paper) | V9 (`runs/v9_instruction.py`) |
  |---|---|---|
  | Segments | Rule-based, ≤32,000 chars and ≤80 steps; boundaries at handoffs/finish/verifier PASS-FAIL signals; 5-step overlap; no LLM | V2's 10K-token chapters, 1,250-token overlap; no structure-aware boundaries |
  | Candidate recall | 1 DeepSeek call per segment (thinking OFF, 4K out) → summary + candidate IDs with evidence; pooled; **up to 80 candidates** | V2's Jev look-back → **top-5** agent steps |
  | Outline | 1 call (thinking HIGH, 6K out) → 3–8 subgoal phases (fallback: one phase per summary; 308/1,140 fell outside 3–8) | none; V2 gap card (request + last 2 steps + forward symptom) instead |
  | Handoff retrieval | executor/verifier candidate → nearest earlier `X (-> Y)` addressed to its role Y; other candidates → nearest earlier handoff as plan context; handoffs added to the candidates | latest earlier message from the user or any different speaker; only agent-written ones added. Measured on dev50: 218/309 candidates got the same instruction as RCTA's rule (all executor/verifier ones); 91 differ: 52 got the user issue where RCTA has none, 36 handoff candidates got the executor's previous message instead of the earlier handoff, 3 got JudgeAgent |
  | Final call | outline + all segment summaries + **full** original candidate text + handoffs; **thinking HIGH**, 8K out (16K/32K for long ones) | gap card + candidates **clipped** (instruction 800, step 1,000 tokens); **thinking OFF**; strict JSON {reason, step} |
  | Decision rule | instruction contains the decisive error and the step carries it out unrepaired → the instruction; the step departs or adds a new error → the step; **repaired errors excluded**; role predicted independently | similar ("carrying out a wrong instruction is not the root; the instruction's giver is") but no repaired-error rule and no role |
  | Validation | programmatic ID/role/quote-grounding checks, **1 retry with feedback**, then abstain | invalid → V2's pick, no retry |
  | Calls | m+2 DeepSeek calls (m segments) | ~24 Jev calls (V2) + 1 DeepSeek |
  - **Which explanation is real:** the gold was already among V9's candidates in 31/50 traces, yet V9 picked it only 8 times. So on most traces **V9 lost at the final selection, not at candidate recall.** Recall (5 vs up to 80) only bounds the other 19. "Outline" is one of several final-stage differences, alongside **thinking HIGH vs OFF**, full vs clipped text, the addressed-handoff rule for handoff candidates, the repaired-error rule and retry-on-invalid; the paper gives no ablation separating them. The cleanest free next test would change one factor at a time in V9's final call (e.g. thinking HIGH only).
- **2026-09-19, V9-think (free):** identical to V9 (same script, inputs, K, candidates and prompt), with ONE change: DeepSeek thinking on (`ALIBI_JUDGE_REASONING=high`, as in RCTA's final call). Output `runs/v9_think_{longrca,trajerrbench}.jsonl`. **Same pre-registered rule: PASS only if exact > 10 on LongRCA dev50 AND ≥ 21 on TEB dev80.** (TEB inputs are the same older-run rows as V9, to keep the test one-factor; the true V2 also scores 21 there.) Written before any result was seen.
  - **V9-think result (free; all 130 done; 4 empty free-tier responses retried; ~10 min per call with thinking on): FAILED.** LongRCA dev50 exact **8 vs V2 10** (same as V9 without thinking; ±3 12 vs 12); TEB dev80 **21 vs 21** (up from 19; ±3 37 vs 38). **Reasoning effort was not the missing ingredient on the long multi-agent coding traces.** The remaining untested differences from RCTA are its own candidate recall (up to 80 per trace), the subgoal outline, full-text candidates, the addressed-handoff rule for handoff candidates, and the repaired-error rule. V2 remains the method; RCTA chasing stops here.
- **2026-09-20, PRE-REGISTERED WebArena measurement (no DeepSeek baseline this time).** V2 (same settings as the SWE held-out) on `docs/heldout_longrca_webarena48_ids.json` = the first 48 of the 100 locked WebArena traces (median ~5 chapters). Estimate $0.166 (worst case +25% $0.21).
  - **Comparison (published only, not re-run):** LongRCA leaderboard, WebArena Verified, 177 traces, all on DeepSeek-V4-Flash: RCTA 59 (33.3%), ECHO 46 (26.0%), All-at-once 30 (16.9%), Step-by-step 20 (11.3%), Binary search 9 (5.1%), FALAT 4 (2.3%). The WebArena papers report task success only, with no root-step numbers.
  - **Rule:** report V2 exact (and ±3) on the 48 with a 95% interval, and an unpaired Fisher exact test against each published method. Claim "beats" or "trails" a method only where p < 0.05; otherwise "no clear difference". The subsets differ (48 of the 177), so the comparison is indicative.
  - **Result (Jev $0.180 vs est. $0.166; 0 errors; 4.6 chapters/trace, median 38K tokens, 15.4 s/trace summed): V2 exact 6/48 = 12.5%** (95% CI 4.7–25.2%), ±3 10 (20.8%), ±5 12 (25.0%). Fisher vs published (177 traces): **trails RCTA** 33.3% (p=0.004); no clear difference vs ECHO 26.0% (p=0.054), All-at-once 16.9% (p=0.52), Step-by-step 11.3% (p=0.80), Binary search 5.1% (p=0.10); **beats FALAT** 2.3% (p=0.007). Point estimate below ECHO and all-at-once: on web tasks the method is **not better** than a whole-trace read on DeepSeek-V4-Flash. The other 52 locked WebArena traces are unused.
- **2026-09-20, shipped: MCP server + Claude Code plugin.** V2 is now reachable through one entry point, `alibi.localize.diagnose()` (10K chapters, CUSUM k=0.2/h=0.5, gap card + checks, `step_plus_check`, `earliest_near_best`), behind a 50K-token length gate: shorter traces are reported as "read it directly" without a judge call. Surfaces: `alibi diagnose`, the `alibi-mcp` server (one `diagnose_trace` tool) and the plugin in `plugin/`, installable with `/plugin marketplace add ahmedezz26/alibi`. Trace inputs: JSON files, LangSmith ids and Claude Code session transcripts (best effort; a 429K-token session parses to 1,341 steps). Repo published at https://github.com/ahmedezz26/alibi (MIT, 128 tests, no paid call in the suite).
- **2026-09-20, 0.1.1: published to PyPI as `agent-alibi`** (the name `alibi` is taken). The Claude Code plugin now launches `uvx --from 'agent-alibi>=0.1,<0.2' alibi-mcp` instead of building the repository, and `uv tool install agent-alibi` installs the command line. A code review of the release found that 0.1.0 had been published from an untagged commit, so its tag could not rebuild it; 0.1.1 is cut from its own tag and the release workflow now refuses to run unless the ref is a tag matching the project version and that version is absent from PyPI. A **cost ceiling** was added with it: `ALIBI_MAX_TRACE_TOKENS` (default 250,000, about $0.02 of Jev) refuses a longer trace with an estimate rather than spending it, since the plugin sets `ALIBI_ALLOW_PAID_MODELS=1` and a 400K-token session transcript would otherwise bill ~55 chapters unannounced.
- **2026-09-20, 0.1.2: usage documentation, and the three defects writing it exposed.** The README now documents the trace formats (a JSON step list, Claude Code `.jsonl` sessions, LangSmith ids), how to find a session file, how to read the output, and the `diagnose_trace` signature for other MCP clients. Writing it found that (a) both the CLI and the MCP server asked for the paid Jev backend on traces *over* the cost ceiling, which `diagnose()` refuses for free, (b) the ceiling could not be raised from the command line, and (c) a mistyped path fell through to the LangSmith adapter and failed with a UUID error from their API. All three are fixed with tests; `--max-tokens` joins `--min-tokens`, and a path-shaped argument that does not exist is now a clean error with exit code 2.
- **2026-09-20, 0.1.3: a privacy hole and a broken path, both found by review.** The MCP server built whatever judge the settings named, with no backend check, so a hand-wired server or one started where a `.env` holds `OPENROUTER_API_KEY` would have sent a private trace to a free OpenRouter endpoint (which may log prompts) before failing on the typed schema; it now refuses before anything is sent. 0.1.2's missing-file guard also rejected `~`-prefixed paths, which is exactly what the plugin's skill tells MCP clients to pass, and broke `JsonFileTraceSource`'s `<dir>/<id>` lookup; paths are expanded and both forms work. The "does this need a judge" test is now one predicate beside the gates it mirrors, because the duplicated copies had drifted: an empty trace with the gate at 0 demanded a paid backend for a question answered for free. `--min-trace-tokens`/`--max-trace-tokens` replace the ambiguous `--max-tokens`, which meant the window size on `score` and `eval`.
- **2026-09-20, correction to the 0.1.3 entry above, and the fixes that follow it.** That entry claimed `<dir>/<id>` lookup worked again; it only did with an explicit `--source json`. In the default `auto` mode the argument still had no suffix, so it was routed to LangSmith and the local path was POSTed to their API - worse than the error it replaced. The regression test that certified the fix pinned `source="json"`, the one mode where it worked. Resolution now returns the file that was found and dispatch reads that, with a test for the default mode. Also fixed in the same pass: `make_judge` was called unguarded, so an unset `TYPESAFE_API_KEY` - the likeliest first run for a plugin user - surfaced as a contentless MCP error; malformed, wrong-shaped and bad-timestamp files gave tracebacks or contentless errors instead of one `TraceFormatError`; and both backend messages omitted `ALIBI_ALLOW_PAID_MODELS=1`, so following them exactly hit a second failure.
- **2026-09-20, the two deferred items closed.** `diagnose` now takes a `judge_factory` and owns the gate decision, so the trace is tokenized once per call rather than twice (measured: 1 pass, was 2) and the "does this need a judge" test exists in one place instead of being duplicated in both surfaces; `needs_judge` is gone with its last caller. The gate flags are `--min-trace-tokens` and `--max-trace-tokens` only: keeping 0.1.2's `--min-tokens`/`--max-tokens` as aliases had made `--min-t` an ambiguous prefix.
