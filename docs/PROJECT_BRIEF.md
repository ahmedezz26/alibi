# Alibi — Project Brief

*Agent trace triage that checks whether your agent's decisions can account for themselves — and backtracks to find the one that can't, once something's gone wrong.*

**Purpose of this file:** Give Claude Code (or any coding agent) enough context to scaffold and build the MVP without re-explaining the idea from scratch each session. Keep this file updated as decisions change — it's the source of truth for *why*, not just *what*.

---

## 1. Problem

AI agents (especially long-running, tool-heavy ones) produce huge execution logs ("traces") — hundreds of tool calls per run, thousands of runs. Finding *why* an agent failed inside one of these traces is currently either:
- Manual (impossible at scale), or
- Done with a single expensive LLM reading the whole trace (slow, costly, doesn't scale to production volume)

Cheap small "judge" models can classify/score short windows of a trace fast and cheaply, but typically have small context windows, so they can only ever see a *slice* of a long trace at once — never the whole thing.

## 2. Core Thesis

Treat a trace as a **time series**, and windowed judge-model calls as **noisy, local sensor readings** over that series. Naive per-window classification (score each chunk independently, flag if score > threshold) will catch locally-obvious failures but is structurally blind to:
- Slow drift / gradual degradation across many steps
- Failures whose root cause is an early decision that only matters much later

The fix: borrow techniques from state estimation / signal processing (Kalman-style filtering, change-point detection, smoothing) rather than treating each window as independent.

**We are not betting the product on any single judge-model vendor.** The judge model is a pluggable backend. The moat is the estimation/filtering layer on top, not "we call Jev in a loop."

## 3. Architecture

### 3.1 Trace ingestion
- Traces come from LangSmith (or other tracing backends — design for this, don't hardcode LangSmith).
- A trace = ordered sequence of steps (tool calls, LLM calls) sharing a trace ID.
- Each step: `{step_id, type, timestamp, inputs, outputs, error}` at minimum.
- Build an adapter interface: `TraceSource.get_trace(trace_id) -> list[Step]`, with a LangSmith implementation first.

### 3.2 Chunking / windowing
- Split a trace into overlapping windows sized to fit the judge model's context budget (configurable, default ~28K tokens to leave headroom).
- Overlap windows (don't hard-cut) to avoid splitting the failure moment across two chunks.
- `chunk_trace(steps, max_tokens, overlap_tokens) -> list[Chunk]`

### 3.3 Judge model interface (pluggable — this is important)
```
class Judge(Protocol):
    def evaluate(self, state: str, questions: list[Question]) -> dict[str, Any]: ...
```
- Implementations:
  - `OpenRouterJudge` — first and default backend. Uses the `openai` Python SDK pointed at `base_url="https://openrouter.ai/api/v1"`, authenticated with `OPENROUTER_API_KEY`. Model ID comes from `ALIBI_JUDGE_MODEL` (default `openai/gpt-5-nano`, an OpenRouter model ID). Typed questions use `response_format={"type": "json_schema", ...}`. Records `response.usage` (tokens + `cost`) and wall-clock latency per call.
  - `FallbackLLMJudge` (any capable model via simple prompt+schema), maybe a self-hosted small open-weight model later.
- The `Judge` protocol stays vendor-neutral — OpenRouter is a backend, not the abstraction.
- Nothing else in the codebase should import a specific judge implementation directly — always go through this interface.

### 3.4 Forward pass: state propagation (Kalman-style)
- Don't call the judge on each chunk independently. Carry a **compressed running state** forward between chunks: a short summary of "what's the agent's plan / progress / anything suspicious so far."
- Each call: `judge.evaluate(state=f"{prior_state}\n{chunk}", questions=[...anomaly questions..., "update state summary"])`
- New state = judge's own summary output (bounded length, e.g. <150 tokens) — must be validated empirically that this doesn't lose signal (see §6, open risk).

### 3.5 Drift detection (CUSUM / Page-Hinkley, not raw thresholding)
- Track cumulative deviation of the anomaly score over the sequence of chunks, not just per-chunk threshold crossing.
- This is what catches slow degradation that never spikes any single window above threshold.
- `cusum_detector(scores: list[float], k: float, h: float) -> list[int]` — returns indices of detected change points.

### 3.6 Multi-resolution passes (coarse-to-fine)
- Pass 1: large windows, coarse/cheap questions, across the *entire* trace set. Purpose: cast a wide net.
- Pass 2: only on flagged traces/regions — smaller windows, more overlap, sharper diagnostic questions.
- Pass 3: on the smallest flagged set, either another judge pass or hand off to a larger/expensive model for root-cause narrative.

### 3.7 Backward smoothing (RTS-style) — the differentiator
- Once a failure is located (forward pass, at step N), run a **backward pass**: re-examine earlier chunks (1..N) with hindsight — "given that the agent ended up doing X at step N, does anything in this earlier chunk look like a causal precursor?"
- This is the piece a pure forward-only filter structurally cannot do — it doesn't know what to look for until it's seen the failure.
- This is the primary algorithmic bet of the project — prioritize validating this first (see §5).

## 4. Data Strategy (in priority order — do not skip to production data first)

1. **AgentRx benchmark** (`microsoft/AgentRx` on Hugging Face, gated) — 73 failed agent trajectories in the public release (29 τ-retail + 44 Magentic-One), each manually annotated with a root-cause failure step + failure category. Use this FIRST to validate the backward-smoothing pass: does it correctly localize the annotated critical step? This gives a real precision/recall number before writing any judge-model-specific code.
2. **Public trajectory benchmarks** for broader/general testing: Toolathlon (108 tasks, 32 apps, 604 tools, ~20 turns/task), τ²-bench, SWE-bench, TerminalBench, MCPBench, GAIA deep-research trajectories. Free, real, on Hugging Face.
3. **ToolMisuseBench-style fault injection** — build a harness that runs an open-source agent (e.g. LangGraph ReAct loop, browser-use) on benign tasks and deliberately injects a controlled fault (truncated tool response, stale observation, wrong parameter) at a known, recorded step. This gives unlimited labeled data on demand — conceptually identical to HIL fault injection testing.
4. **Design partners** — only after 1–3 show signal. Real production traces, but requires a redaction/anonymization step before ingestion (traces will contain real customer data / API responses). Treat this as real engineering + sales cost, not a formality.

## 5. Build Order (MVP first, don't build everything at once)

1. Trace ingestion adapter (LangSmith) + chunking with overlap
2. `Judge` interface + one working backend: `OpenRouterJudge` (model from `ALIBI_JUDGE_MODEL`, default `openai/gpt-5-nano`)
3. Forward pass: per-chunk scoring only (no state propagation yet) — get *something* end-to-end working
4. Add state propagation between chunks
5. Add CUSUM drift detection on top of the score sequence
6. **Validate against AgentRx**: does the pipeline (with backward pass) correctly find the annotated critical failure step? Record precision/recall. This is the go/no-go checkpoint.
7. If validated: build the fault-injection harness for synthetic labeled data at scale
8. Multi-resolution (coarse-to-fine) passes
9. Evaluate/swap nano-class models (via `ALIBI_JUDGE_MODEL`, or another `Judge` backend) as the production judge once cost/latency at scale actually matters
10. Design partner pilot

Do not build 7–10 before step 6 passes. Step 6 is the point of the whole project — validate the algorithm before optimizing cost or productizing.

## 6. Tech Stack (proposed, adjust freely)
- Python, managed with `uv`
- `langsmith` SDK for trace ingestion
- Pluggable judge interface as above; first implementation is `OpenRouterJudge` via the `openai` SDK (OpenRouter is OpenAI-compatible)
- Config via env vars (`OPENROUTER_API_KEY`, `ALIBI_JUDGE_MODEL`) loaded from a git-ignored `.env`, with a committed `.env.example`
- `numpy` for CUSUM/signal processing
- Simple local JSON/parquet storage for traces + results during prototyping — no need for a database yet

## 7. Success Criteria for MVP
- Backward-smoothing pass correctly identifies the AgentRx-annotated critical step for >X% of the 73 annotated trajectories (set X after first run — no prior baseline exists yet)
- Pipeline runs end-to-end on a full public benchmark set (e.g. Toolathlon) without manual intervention
- Cost/latency measured and logged per pass, per judge backend, so backend swaps are comparable apples-to-apples

## 8. Open Risks / Questions (do not assume these are solved)
- **State compression may be lossy in ways that matter.** "Summarize plan + suspicion in <150 tokens" could silently drop the exact signal the backward pass needs. Needs empirical checking, not just implementation.
- **CUSUM thresholds need calibration data.** No labeled "normal variance" baseline exists yet for agent traces the way it does in, e.g., ADAS fault detection. AgentRx + fault injection are the initial substitute for this.
- **Whichever nano-class model backend you pick will likely be early-stage and may change API/pricing/availability.** Do not hard-couple architecture to any single vendor's interface — this is the whole reason for the pluggable `Judge` abstraction.
- **Not every OpenRouter model/provider supports strict `json_schema` structured outputs.** Check the model's supported parameters before changing `ALIBI_JUDGE_MODEL`.
- **Production trace data will contain sensitive content** (customer data, internal API payloads) — redaction strategy needs to exist before any design-partner data touches the system.

## 9. Glossary (for a coding agent without prior context on this conversation)
- **Trace**: full recorded execution of one agent run (tree of steps/spans, e.g. from LangSmith)
- **Chunk/window**: a token-budget-limited slice of a trace's steps
- **Judge model / nano-class model**: a small, cheap, fast model with a small context window (default: `openai/gpt-5-nano` via OpenRouter) used to score/classify chunks via typed questions
- **State propagation**: carrying a compressed summary forward between chunk evaluations instead of treating each chunk independently
- **CUSUM**: cumulative sum control chart — detects gradual drift by accumulating deviations over time, not just single-point thresholds
- **Backward/RTS smoothing**: re-examining earlier data with the benefit of hindsight from a later detected event, to find causal precursors
