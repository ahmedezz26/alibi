"""The product entry point: run V2 (forward filter, CUSUM, look-back; Jev as the sensor) on
one trace and return the top suspect steps. Shared by the CLI, the MCP server and the plugin.

Only long traces are analysed: below ``settings.min_trace_tokens`` a single direct read is
enough, and above ``settings.max_trace_tokens`` the Jev bill would be larger than the caller
is likely to have agreed to. In both cases no judge call is made.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from alibi.backward import backward_pass_typed, combine_scores, suspect_probs
from alibi.chunking import chunk_trace, render_step, trace_tokens
from alibi.config import Settings
from alibi.drift import detect_drift
from alibi.forward import forward_pass_typed
from alibi.judges import Judge
from alibi.types import Step

# V2, the adopted method (see docs/research-log.md).
V2_CONFIG = {
    "card": "gap",
    "checks": True,
    "rule": "step_plus_check",
    "pick": "earliest_near_best",
    "cusum_k": 0.2,
    "cusum_h": 0.5,
}
PREVIEW_CHARS = 300
# Forward pass plus look-back: the measured runs cost about twice the trace's tokens.
READS_PER_TOKEN = 2


class Suspect(BaseModel):
    step: int = Field(description="0-based step index in the trace.")
    score: float = Field(description="Fused evidence score (higher = more likely the cause).")
    chapter: int = Field(description="Index of the 10K-token chapter holding the step.")
    step_type: str
    name: str | None = None
    preview: str = Field(description="Start of the step's rendered text.")


class Diagnosis(BaseModel):
    gated: bool = Field(description="True when the trace was too short to analyse.")
    message: str
    trace_tokens: int
    n_steps: int
    n_chapters: int = 0
    alarm_chapter: int | None = None
    anchor: str | None = None
    suspects: list[Suspect] = Field(default_factory=list)
    judge_calls: int = 0
    judge_seconds: float = 0.0
    cost_usd: float = 0.0


def diagnose(steps: list[Step], judge: Judge | None, settings: Settings) -> Diagnosis:
    tokens = trace_tokens(steps)
    if not steps:
        return Diagnosis(
            gated=True,
            message="The trace has no steps to analyse.",
            trace_tokens=tokens,
            n_steps=0,
        )
    if tokens < settings.min_trace_tokens:
        return Diagnosis(
            gated=True,
            message=(
                f"Trace is {tokens:,} tokens, under the {settings.min_trace_tokens:,}-token "
                "threshold: a single direct read is enough; Alibi adds value on long traces."
            ),
            trace_tokens=tokens,
            n_steps=len(steps),
        )
    if tokens > settings.max_trace_tokens:
        return Diagnosis(
            gated=True,
            message=(
                f"Trace is {tokens:,} tokens, over the {settings.max_trace_tokens:,}-token "
                f"ceiling: analysing it would cost about ${_estimated_cost(tokens, settings):.2f} "
                "of Jev. Raise ALIBI_MAX_TRACE_TOKENS to analyse it anyway."
            ),
            trace_tokens=tokens,
            n_steps=len(steps),
        )
    if judge is None:
        raise ValueError("a typed (Jev) judge is required for traces above the length gate")
    window = settings.judge_window_tokens
    chunks = chunk_trace(steps, window, window // 8)
    fwd = forward_pass_typed(judge, chunks, steps)
    alarms = [
        cp.alarm
        for cp in detect_drift([s.score for s in fwd], V2_CONFIG["cusum_k"], V2_CONFIG["cusum_h"])
    ]
    loc = backward_pass_typed(
        judge,
        chunks,
        steps,
        fwd,
        alarms,
        known_failed=True,
        rule=V2_CONFIG["rule"],
        card=V2_CONFIG["card"],
        checks=V2_CONFIG["checks"],
        pick=V2_CONFIG["pick"],
    )
    ranked = list(loc.ranked)
    # Recompute the full score map: loc.ranked is truncated to TOP_K=3, but critical_step may
    # not be in it. Use combine_scores to get the authoritative score for every step.
    combined = combine_scores(
        loc.evidence,
        [s.score for s in fwd],
        [suspect_probs(s.answers) for s in fwd],
        V2_CONFIG["rule"],
    )
    # The pick (earliest near the top) leads, then the rest of the ranking.
    order = [loc.critical_step] + [i for i, _ in ranked if i != loc.critical_step]
    suspects = [_suspect(i, combined.get(i, 0.0), chunks, steps) for i in order[:3]]
    calls = getattr(judge, "calls", [])
    return Diagnosis(
        gated=False,
        message=f"Read these {len(suspects)} steps first, in order.",
        trace_tokens=tokens,
        n_steps=len(steps),
        n_chapters=len(chunks),
        alarm_chapter=alarms[0] if alarms else None,
        anchor=loc.anchor,
        suspects=suspects,
        judge_calls=len(calls),
        judge_seconds=round(sum(c.latency_s for c in calls), 1),
        cost_usd=round(sum(c.cost or 0 for c in calls), 5),
    )


def _estimated_cost(tokens: int, settings: Settings) -> float:
    """Jev bills input tokens. Every chapter is read once going forward and once coming back,
    so the trace is paid for about twice; overlap and the cards add a little on top."""
    return tokens / 1_000_000 * settings.typesafe_price_per_mtok * READS_PER_TOKEN


def _suspect(i: int, score: float, chunks, steps: list[Step]) -> Suspect:
    chapter = min(c.index for c in chunks if c.start <= i < c.end)
    return Suspect(
        step=i,
        score=round(score, 4),
        chapter=chapter,
        step_type=steps[i].type,
        name=steps[i].name,
        preview=_preview(i, steps[i]),
    )


def _preview(i: int, step: Step) -> str:
    """The step's own text when it has a `content` field, else its rendered form."""
    text = step.inputs.get("content")
    return str(text if text is not None else render_step(i, step))[:PREVIEW_CHARS]
