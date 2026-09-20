"""The product entry point: run V2 (forward filter, CUSUM, look-back; Jev as the sensor) on
one trace and return the top suspect steps. Shared by the CLI, the MCP server and the plugin.

Only long traces are analysed: below ``settings.min_trace_tokens`` a single direct read is
enough, and above ``settings.max_trace_tokens`` the Jev bill would be larger than the caller
is likely to have agreed to. In both cases no judge call is made.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from alibi.backward import backward_pass_typed, combine_scores, suspect_probs
from alibi.chunking import chunk_trace, render_step, trace_tokens
from alibi.config import Settings, load_settings
from alibi.drift import detect_drift
from alibi.forward import forward_pass_typed
from alibi.judges import Judge, make_judge
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
# Below this a trace is too short for "every step is its own chapter" to mean anything.
MIN_STEPS_TO_GROUP = 4
# A tenth of the measured chapter size. A window this small is a typo, not a choice - and
# requiring it keeps the guard below from ever firing on a default or deliberate setting,
# including a trace whose steps are each genuinely larger than a chapter.
DEGENERATE_WINDOW_TOKENS = 1_000
# Forward pass plus look-back: the measured runs cost about twice the trace's tokens.
READS_PER_TOKEN = 2


SUPPORTED_BACKEND = "typesafe"
BACKEND_HELP = (
    "Alibi needs the Jev backend: set ALIBI_JUDGE_BACKEND=typesafe, TYPESAFE_API_KEY and "
    "ALIBI_ALLOW_PAID_MODELS=1. Nothing was sent."
)


PAID_HELP = (
    "the Jev (TypeSafe) judge is paid: set ALIBI_ALLOW_PAID_MODELS=1 (only with the user's "
    "explicit approval), and TYPESAFE_API_KEY if it is not set yet - this check runs first, "
    "so it cannot tell you whether the key is there. Nothing was sent."
)


def unsupported_backend(settings: Settings) -> str | None:
    """Why this trace cannot be judged, or None when the backend is usable. Jev is the only
    sensor the method was measured with, and free endpoints may log prompts."""
    if settings.judge_backend != SUPPORTED_BACKEND:
        return BACKEND_HELP
    return None


def spend_not_allowed(settings: Settings) -> str | None:
    """Why this trace must not be billed, or None when spending is allowed. Jev has no free
    tier, so every analysed trace costs money; only the backend above reaches this check."""
    if not settings.allow_paid_models:
        return PAID_HELP
    return None


class JudgeUnavailable(Exception):
    """The judge cannot be built: an unsupported backend, a missing key, or the spend guard.

    Deliberately not a RuntimeError, so a surface can catch it without also catching failures
    raised by the judge while it is analysing a trace, which have already cost money.
    """


class SettingsError(ValueError):
    """A setting the pipeline cannot work with, named as the user set it.

    A ValueError so it stays compatible with callers that caught one, but its own type so a
    surface can report it as configuration without also swallowing internal ValueErrors.
    """


class AnalysisFailed(Exception):
    """The pipeline failed once the judge was reading the trace.

    Raised only after the judge exists, which ``diagnose`` arranges to mean "about to send".
    ``completed`` counts the judge calls that finished during *this* diagnosis, i.e. what was
    billed. At 0 nothing is billed and the first request may never have left the machine -
    the judge can also fail client-side, so neither the spend nor the send is asserted. It is
    ``None`` when the judge keeps no call log, which is unknown, not nothing: ``Judge`` only
    promises ``evaluate``, and a host may inject a client that does its own accounting.
    """

    def __init__(self, cause: Exception, completed: int | None) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.completed = completed

    def surface_message(self, *, cause_detail: bool = True) -> str:
        """What a user is told: what broke, and exactly what it cost.

        ``cause_detail`` carries the cause's own message, which is the judge client's and may
        quote the chapter it was given. Keep it for a local terminal; drop it where the string
        is relayed onwards, and the type name still says what kind of failure it was.
        """
        what = type(self.cause).__name__
        if cause_detail:
            what = f"{what}: {str(self.cause)[:200]}"
        if self.completed is None:
            cost = "This judge keeps no call log, so what was billed is unknown; check with Jev."
        elif self.completed:
            cost = f"{self.completed} Jev call(s) completed and are billed."
        else:
            cost = (
                "No Jev call completed, so nothing is billed, though the first request "
                "may already have been sent."
            )
        return f"Analysis failed while the judge was reading the trace ({what}). {cost}"


def call_count(judge: Judge) -> int | None:
    """How many calls this judge has logged, or None if it logs none.

    ``Judge`` only promises ``evaluate``; ``calls`` is a convention both shipped backends
    follow. The distinction matters because these numbers are told to a user as money.
    """
    calls = getattr(judge, "calls", None)
    return None if calls is None else len(calls)


def build_judge(settings: Settings, make: Callable[[], Judge] | None = None) -> Judge:
    """The judge both surfaces use, with one policy and one error type.

    ``make`` replaces the *construction* of the client - a test fake, or a host that wires up
    its own - but not the policy deciding whether to build one at all: the backend check and
    the spend guard run either way, so an injected judge cannot sidestep them. Holding a key
    is part of construction, so an injected judge is expected to bring its own.
    """
    for problem in (unsupported_backend(settings), spend_not_allowed(settings)):
        if problem:
            raise JudgeUnavailable(problem)
    try:
        return (make or (lambda: make_judge(settings)))()
    except RuntimeError as e:  # a missing key, or the paid-model guard
        raise JudgeUnavailable(f"{e}. Nothing was sent.") from e


def gate(steps: list[Step], settings: Settings) -> tuple[int, str | None]:
    """The trace's size, and why it will not be analysed (None when it will be). Every
    refusal costs nothing: no judge is built, nothing is kept, and nothing leaves the
    machine. ``trace_tokens`` renders one step at a time and holds none of them."""
    tokens = trace_tokens(steps)
    if not steps:
        return tokens, "The trace has no steps to analyse."
    if tokens < settings.min_trace_tokens:
        return tokens, (
            f"Trace is {tokens:,} tokens, under the {settings.min_trace_tokens:,}-token "
            "threshold: a single direct read is enough; Alibi adds value on long traces."
        )
    if tokens > settings.max_trace_tokens:
        return tokens, (
            f"Trace is {tokens:,} tokens, over the {settings.max_trace_tokens:,}-token "
            f"ceiling: analysing it would cost about ${_estimated_cost(tokens, settings):.2f} "
            "of Jev. Raise ALIBI_MAX_TRACE_TOKENS to analyse it anyway."
        )
    return tokens, None


class Suspect(BaseModel):
    step: int = Field(description="0-based step index in the trace.")
    score: float = Field(description="Fused evidence score (higher = more likely the cause).")
    chapter: int = Field(description="Index of the 10K-token chapter holding the step.")
    step_type: str
    name: str | None = None
    preview: str = Field(description="Start of the step's rendered text.")


class Diagnosis(BaseModel):
    gated: bool = Field(
        description=(
            "True when the trace was not analysed and nothing was spent: it is empty, under "
            "the length gate, or over the cost ceiling. `message` says which."
        )
    )
    message: str
    trace_tokens: int
    n_steps: int
    n_chapters: int = 0
    alarm_chapter: int | None = None
    anchor: str | None = None
    suspects: list[Suspect] = Field(default_factory=list)
    judge_calls: int | None = Field(
        default=0, description="Judge calls made for this trace; null if the judge logs none."
    )
    judge_seconds: float | None = 0.0
    cost_usd: float | None = Field(
        default=0.0, description="Billed for this trace in USD; null if the judge logs none."
    )


def diagnose(
    steps: list[Step],
    judge: Judge | None = None,
    settings: Settings | None = None,
    *,
    judge_factory: Callable[[], Judge] | None = None,
) -> Diagnosis:
    """Localize the failure in one trace. Give it a ``judge``, or a ``judge_factory`` that is
    called only if the trace is actually analysed - so a caller need not build a paid client,
    or hold a key, for a trace that will be refused. ``settings`` defaults to the environment.

    The reported cost is this trace's share of the judge's call log, so a factory may return
    a judge it reuses between diagnoses. It must not return one that is *concurrently* in
    another diagnosis, though: the calls interleave in one log and each run would report the
    other's spend. One judge per diagnosis is always safe.

    The gate runs before anything is rendered or kept, so a refusal costs one streaming pass
    over the trace and nothing else. The judge is built last, immediately before the first
    call, so "a judge exists" means "the trace is about to be sent": everything that can go
    wrong for free - the gate, a bad window size - has already gone right. A failure after
    that point is an ``AnalysisFailed``, which carries what it cost.
    """
    if judge is not None and judge_factory is not None:
        raise ValueError("pass judge or judge_factory, not both")
    settings = load_settings() if settings is None else settings
    tokens, refusal = gate(steps, settings)
    if refusal is not None:
        return Diagnosis(gated=True, message=refusal, trace_tokens=tokens, n_steps=len(steps))
    window = settings.judge_window_tokens
    try:  # a bad window size costs nothing here: no judge exists yet
        chunks = chunk_trace(steps, window, window // 8)
    except ValueError as e:
        raise SettingsError(f"ALIBI_JUDGE_WINDOW_TOKENS={window} cannot chunk a trace: {e}") from e
    # Chapters can never outnumber steps, so reaching that means no step shared a chapter
    # with another: nothing would be read in context, the gate's estimate assumed chapters,
    # and the run would be a judge call per step. Only refused when the window is also far
    # below the measured chapter size, because the same shape is legitimate when the steps
    # themselves are larger than a chapter. Caught here: after the next line it is billable.
    if (
        len(steps) >= MIN_STEPS_TO_GROUP
        and len(chunks) >= len(steps)
        and window < DEGENERATE_WINDOW_TOKENS
    ):
        raise SettingsError(
            f"ALIBI_JUDGE_WINDOW_TOKENS={window:,} is too small for this trace: each of its "
            f"{len(steps):,} steps became its own chapter, so no step would be read beside "
            f"its neighbours and the run would make about {2 * len(steps):,} judge calls. "
            "The method was measured with 10,000-token chapters."
        )
    if judge is None:
        if judge_factory is None:
            raise ValueError("a typed (Jev) judge is required for traces above the length gate")
        judge = judge_factory()
    # Counted from here, not from zero: a host may reuse one judge across traces, and these
    # numbers are a money claim about this trace.
    before = call_count(judge)
    try:
        return _analyse(judge, chunks, steps, tokens, before or 0)
    except Exception as e:  # noqa: BLE001 - re-raised with what it cost
        after = call_count(judge)
        spent = None if before is None or after is None else after - before
        raise AnalysisFailed(e, spent) from e


def _analyse(judge: Judge, chunks, steps: list[Step], tokens: int, before: int = 0) -> Diagnosis:
    """The V2 pipeline itself. Every judge call lives in here, so its caller can say what a
    failure cost without guessing. ``before`` is the judge's call count on entry."""
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
    logged = getattr(judge, "calls", None)
    calls = None if logged is None else logged[before:]  # this trace's, not the judge's life
    return Diagnosis(
        gated=False,
        message=f"Read these {len(suspects)} steps first, in order.",
        trace_tokens=tokens,
        n_steps=len(steps),
        n_chapters=len(chunks),
        alarm_chapter=alarms[0] if alarms else None,
        anchor=loc.anchor,
        suspects=suspects,
        judge_calls=None if calls is None else len(calls),
        judge_seconds=None if calls is None else round(sum(c.latency_s for c in calls), 1),
        cost_usd=None if calls is None else round(sum(c.cost or 0 for c in calls), 5),
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
