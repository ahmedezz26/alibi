"""Backward (RTS-style) smoothing pass, brief §3.7: the project's primary bet.

Once the forward pass locates a failure at chunk N, re-examine chunks 0..N with
hindsight: "given the agent ended up failing like this, is anything in this earlier
window a causal precursor?" The output localizes a single critical step.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from alibi.chunking import approx_tokens, clip_step, render_step
from alibi.forward import (
    HEALTH_LEVELS,
    SUSPECT_KEY,
    WARNING_SIGNS,
    ChunkScore,
    clip_tokens,
    task_start,
)
from alibi.judges import Judge
from alibi.types import Chunk, Question, Step

PRECURSOR_KEY = "precursor_score"
STEP_KEY = "critical_step"

BACKWARD_QUESTIONS = [
    Question(
        PRECURSOR_KEY,
        "How likely is it that the CURRENT window contains the root cause of the known later "
        "failure: the earliest step where the AGENT itself went wrong (a bad decision, tool "
        "call, argument, or claim) that made the failure inevitable? Correct tool outputs or "
        "observations are not root causes, even if the agent later mishandles them. "
        "0 = unrelated, 1 = this is clearly where it went wrong.",
        "number",
    ),
    Question(
        STEP_KEY,
        "The [step N] number, shown in the CURRENT window, of the agent's step most likely to "
        "be the root cause. Always name one, even if uncertain; express uncertainty through "
        f"{PRECURSOR_KEY} instead.",
        "integer",
    ),
    Question(
        "rationale", "One sentence explaining the causal link, citing step numbers.", "string"
    ),
]


@dataclass(frozen=True)
class PrecursorScore:
    chunk_index: int
    score: float
    step: int  # global step index inside the chunk (clamped into range if judge strayed)
    step_valid: bool  # False if the judge named a step outside the chunk and it was clamped
    answers: dict[str, Any]


@dataclass(frozen=True)
class Localization:
    failure_chunk: int
    critical_step: int
    precursors: list[PrecursorScore]
    anchor: str  # "forward": located by the forward pass; "outcome": known-failed run
    ranked: tuple[tuple[int, float], ...] = ()  # typed pass: top steps with probabilities
    evidence: tuple = ()  # typed pass: ChunkEvidence per examined chunk, in chunk order


DEFAULT_MIN_FAILURE_SCORE = 0.5

OUTCOME_FAILURE = (
    "The run ultimately FAILED: the agent did not correctly accomplish the user's task. "
    "Which step caused this is unknown; the failure may not look obvious locally."
)


def failure_chunk_index(
    forward: Sequence[ChunkScore],
    alarms: Sequence[int] = (),
    min_score: float = DEFAULT_MIN_FAILURE_SCORE,
) -> int | None:
    """First CUSUM alarm; else the highest-scoring chunk if it clears ``min_score``
    (earliest on ties); else None, meaning the forward pass located nothing."""
    if alarms:
        return alarms[0]
    best = max(forward, key=lambda s: (s.score, -s.chunk_index))
    return best.chunk_index if best.score >= min_score else None


def describe_failure(failed: ChunkScore) -> str:
    return (
        f"Later, in steps {failed.start}..{failed.end - 1}, the agent failed "
        f"(anomaly score {failed.score:.2f}): {failed.answers.get('rationale', '')}"
    )


def build_prompt(failure: str, forward: ChunkScore, chunk: Chunk) -> str:
    parts = [f"Known later failure:\n{failure}"]
    if forward.state_in:
        parts.append(f"Running summary before this window:\n{forward.state_in}")
    parts.append(f"Current window:\n{chunk.text}")
    return "\n\n".join(parts)


def backward_pass(
    judge: Judge,
    chunks: Sequence[Chunk],
    forward: Sequence[ChunkScore],
    alarms: Sequence[int] = (),
    known_failed: bool = False,
    min_failure_score: float = DEFAULT_MIN_FAILURE_SCORE,
    workers: int = 4,
) -> Localization:
    """With ``known_failed`` (an outcome signal, e.g. a failed task), fall back to
    anchoring on the last chunk when the forward pass located nothing."""
    n = failure_chunk_index(forward, alarms, min_failure_score)
    if n is not None:
        anchor, failure = "forward", describe_failure(forward[n])
    elif known_failed:
        n, anchor, failure = len(chunks) - 1, "outcome", OUTCOME_FAILURE
    else:
        n = max(forward, key=lambda s: (s.score, -s.chunk_index)).chunk_index
        anchor, failure = "forward", describe_failure(forward[n])

    def examine(chunk: Chunk) -> dict[str, Any]:
        return judge.evaluate(
            build_prompt(failure, forward[chunk.index], chunk), BACKWARD_QUESTIONS
        )

    # Each backward call depends only on the forward pass, so they run concurrently.
    candidates = list(reversed(chunks[: n + 1]))  # nearest the failure first
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        all_answers = list(pool.map(examine, candidates))

    precursors = []
    for chunk, answers in zip(candidates, all_answers, strict=True):
        step = int(answers[STEP_KEY])
        precursors.append(
            PrecursorScore(
                chunk_index=chunk.index,
                score=min(1.0, max(0.0, float(answers[PRECURSOR_KEY]))),
                step=min(max(step, chunk.start), chunk.end - 1),
                step_valid=chunk.start <= step < chunk.end,
                answers=answers,
            )
        )
    precursors.reverse()

    # Valid steps first (an out-of-window step means the judge misread that window), then
    # highest precursor score; ties go to the earliest chunk (root causes come first).
    best = max(precursors, key=lambda p: (p.step_valid, p.score, -p.chunk_index))
    return Localization(
        failure_chunk=n,
        critical_step=best.step,
        precursors=precursors,
        anchor=anchor,
    )


# --- Typed backward pass for System One judges (Jev): probabilities, no free text. ---

CHAPTER_KEY = "cause_in_window"
CARD_STEP_TOKENS = 1_500
RAISED_SIGN = 0.5
TOP_K = 3

CHAPTER_QUESTION = Question(
    CHAPTER_KEY,
    "The root cause of the known later failure happens in the current window: one of its "
    "steps is where the AGENT itself first went wrong.",
    "noul",
    {
        "true": "The agent makes the bad decision, tool call, argument, plan or claim that "
        "leads to the known failure in one of the current window's steps.",
        "false": "The current window's steps are fine, or they only show the effects of a "
        "mistake made in another window.",
    },
)

STEP_QUESTION = (
    "[step {i}] in the current window is the root cause of the known later failure: the "
    "earliest point where the AGENT itself went wrong (a bad decision, tool call, argument, "
    "plan or claim). Correct tool outputs or observations are not root causes, even if the "
    "agent later mishandles them."
)


# Quiet mistakes seen in real (not injected) failures: each only looks wrong when compared
# with the request, a tool's abilities, or earlier evidence. Asked per step when enabled.
STEP_CHECKS = {
    "request": "In [step {i}] the agent ignores, changes or contradicts something the user or "
    "task explicitly asked for.",
    "capability": "In [step {i}] the agent asks a tool or another agent to do something it "
    "cannot do, or calls a tool in a way it does not support.",
    "unfounded": "In [step {i}] the agent relies on a fact, value or assumption it never "
    "actually obtained or checked.",
    "observation": "In [step {i}] the agent ignores or misreads what a tool output or "
    "observation just showed.",
}


# V7: the quiet root cause is often where the agent COMMITS to a diagnosis or plan and then
# acts on it (seen on LongRCA dev: the first diagnostician-to-executor handoff).
COMMIT_QUESTION = (
    "In [step {i}] the agent commits to a diagnosis or plan (what is wrong, or what to do) "
    "that it or another agent then acts on in later steps."
)


# Steps the agent did not write (tool outputs, system prompts) can't be its own mistake.
NON_AGENT_TYPES = frozenset({"system", "tool", "function", "observation", "environment"})


def is_agent_step(step: Step) -> bool:
    return step.type.lower() not in NON_AGENT_TYPES


def step_questions(chunk: Chunk, checks: bool = False, commit: bool = False) -> list[Question]:
    qs = [CHAPTER_QUESTION]
    for i, step in zip(range(chunk.start, chunk.end), chunk.steps, strict=True):
        qs.append(Question(f"s{i}", STEP_QUESTION.format(i=i), "noul"))
        if checks and is_agent_step(step):
            qs += [Question(f"c_{k}_{i}", v.format(i=i), "noul") for k, v in STEP_CHECKS.items()]
        if commit and is_agent_step(step):
            qs.append(Question(f"k_{i}", COMMIT_QUESTION.format(i=i), "noul"))
    return qs


def problem_card(failed: ChunkScore, steps: Sequence[Step]) -> str:
    level = HEALTH_LEVELS[round(failed.score * (len(HEALTH_LEVELS) - 1))]
    lines = [
        f"Known later failure: by the end of steps {failed.start}..{failed.end - 1} the agent "
        f"was judged: {level} (health {failed.score:.2f})."
    ]
    raised = [
        f"- {v} {failed.answers[k]:.0%}"
        for k, v in WARNING_SIGNS.items()
        if failed.answers.get(k, 0.0) >= RAISED_SIGN
    ]
    if raised:
        lines.append("Warning signs raised:\n" + "\n".join(raised))
    suspect = failed.answers.get(SUSPECT_KEY) or {}
    if str(suspect.get("choice", "")).isdigit():
        i = min(max(int(suspect["choice"]), failed.start), failed.end - 1)
        step = clip_tokens(render_step(i, steps[i]), CARD_STEP_TOKENS)
        lines.append(f"The most suspicious step there:\n{step}")
    return "\n".join(lines)


def outcome_card(steps: Sequence[Step]) -> str:
    tail = "\n".join(render_step(i, steps[i]) for i in range(max(0, len(steps) - 2), len(steps)))
    return f"{OUTCOME_FAILURE}\nThe final steps of the run:\n" + clip_tokens(tail, CARD_STEP_TOKENS)


def gap_card(steps: Sequence[Step], failed: ChunkScore | None) -> str:
    """Show the gap real mistakes live in: what was asked vs how the run ended, plus the
    forward pass's symptom when there is one."""
    ask = task_start(steps)
    end = "\n".join(render_step(i, steps[i]) for i in range(max(1, len(steps) - 2), len(steps)))
    parts = [
        f"What the user asked (start of the run):\n{ask}",
        f"How the run ended:\n{clip_tokens(end, CARD_STEP_TOKENS)}",
        problem_card(failed, steps) if failed is not None else OUTCOME_FAILURE,
    ]
    return "\n\n".join(parts)


def build_typed_prompt(task: str, card: str, forward: ChunkScore, chunk: Chunk) -> str:
    parts = []
    if chunk.start > 0 and task:
        parts.append(f"Task (start of the run):\n{task}")
    parts.append(card)
    if forward.state_in:
        parts.append(forward.state_in)
    parts.append(f"Current window:\n{chunk.text}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class ChunkEvidence:
    """What the look-back learned about one chunk: P(cause in chunk), P(step is cause)."""

    chunk_index: int
    p_chunk: float
    p_steps: dict[int, float]
    p_checks: dict[int, dict[str, float]] = field(default_factory=dict)
    p_commit: dict[int, float] = field(default_factory=dict)


def _rise(health: Sequence[float], i: int) -> float:
    return max(0.0, health[i] - (health[i - 1] if i > 0 else 0.0))


# Fixed, hand-written ways to turn evidence into one score per step (no learning).
# Which one to use is a discrete choice made on TRAIN only.
COMBINE_RULES = {
    # P(cause in chunk) x P(step is cause): the original rule.
    "chapter_x_step": lambda e, health, suspect, i: e.p_chunk * e.p_steps[i],
    # The step question alone.
    "step_only": lambda e, health, suspect, i: e.p_steps[i],
    # Favour chunks where the forward pass saw health jump (weight 0.5 .. 1.5).
    "rise_prior": lambda e, health, suspect, i: (
        e.p_chunk * e.p_steps[i] * (0.5 + _rise(health, e.chunk_index))
    ),
    # Average the step answer with its strongest quiet-mistake check (STEP_CHECKS).
    "step_plus_check": lambda e, health, suspect, i: (
        e.p_chunk * (e.p_steps[i] + max(e.p_checks.get(i, {}).values(), default=0.0)) / 2
    ),
    # V7: V2's score, weighted by P(the agent commits here to a diagnosis/plan it acts on).
    "commit_x_step_plus_check": lambda e, health, suspect, i: (
        e.p_chunk
        * (e.p_steps[i] + max(e.p_checks.get(i, {}).values(), default=0.0))
        / 2
        * e.p_commit.get(i, 0.0)
    ),
    # Average the look-back's step answer with the forward pass's suspect-step Choice.
    "forward_suspect": lambda e, health, suspect, i: (
        e.p_chunk * (e.p_steps[i] + suspect[e.chunk_index].get(i, 0.0)) / 2
    ),
}


def combine_scores(
    evidence: Sequence[ChunkEvidence],
    health: Sequence[float],
    suspect: Sequence[dict[int, float]],
    rule: str = "chapter_x_step",
) -> dict[int, float]:
    """One score per step; a step seen in overlapping chunks keeps its stronger reading."""
    score = COMBINE_RULES[rule]
    out: dict[int, float] = {}
    for e in evidence:
        for i in e.p_steps:
            out[i] = max(out.get(i, 0.0), score(e, health, suspect, i))
    return out


NEAR_BEST = 0.8


def pick_step(scores: dict[int, float], how: str = "best") -> int:
    """``best``: highest score (earliest on ties). ``earliest_near_best``: the earliest step
    scoring at least NEAR_BEST x the top score (root causes precede their symptoms)."""
    if how == "earliest_near_best":
        top = max(scores.values())
        return min(i for i, v in scores.items() if v >= NEAR_BEST * top)
    if how != "best":
        raise ValueError(f"unknown pick {how!r}")
    return min(scores, key=lambda i: (-scores[i], i))


def suspect_probs(answers: dict[str, Any]) -> dict[int, float]:
    probs = (answers.get(SUSPECT_KEY) or {}).get("probabilities") or {}
    return {int(k): float(v) for k, v in probs.items() if str(k).isdigit()}


def backward_pass_typed(
    judge: Judge,
    chunks: Sequence[Chunk],
    steps: Sequence[Step],
    forward: Sequence[ChunkScore],
    alarms: Sequence[int] = (),
    known_failed: bool = False,
    min_failure_score: float = DEFAULT_MIN_FAILURE_SCORE,
    workers: int = 4,
    rule: str = "chapter_x_step",
    card: str = "symptom",
    checks: bool = False,
    pick: str = "best",
    commit: bool = False,
    reach_after: int = 0,
) -> Localization:
    """Re-examine the failure chunk and every chunk before it. V8: ``reach_after`` also
    examines that many chunks after it, since the forward alarm can fire before the cause
    (on LongRCA dev coding, 12/50 causes came after the alarm chapter). Each step is
    scored by a fixed ``COMBINE_RULES`` rule (default
    P(cause in chunk) x P(step is the cause); the highest wins, and the top steps are
    returned with their scores. All evidence is kept so rules can be compared offline."""
    if card not in ("symptom", "gap"):
        raise ValueError(f"card must be symptom|gap, got {card!r}")
    n = failure_chunk_index(forward, alarms, min_failure_score)
    anchor = "forward"
    if n is None and known_failed:
        n, anchor = len(chunks) - 1, "outcome"
    elif n is None:
        n = max(forward, key=lambda s: (s.score, -s.chunk_index)).chunk_index
    failed = forward[n] if anchor == "forward" else None
    if card == "gap":
        card_text, task = gap_card(steps, failed), ""  # the card already holds the request
    else:
        card_text = problem_card(failed, steps) if failed is not None else outcome_card(steps)
        task = task_start(steps)

    def examine(chunk: Chunk) -> dict[str, Any]:
        prompt = build_typed_prompt(task, card_text, forward[chunk.index], chunk)
        return judge.evaluate(prompt, step_questions(chunk, checks, commit))

    candidates = list(reversed(chunks[: n + 1 + reach_after]))  # latest first
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        all_answers = list(pool.map(examine, candidates))

    precursors, evidence = [], []
    for chunk, answers in zip(candidates, all_answers, strict=True):
        p_chunk = min(1.0, max(0.0, float(answers[CHAPTER_KEY])))
        p_steps = {i: float(answers[f"s{i}"]) for i in range(chunk.start, chunk.end)}
        p_checks = {
            i: {k: float(answers[f"c_{k}_{i}"]) for k in STEP_CHECKS}
            for i in p_steps
            if f"c_{next(iter(STEP_CHECKS))}_{i}" in answers
        }
        p_commit = {i: float(answers[f"k_{i}"]) for i in p_steps if f"k_{i}" in answers}
        best_in_chunk = max(p_steps, key=lambda i: (p_steps[i], -i))
        precursors.append(PrecursorScore(chunk.index, p_chunk, best_in_chunk, True, answers))
        evidence.append(ChunkEvidence(chunk.index, p_chunk, p_steps, p_checks, p_commit))
    precursors.reverse()
    evidence.reverse()

    health = [f.score for f in forward]
    suspect = [suspect_probs(f.answers) for f in forward]
    combined = combine_scores(evidence, health, suspect, rule)
    ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
    return Localization(
        failure_chunk=n,
        critical_step=pick_step(combined, pick),
        precursors=precursors,
        anchor=anchor,
        ranked=tuple((i, round(p, 6)) for i, p in ranked),
        evidence=tuple(evidence),
    )


# --- Finalist round: compare the shortlisted steps side by side (coarse-to-fine). ---

FINALIST_KEY = "root_cause_finalist"
DEFAULT_FINALISTS = 8

FINALIST_QUESTION = (
    "The run failed. Which candidate step is the root cause: the EARLIEST point where the "
    "AGENT itself went wrong (a bad decision, tool call, argument, plan or claim) that led to "
    "the failure? Later steps that only suffer from or react to an earlier mistake are not "
    "the root cause; correct tool outputs are not root causes."
)


def fit_candidates(texts: dict[int, str], budget_tokens: int) -> dict[int, str]:
    """Share ``budget_tokens`` fairly: small texts stay whole, and what they leave unused
    goes to larger ones, which are clipped (head + tail) only if they still don't fit."""
    if sum(approx_tokens(t) for t in texts.values()) <= budget_tokens:
        return dict(texts)
    out, remaining = {}, budget_tokens
    by_size = sorted(texts, key=lambda k: approx_tokens(texts[k]))
    for n_left, k in zip(range(len(by_size), 0, -1), by_size, strict=True):
        share = max(1, remaining // n_left)
        out[k] = clip_step(texts[k], share)
        remaining -= approx_tokens(out[k])
    return {k: out[k] for k in texts}


def finalist_round(
    judge: Judge,
    steps: Sequence[Step],
    candidates: Sequence[int],
    budget_tokens: int,
    context: int = 0,
) -> dict[int, float]:
    """One Choice over the shortlisted steps, shown side by side in time order with the
    user's request and how the run ended. With ``context`` > 0 each candidate is shown with
    that many neighbouring steps on each side (e.g. the tool output that makes it wrong).
    Returns a probability per candidate step."""
    order = sorted(set(candidates))

    def block(i: int) -> str:
        if context <= 0:
            return render_step(i, steps[i])
        lo, hi = max(0, i - context), min(len(steps), i + context + 1)
        return "\n".join(
            (">>> CANDIDATE " if j == i else "") + render_step(j, steps[j]) for j in range(lo, hi)
        )

    fitted = fit_candidates({i: block(i) for i in order}, budget_tokens)
    end = "\n".join(render_step(i, steps[i]) for i in range(max(1, len(steps) - 2), len(steps)))
    prompt = "\n\n".join(
        [
            f"What the user asked (start of the run):\n{task_start(steps)}",
            f"How the run ended:\n{clip_tokens(end, CARD_STEP_TOKENS)}",
            "Candidate steps (in time order):\n\n" + "\n\n".join(fitted[i] for i in order),
        ]
    )
    question = Question(
        FINALIST_KEY, FINALIST_QUESTION, "choice", {str(i): f"[step {i}]" for i in order}
    )
    probs = judge.evaluate(prompt, [question])[FINALIST_KEY]["probabilities"]
    return {int(k): float(v) for k, v in probs.items()}
