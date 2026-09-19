"""Forward pass: score chunks in order, optionally carrying a compressed running state.

Build-order step 3 is the stateless mode (``propagate_state=False``); step 4 adds a
bounded running summary written by the judge and fed into the next chunk's call.

``forward_pass_typed`` is the same loop for System One judges (Jev), which cannot write
text: the running state is a memory card of numbers (a health Score plus one Noul per
warning sign) instead of a written summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from alibi.chunking import TokenCounter, approx_tokens, render_step
from alibi.judges import Judge
from alibi.types import Chunk, Question, Step

SCORE_KEY = "anomaly_score"
STATE_KEY = "state_summary"
DEFAULT_MAX_STATE_TOKENS = 150

DEFAULT_QUESTIONS = [
    Question(
        SCORE_KEY,
        "How likely is it that the agent makes a mistake in the CURRENT window (wrong tool, "
        "wrong arguments, ignoring or misreading an observation, violating a task constraint, "
        "unjustified claim)? Judge against the running summary too. "
        "0 = clearly fine, 1 = clearly wrong.",
        "number",
    ),
    Question("rationale", "One sentence justifying the score, citing step numbers.", "string"),
]

STATE_QUESTION = Question(
    STATE_KEY,
    f"Updated running summary for judging LATER windows, under {DEFAULT_MAX_STATE_TOKENS} "
    "tokens. Keep the user's task and every explicit constraint verbatim; key facts observed "
    "so far (with step numbers); the agent's current plan/progress; anything suspicious.",
    "string",
)


@dataclass(frozen=True)
class ChunkScore:
    chunk_index: int
    start: int
    end: int
    score: float
    answers: dict[str, Any]
    state_in: str = ""  # running summary the judge saw (kept to audit lossy compression)


def build_prompt(state: str, chunk: Chunk) -> str:
    if not state:
        return f"Current window:\n{chunk.text}"
    return f"Running summary of earlier windows:\n{state}\n\nCurrent window:\n{chunk.text}"


def truncate_state(state: str, max_tokens: int, count_tokens: TokenCounter = approx_tokens) -> str:
    if count_tokens(state) <= max_tokens:
        return state
    words = state.split()
    while words and count_tokens(" ".join(words) + " …") > max_tokens:
        words.pop()
    return " ".join(words) + " …"


def forward_pass(
    judge: Judge,
    chunks: Sequence[Chunk],
    questions: Sequence[Question] = DEFAULT_QUESTIONS,
    score_key: str = SCORE_KEY,
    propagate_state: bool = True,
    max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS,
) -> list[ChunkScore]:
    asked = list(questions) + ([STATE_QUESTION] if propagate_state else [])
    scores = []
    state = ""
    for chunk in chunks:
        answers = judge.evaluate(build_prompt(state, chunk), asked)
        scores.append(
            ChunkScore(
                chunk_index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                score=min(1.0, max(0.0, float(answers[score_key]))),
                answers=answers,
                state_in=state,
            )
        )
        if propagate_state:
            state = truncate_state(str(answers[STATE_KEY]), max_state_tokens)
    return scores


HEALTH_KEY = "health"
SUSPECT_KEY = "suspect_step"
TASK_START_TOKENS = 1_500

HEALTH_LEVELS = [
    "On track: every step so far is a sensible move toward the user's task.",
    "Minor problem: a small slip or inefficiency that does not threaten the task.",
    "Big problem: a mistake that will likely make the task fail unless the agent fixes it.",
    "Failed: the run can no longer correctly accomplish the user's task.",
]

# Warning signs tracked on the memory card: question id -> statement judged for truth.
WARNING_SIGNS = {
    "unfixed_error": "The agent has hit an error or a wrong result that it has not fixed.",
    "misread_task": "The agent has misread, ignored or changed part of the user's task or "
    "its constraints.",
    "off_plan": "The agent has drifted away from a sensible plan for the user's task.",
    "trusted_wrong": "The agent has accepted a wrong or unverified result as correct.",
}


def clip_tokens(text: str, max_tokens: int) -> str:
    return text if approx_tokens(text) <= max_tokens else text[: max_tokens * 4] + " …"


def task_start(steps: Sequence[Step]) -> str:
    return clip_tokens(render_step(0, steps[0]), TASK_START_TOKENS) if steps else ""


def typed_questions(chunk: Chunk) -> list[Question]:
    return [
        Question(
            HEALTH_KEY,
            "How is the agent doing by the END of the current window, counting everything "
            "that happened before it (see the memory card)?",
            "score",
            HEALTH_LEVELS,
        ),
        *(
            Question(k, f"By the end of the current window: {v}", "noul")
            for k, v in WARNING_SIGNS.items()
        ),
        Question(
            SUSPECT_KEY,
            "Which step in the current window is most likely a mistake by the AGENT itself "
            "(a bad decision, tool call, argument, plan or claim)?",
            "choice",
            {str(i): f"[step {i}]" for i in range(chunk.start, chunk.end)},
        ),
    ]


def memory_card(answers: dict[str, Any]) -> str:
    lines = [f"- how the agent is doing: {answers[HEALTH_KEY]:.2f} (0 = on track, 1 = failed)"]
    lines += [f"- {v} {answers[k]:.0%}" for k, v in WARNING_SIGNS.items()]
    return "Memory card from earlier windows (judged probabilities):\n" + "\n".join(lines)


def build_typed_prompt(task: str, card: str, chunk: Chunk) -> str:
    parts = []
    if chunk.start > 0 and task:
        parts.append(f"Task (start of the run):\n{task}")
    if card:
        parts.append(card)
    parts.append(f"Current window:\n{chunk.text}")
    return "\n\n".join(parts)


def forward_pass_typed(
    judge: Judge,
    chunks: Sequence[Chunk],
    steps: Sequence[Step],
    propagate_state: bool = True,
) -> list[ChunkScore]:
    task = task_start(steps)
    scores = []
    card = ""
    for chunk in chunks:
        answers = judge.evaluate(build_typed_prompt(task, card, chunk), typed_questions(chunk))
        scores.append(
            ChunkScore(
                chunk_index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                score=min(1.0, max(0.0, float(answers[HEALTH_KEY]))),
                answers=answers,
                state_in=card,
            )
        )
        if propagate_state:
            card = memory_card(answers)
    return scores
