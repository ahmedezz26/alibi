from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Step:
    """One step (tool call, LLM call, ...) of an agent trace."""

    step_id: str
    type: str
    timestamp: datetime | None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    error: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A token-budgeted, possibly overlapping window of consecutive steps."""

    index: int
    steps: tuple[Step, ...]
    start: int  # index of the first step in the full trace
    end: int  # exclusive index of the last step in the full trace
    text: str
    tokens: int


@dataclass(frozen=True)
class Annotation:
    """Ground truth for one failed trace: the step that caused the failure."""

    trace_id: str
    split: str
    critical_step: int  # 0-based position in the trace's step list
    category: str
    failure_summary: str = ""


# boolean/integer/number/string: free-form LLM judges. noul/score/choice: System One judges
# (Jev), which answer with probabilities and cannot write free text.
AnswerType = Literal["boolean", "integer", "number", "string", "noul", "score", "choice"]


@dataclass(frozen=True)
class Question:
    """A typed question the judge answers about a chunk."""

    id: str
    prompt: str
    answer_type: AnswerType = "string"
    # noul: optional {"true": ..., "false": ...}; score: ordered levels; choice: {option: meaning}
    criteria: Any = None
