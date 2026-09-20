"""Fakes shared by the tests that drive the whole pipeline offline.

One definition of the judge and of the trace shape, so a change to the Question contract
breaks every surface that depends on it instead of only the copy that was remembered.
"""

from __future__ import annotations

from alibi.judges.base import CallRecord
from alibi.types import Step


class TypedJudge:
    """Health jumps at the last chapter; step 7 is the likeliest cause.

    Records a call each time, so tests can check that cost and latency reach the Diagnosis.
    """

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def evaluate(self, state, questions):
        self.calls.append(CallRecord("fake", 0.5, None, None, 0.002))
        out = {}
        for q in questions:
            if q.answer_type == "score":
                out[q.id] = 1.0 if "[step 8]" in state else 0.0
            elif q.answer_type == "choice":
                first = next(iter(q.criteria))
                out[q.id] = {"choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
            elif q.id == "s7":
                out[q.id] = 0.9
            else:
                out[q.id] = 0.1
        return out


class FailingAfter(TypedJudge):
    """Answers ``n`` calls and then fails, as a dropped connection mid-trace would.

    The answered calls are in ``calls``, so a surface can be checked on whether it reports
    what was actually billed.
    """

    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.n = n

    def evaluate(self, state, questions):
        if len(self.calls) >= self.n:
            raise RuntimeError("Jev: connection reset")
        return super().evaluate(state, questions)


class UncountedJudge:
    """A host's own client: it answers, but keeps no ``calls`` log at all.

    ``Judge`` promises only ``evaluate``, so this is a legal judge. What it cost cannot be
    read off it - which is not the same as it having cost nothing.
    """

    def __init__(self) -> None:
        self._answers = TypedJudge()

    def evaluate(self, state, questions):
        return self._answers.evaluate(state, questions)

    def answered(self) -> int:
        """How many calls it really took, for a test to compare against."""
        return len(self._answers.calls)


class NearBestJudge:
    """Health jumps at chapter 3, and four steps draw enough suspicion to be ranked.

    The look-back's combined scores come out as {1: 0.425, 5: 0.475, 6: 0.5, 7: 0.49}, so the
    top-3 ``ranked`` tuple is (6, 7, 5) while ``earliest_near_best`` picks step 1 (0.425 is
    within 80% of the top 0.5). The lead suspect is therefore absent from ``ranked``.
    """

    STEP_PROBS = {1: 0.85, 5: 0.95, 6: 1.0, 7: 0.98}

    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
        out = {}
        late = any(f"[step {i}]" in state for i in (7, 8))
        for q in questions:
            if q.answer_type == "score":  # the forward pass's health question
                out[q.id] = 1.0 if late else 0.0
            elif q.answer_type == "choice":
                first = next(iter(q.criteria))
                out[q.id] = {"choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
            elif q.id == "cause_in_window":
                out[q.id] = 1.0
            elif q.id.startswith("s") and q.id[1:].isdigit():
                out[q.id] = self.STEP_PROBS.get(int(q.id[1:]), 0.05)
            else:  # the warning signs and the per-step checks
                out[q.id] = 0.0
        return out


def steps(n: int = 9, chars: int = 400) -> list[Step]:
    """A trace of ``n`` steps: a user request, then alternating assistant and tool steps."""
    kinds = ["user", *["assistant", "tool"] * ((n + 1) // 2)]
    return [Step(str(i), kinds[i], None, {"content": "x" * chars}) for i in range(n)]


def step_dicts(n: int = 9, chars: int = 400) -> list[dict]:
    """The same trace as ``steps``, in the JSON form a trace file holds."""
    return [{"type": s.type, "inputs": s.inputs} for s in steps(n, chars)]
