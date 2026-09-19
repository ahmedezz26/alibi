import pytest

from alibi.config import load_settings
from alibi.localize import Diagnosis, diagnose
from alibi.types import Step


class TypedJudge:
    """Health jumps at the last chapter; step 7 is the likeliest cause."""

    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
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


def steps(n=9, chars=400):
    kinds = ["user"] + ["assistant", "tool"] * n
    return [Step(str(i), kinds[i], None, {"content": "x" * chars}) for i in range(n)]


def settings(**over):
    s = load_settings()
    return s.__class__(**{**s.__dict__, "judge_window_tokens": 300, **over})


def test_short_trace_is_gated_without_calling_the_judge():
    d = diagnose(steps(), judge=None, settings=settings(min_trace_tokens=10_000))
    assert isinstance(d, Diagnosis)
    assert d.gated and d.suspects == [] and d.judge_calls == 0
    assert "direct read" in d.message


def test_long_trace_returns_top_suspects_with_context():
    d = diagnose(steps(), judge=TypedJudge(), settings=settings(min_trace_tokens=0))
    assert not d.gated
    assert d.n_chapters > 1 and d.alarm_chapter is not None
    assert 1 <= len(d.suspects) <= 3
    assert d.suspects[0].step == 7
    assert d.suspects[0].step_type == "assistant"
    assert d.suspects[0].preview.startswith("xxx")
    assert all(s.chapter >= 0 for s in d.suspects)


def test_long_trace_without_a_judge_is_an_error():
    with pytest.raises(ValueError, match="judge"):
        diagnose(steps(), judge=None, settings=settings(min_trace_tokens=0))
