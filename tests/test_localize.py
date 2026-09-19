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


class RegressionJudge:
    """Produces a scenario where critical_step is not in the top 3 ranked.

    Early chunks have high health, but suspect probabilities favor later steps.
    This makes the top-3 ranked be later steps, while earliest_near_best still picks
    an earlier step that has a non-zero combined score.
    """

    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
        out = {}
        # Identify which steps are in the current state
        has_early = any(f"[step {i}]" in state for i in range(4))

        for q in questions:
            if q.answer_type == "score":
                # Early chunks score high; this will make early steps rank in top 3
                out[q.id] = 0.95 if has_early else 0.5
            elif q.answer_type == "choice":
                first = next(iter(q.criteria))
                # For suspect step probabilities: give high probability to later steps
                # to push them into top 3. This is the "probabilities" dict.
                # Keys should be step indices.
                out[q.id] = {
                    "choice": first,
                    "probabilities": {7: 0.9, 8: 0.85, 2: 0.8},
                    "confidence": 0.9,
                }
            else:
                out[q.id] = 0.1
        return out


def test_critical_step_not_in_ranked_has_correct_score():
    """Regression test: critical_step outside top-3 ranked still gets correct score.

    This test verifies the fix for a bug where diagnose() would report a suspect
    with score 0.0 if its step was picked as critical_step but was not in the
    top-3 ranked tuple (which is truncated by backward_pass_typed).
    """
    d = diagnose(steps(), judge=RegressionJudge(), settings=settings(min_trace_tokens=0))
    assert not d.gated
    assert len(d.suspects) > 0
    # The first suspect should have a non-zero score, even if its step is not in the
    # truncated ranked tuple. The score should come from the full combined map.
    assert d.suspects[0].score > 0.0, "critical_step should have correct combined score, not 0.0"
