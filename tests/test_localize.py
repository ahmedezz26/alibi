import pytest

from alibi.config import load_settings
from alibi.judges.base import CallRecord
from alibi.localize import Diagnosis, _preview, diagnose
from alibi.types import Step


class TypedJudge:
    """Health jumps at the last chapter; step 7 is the likeliest cause."""

    def __init__(self):
        self.calls = []

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
    # cost and time are summed over the judge's own call log
    assert d.judge_calls > 0
    assert d.judge_seconds == pytest.approx(0.5 * d.judge_calls)
    assert d.cost_usd == pytest.approx(0.002 * d.judge_calls)


def test_a_trace_above_the_cost_ceiling_is_refused_with_an_estimate():
    """Jev is billed per input token, so a very long trace is refused before it is spent."""
    d = diagnose(steps(), judge=None, settings=settings(min_trace_tokens=0, max_trace_tokens=100))
    assert d.gated and d.suspects == [] and d.judge_calls == 0
    assert "ALIBI_MAX_TRACE_TOKENS" in d.message
    assert "$" in d.message


def test_a_trace_under_the_ceiling_is_analysed():
    d = diagnose(
        steps(), judge=TypedJudge(), settings=settings(min_trace_tokens=0, max_trace_tokens=10_000)
    )
    assert not d.gated and d.suspects


def test_a_trace_with_no_steps_is_reported_not_crashed():
    d = diagnose([], judge=None, settings=settings(min_trace_tokens=0))
    assert d.gated and d.n_steps == 0 and d.suspects == []


def test_preview_falls_back_to_the_rendered_step():
    """A step without a ``content`` field is previewed by its rendered form."""
    assert _preview(0, Step("0", "tool", None, {"result": "y" * 50})).startswith("[step 0]")


def test_long_trace_without_a_judge_is_an_error():
    with pytest.raises(ValueError, match="judge"):
        diagnose(steps(), judge=None, settings=settings(min_trace_tokens=0))


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


def test_lead_suspect_keeps_its_real_score_when_it_is_outside_the_top_three():
    """``earliest_near_best`` picks over every step, but ``Localization.ranked`` keeps only the
    top 3, so the lead suspect's score must come from the full score map, not default to 0."""
    d = diagnose(steps(), judge=NearBestJudge(), settings=settings(min_trace_tokens=0))
    assert [s.step for s in d.suspects] == [1, 6, 7]
    assert [s.score for s in d.suspects] == pytest.approx([0.425, 0.5, 0.49])
