import pytest
from fakes import FailingAfter, NearBestJudge, TypedJudge, UncountedJudge, steps

from alibi.config import load_settings
from alibi.localize import (
    AnalysisFailed,
    Diagnosis,
    JudgeUnavailable,
    SettingsError,
    _preview,
    build_judge,
    diagnose,
)
from alibi.types import Step


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


def test_lead_suspect_keeps_its_real_score_when_it_is_outside_the_top_three():
    """``earliest_near_best`` picks over every step, but ``Localization.ranked`` keeps only the
    top 3, so the lead suspect's score must come from the full score map, not default to 0."""
    d = diagnose(steps(), judge=NearBestJudge(), settings=settings(min_trace_tokens=0))
    assert [s.step for s in d.suspects] == [1, 6, 7]
    assert [s.score for s in d.suspects] == pytest.approx([0.425, 0.5, 0.49])


def test_the_judge_factory_is_used_when_the_trace_is_analysed():
    """Production passes a factory, not a judge; its result must reach the pipeline."""
    built = []

    def factory():
        built.append(1)
        return TypedJudge()

    d = diagnose(steps(), settings=settings(min_trace_tokens=0), judge_factory=factory)
    assert built == [1]
    assert not d.gated and d.suspects[0].step == 7


def test_a_gated_trace_never_calls_the_factory():
    def factory():
        raise AssertionError("a gated trace must not build a judge")

    d = diagnose(steps(), settings=settings(min_trace_tokens=10_000), judge_factory=factory)
    assert d.gated


def test_a_judge_and_a_factory_together_is_an_error():
    with pytest.raises(ValueError, match="not both"):
        diagnose(steps(), judge=TypedJudge(), settings=settings(), judge_factory=TypedJudge)


def test_the_shipped_positional_call_still_works():
    """0.1.3's ``diagnose(steps, judge, settings)`` is the documented product surface; an
    importer that calls it positionally must not break on a patch release."""
    d = diagnose(steps(), TypedJudge(), settings(min_trace_tokens=0))
    assert not d.gated and d.suspects[0].step == 7


def test_settings_default_to_the_environment(monkeypatch):
    """Drive the gate the *other* way: this trace is ~1K tokens, so it is gated under the
    50K default whatever happens. Only a trace that is read and then refused for being over
    a ceiling of 1 proves the environment was consulted at all."""
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")
    monkeypatch.setenv("ALIBI_MAX_TRACE_TOKENS", "1")
    d = diagnose(steps())
    assert d.gated and "ceiling" in d.message


def test_a_failure_mid_analysis_carries_what_it_cost():
    """The surfaces promise exit 2 = nothing spent, so the count has to be real."""
    with pytest.raises(AnalysisFailed) as caught:
        diagnose(steps(), FailingAfter(2), settings(min_trace_tokens=0))
    assert caught.value.completed == 2
    assert "2 Jev call(s) completed" in caught.value.surface_message()
    assert "connection reset" in caught.value.surface_message()


def test_the_counts_belong_to_this_trace_not_the_judge_s_lifetime():
    """Nothing says a host's factory must return a fresh judge, and reusing one client is
    the natural thing to do - but `cost_usd` is a money claim about *this* trace."""
    judge = FailingAfter(1000)  # far beyond one trace: it answers everything for now
    first = diagnose(steps(), judge, settings(min_trace_tokens=0))
    second = diagnose(steps(), judge, settings(min_trace_tokens=0))
    assert second.judge_calls == first.judge_calls
    assert second.cost_usd == pytest.approx(first.cost_usd)

    judge.n = len(judge.calls) + 2  # two more answers on the third trace, then die
    with pytest.raises(AnalysisFailed) as caught:
        diagnose(steps(), judge, settings(min_trace_tokens=0))
    assert caught.value.completed == 2


def test_a_failure_before_any_call_claims_neither_spend_nor_send():
    """The judge can also fail client-side, so 0 completed asserts nothing either way."""

    class DeadOnArrival(TypedJudge):
        def evaluate(self, state, questions):
            raise ValueError("at least one question is required")

    with pytest.raises(AnalysisFailed) as caught:
        diagnose(steps(), DeadOnArrival(), settings(min_trace_tokens=0))
    assert caught.value.completed == 0
    message = caught.value.surface_message()
    assert "nothing is billed" in message and "may already have been sent" in message


def test_a_bad_window_size_never_reaches_the_judge():
    """It is a settings error, free and fixable, so it must not become a spend report."""

    def factory():
        raise AssertionError("the judge is built after the trace is chunked, not before")

    with pytest.raises(SettingsError, match="ALIBI_JUDGE_WINDOW_TOKENS"):
        diagnose(
            steps(),
            settings=settings(min_trace_tokens=0, judge_window_tokens=0),
            judge_factory=factory,
        )


def test_an_injected_judge_cannot_sidestep_the_spend_guard(monkeypatch):
    """``make`` replaces construction, not policy: Jev is paid whoever builds the client."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    with pytest.raises(JudgeUnavailable, match="ALIBI_ALLOW_PAID_MODELS"):
        build_judge(load_settings(), TypedJudge)


def test_an_injected_judge_is_built_once_the_spend_is_allowed(monkeypatch):
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    assert isinstance(build_judge(load_settings(), TypedJudge), TypedJudge)


def test_a_judge_that_logs_no_calls_reports_unknown_not_free():
    """``Judge`` promises only ``evaluate``. A host client that does its own accounting has
    still spent money, so the cost is null - reporting 0 would be a false claim."""
    judge = UncountedJudge()
    d = diagnose(steps(), judge, settings(min_trace_tokens=0))
    assert not d.gated and d.suspects
    assert judge.answered() > 0  # it really did call out
    assert d.judge_calls is None and d.cost_usd is None and d.judge_seconds is None


def test_a_failure_on_an_uncounted_judge_does_not_claim_nothing_was_billed():
    class Uncountable(UncountedJudge):
        def evaluate(self, state, questions):
            super().evaluate(state, questions)
            raise RuntimeError("Jev: connection reset")

    with pytest.raises(AnalysisFailed) as caught:
        diagnose(steps(), Uncountable(), settings(min_trace_tokens=0))
    assert caught.value.completed is None
    message = caught.value.surface_message()
    assert "unknown" in message and "nothing is billed" not in message


def test_a_window_too_small_to_group_steps_is_refused_before_the_judge():
    """A user who writes 10 meaning 10,000 would otherwise get one chapter per step: the
    method reads nothing in context and bills a call per step."""

    def factory():
        raise AssertionError("a degenerate window must not reach the judge")

    with pytest.raises(SettingsError, match="ALIBI_JUDGE_WINDOW_TOKENS"):
        diagnose(
            steps(),
            settings=settings(min_trace_tokens=0, judge_window_tokens=10),
            judge_factory=factory,
        )


def test_a_workable_window_is_not_refused():
    """The guard must not fire on the shape the method was measured with."""
    d = diagnose(steps(), TypedJudge(), settings(min_trace_tokens=0))
    assert not d.gated and d.n_chapters < len(steps())


def test_a_trace_of_oversized_steps_is_still_analysed():
    """The same shape - one step per chapter - is legitimate when the steps are the big
    thing, not the window. Refusing those would block a real trace shape (one huge tool
    output per step), so the guard requires a window far below the measured 10,000."""
    big = steps(n=6, chars=60_000)  # each step alone exceeds a 10K chapter
    d = diagnose(big, TypedJudge(), settings(min_trace_tokens=0, judge_window_tokens=10_000))
    assert not d.gated and d.n_chapters == len(big)
