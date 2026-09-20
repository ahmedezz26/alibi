"""The product must run the configuration the research measured.

Every number in `docs/research-log.md` came from `alibi eval`, i.e. `evaluate_windowed`.
Users get `alibi.localize.diagnose`. The two share `forward.py`, `backward.py` and
`drift.py`, but each has its own wrapper deciding *how* to run them - chapter size, card,
checks, combine rule, pick, CUSUM thresholds - and nothing else in the suite compares them.
Without this file, editing `V2_CONFIG` leaves 179 tests green while the product quietly
becomes a variant nobody measured, still claiming the measured accuracy.

The research settings below are written out as literals on purpose. Importing them from the
product would make this file agree with whatever the product does, which is the opposite of
the point: these are the record, from the 2026-09-19 step-6 and TrajErrBench entries.
"""

from fakes import NearBestJudge, steps

from alibi.config import load_settings
from alibi.evaluate import EvalConfig, evaluate_windowed
from alibi.localize import V2_CONFIG, diagnose
from alibi.types import Annotation

RESEARCH_V2 = {
    "pipeline": "typed",  # Jev answers probabilities, not prose
    "mode": "windowed",
    "propagate_state": True,
    "card": "gap",
    "checks": True,
    "combine_rule": "step_plus_check",
    "pick": "earliest_near_best",
    "cusum_k": 0.2,
    "cusum_h": 0.5,
}
RESEARCH_CHAPTER_TOKENS = 10_000
RESEARCH_OVERLAP_DIVISOR = 8  # overlap = chapter // 8
RESEARCH_MIN_TRACE_TOKENS = 50_000  # the method's edge was measured on long traces only


class ComparisonJudge(NearBestJudge):
    """Deterministic, and sensitive to every knob under test.

    Step probabilities put the strongest suspicion late (step 6), so ``pick`` is visible.
    A single quiet-mistake check on step 1 lifts it to 0.475 against step 6's 0.5 - within
    ``earliest_near_best``'s 80% band, but only when checks are asked *and* the rule that
    averages them in is used. So `checks`, `combine_rule` and `pick` each move the answer.
    The `card` does not change any answer, so it is compared as text instead: every prompt
    the judge is given is recorded, and the two wrappers must ask exactly the same things.
    """

    STEP_PROBS = {1: 0.45, 5: 0.95, 6: 1.0, 7: 0.98}
    CHECKED_STEP = 1
    CHECK_PROB = 0.5

    def __init__(self):
        super().__init__()
        self.transcript = []

    def evaluate(self, state, questions):
        self.transcript.append((state, tuple((q.id, q.prompt, q.answer_type) for q in questions)))
        answers = super().evaluate(state, questions)
        for q in questions:
            if q.id.startswith("c_") and q.id.rsplit("_", 1)[-1] == str(self.CHECKED_STEP):
                answers[q.id] = self.CHECK_PROB
        return answers


def settings(window, **over):
    s = load_settings()
    return s.__class__(
        **{**s.__dict__, "judge_window_tokens": window, "min_trace_tokens": 0, **over}
    )


def test_diagnose_localizes_where_the_research_harness_localizes():
    """One trace, one judge, both wrappers: the step a user is told to read first must be
    the step the benchmark would have scored."""
    window = 300  # small enough that this fixture trace becomes several chapters
    trace = steps()

    product_judge, research_judge = ComparisonJudge(), ComparisonJudge()
    product = diagnose(trace, product_judge, settings(window))
    research = evaluate_windowed(
        Annotation("fixture", "test", 0, "synthetic"),
        trace,
        research_judge,
        EvalConfig(
            max_tokens=window, overlap_tokens=window // RESEARCH_OVERLAP_DIVISOR, **RESEARCH_V2
        ),
    )

    assert product.suspects[0].step == research["pred"]
    # and it got there the same way, so a drifting CUSUM or anchor cannot cancel out
    assert product.n_chapters == research["n_chunks"]
    assert product.anchor == research["anchor"]
    assert product.alarm_chapter == (research["alarms"][0] if research["alarms"] else None)
    assert [s.step for s in product.suspects[1:]] == [
        i for i, _ in research["ranked"] if i != research["pred"]
    ][: len(product.suspects) - 1]
    # and it asked the judge the same things, in the same order: same chapters, same card,
    # same questions. This is what catches a change no answer would reveal, such as `card`.
    assert product_judge.transcript == research_judge.transcript


def test_the_product_runs_the_measured_v2_configuration():
    """The knobs themselves, so a change is a failing test and not a silent variant."""
    assert V2_CONFIG == {
        "card": RESEARCH_V2["card"],
        "checks": RESEARCH_V2["checks"],
        "rule": RESEARCH_V2["combine_rule"],
        "pick": RESEARCH_V2["pick"],
        "cusum_k": RESEARCH_V2["cusum_k"],
        "cusum_h": RESEARCH_V2["cusum_h"],
    }


def test_the_shipped_defaults_are_the_measured_chapter_size_and_gate():
    """What the behavioural test above cannot pin, because it sets its own small window."""
    s = load_settings()
    assert s.judge_window_tokens == RESEARCH_CHAPTER_TOKENS
    assert s.judge_window_tokens // RESEARCH_OVERLAP_DIVISOR == 1_250
    assert s.min_trace_tokens == RESEARCH_MIN_TRACE_TOKENS
