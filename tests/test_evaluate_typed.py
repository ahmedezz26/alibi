import pytest

from alibi.evaluate import EvalConfig, evaluate_one, evaluate_windowed
from alibi.types import Annotation, Step


class AnyTypedJudge:
    """Health jumps at the last chunk; step 4 is the most likely root cause."""

    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
        out = {}
        for q in questions:
            if q.answer_type == "score":
                out[q.id] = 1.0 if "[step 8]" in state else 0.0
            elif q.answer_type == "choice":
                out[q.id] = {"choice": next(iter(q.criteria)), "probabilities": {}, "confidence": 1}
            elif q.id.startswith("s") and q.id[1:].isdigit():
                out[q.id] = 0.9 if q.id == "s4" else 0.1
            else:
                out[q.id] = 0.5
        return out


def steps():
    return [
        Step(step_id=str(i), type="tool", timestamp=None, inputs={"x": "y" * 40}) for i in range(9)
    ]


def test_typed_pipeline_row_has_prediction_and_ranking():
    ann = Annotation("t/1", "test", critical_step=4, category="c")
    cfg = EvalConfig(max_tokens=40, overlap_tokens=5, mode="windowed", pipeline="typed")

    row = evaluate_windowed(ann, steps(), AnyTypedJudge(), cfg)

    assert row["pred"] == 4 and row["ranked"][0][0] == 4
    assert row["n_chunks"] > 1 and row["pipeline"] == "typed"
    assert "health" in row["forward_answers"][0]


def test_typed_pipeline_refuses_whole_trace_modes():
    ann = Annotation("t/1", "test", critical_step=4, category="c")

    class Source:
        def get_trace(self, _):
            return steps()

    cfg = EvalConfig(max_tokens=40, overlap_tokens=5, mode="whole", pipeline="typed")
    with pytest.raises(ValueError, match="windowed"):
        evaluate_one(ann, Source(), AnyTypedJudge(), cfg)


def test_row_stores_evidence_so_rules_can_be_rescored_offline():
    from alibi.evaluate import rescore

    ann = Annotation("t/1", "test", critical_step=4, category="c")
    cfg = EvalConfig(max_tokens=40, overlap_tokens=5, mode="windowed", pipeline="typed")
    row = evaluate_windowed(ann, steps(), AnyTypedJudge(), cfg)

    assert row["combine_rule"] == "chapter_x_step"
    assert row["backward_evidence"] and "p_steps" in row["backward_evidence"][0]
    assert rescore(row, "chapter_x_step") == row["pred"]
    assert rescore(row, "step_only") == 4


def test_config_switches_reach_the_pipeline_and_rescore_supports_pick():
    from alibi.evaluate import rescore

    ann = Annotation("t/1", "test", critical_step=4, category="c")
    cfg = EvalConfig(
        max_tokens=40, overlap_tokens=5, mode="windowed", pipeline="typed", card="gap", checks=True
    )
    agent_steps = [
        Step(step_id=str(i), type="assistant", timestamp=None, inputs={"x": "y" * 40})
        for i in range(9)
    ]
    row = evaluate_windowed(ann, agent_steps, AnyTypedJudge(), cfg)
    assert row["backward_evidence"][0]["p_checks"]
    assert rescore(row, "step_plus_check") in range(9)
    assert rescore(row, "chapter_x_step", pick="earliest_near_best") <= rescore(
        row, "chapter_x_step"
    )


def test_row_records_the_routers_trace_kind():
    ann = Annotation("t/1", "test", critical_step=4, category="c")
    cfg = EvalConfig(max_tokens=40, overlap_tokens=5, mode="windowed", pipeline="typed")
    row = evaluate_windowed(ann, steps(), AnyTypedJudge(), cfg)
    assert row["trace_kind"] == "other"  # tool steps only: no agent talk or code actions
