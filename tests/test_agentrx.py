import json

import pytest

from alibi.evaluate import summarize
from alibi.sources.agentrx import AgentRxTraceSource, load_annotations


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def data_dir(tmp_path):
    failures = [
        {"failure_id": "1", "step_number": 2, "failure_category": "A"},
        {"failure_id": "2", "step_number": 3, "failure_category": "B"},
    ]
    write_jsonl(
        tmp_path / "tau_retail.jsonl",
        [
            {
                "trajectory_id": "7",
                "failure_summary": "s",
                "failures": failures,
                "root_cause_failure_id": "2",
            }
        ],
    )
    steps = [
        {"index": i, "substeps": [{"sub_index": 1, "role": r, "content": f"c{i}"}]}
        for i, r in [(2, "assistant"), (1, "user"), (3, "tool")]
    ]
    write_jsonl(
        tmp_path / "tau_retail_dataset.jsonl", [{"trajectory_id": "tau_retail_7", "steps": steps}]
    )
    write_jsonl(tmp_path / "magentic_one.jsonl", [])
    write_jsonl(tmp_path / "magentic_dataset.jsonl", [])
    return tmp_path


def test_annotations_join_and_use_root_cause_zero_based(data_dir):
    [ann] = load_annotations(data_dir)
    assert ann.trace_id == "tau_retail_7"
    assert ann.critical_step == 2  # step_number 3 -> position 2
    assert ann.category == "B"


def test_trace_steps_ordered_with_roles(data_dir):
    steps = AgentRxTraceSource(data_dir).get_trace("tau_retail_7")
    assert [s.step_id for s in steps] == ["1", "2", "3"]
    assert [s.type for s in steps] == ["user", "assistant", "tool"]
    assert steps[2].inputs == {"content": "c3"}


def test_summarize_precision_recall_and_baseline():
    base = {"n_chunks": 2, "failure_chunk_contains_gold": True, "gold_before_failure_chunk": False}
    rows = [
        {**base, "trace_id": "a", "gold": 5, "pred": 5, "n_steps": 10, "cost": 0.01},
        {**base, "trace_id": "b", "gold": 5, "pred": 7, "n_steps": 10, "cost": 0.01},
        {**base, "trace_id": "c", "gold": 5, "pred": None, "n_steps": 10, "cost": 0.01},
        {"trace_id": "d", "error": "boom"},
    ]
    s = summarize(rows)
    assert (s["trajectories"], s["errors"], s["predictions"]) == (3, 1, 2)
    assert s["precision@±0"] == 0.5 and s["recall@±0"] == 0.333
    assert s["precision@±3"] == 1.0 and s["recall@±3"] == 0.667
    assert s["random_recall@±0"] == 0.1


def test_whole_trace_baseline_single_call_and_clamps(data_dir):
    from alibi.evaluate import EvalConfig, evaluate_one

    class OneShotJudge:
        def __init__(self):
            self.prompts = []

        def evaluate(self, state, questions):
            self.prompts.append(state)
            return {"critical_step": 99, "confidence": 0.4, "rationale": "r"}

    judge = OneShotJudge()
    [ann] = load_annotations(data_dir)
    row = evaluate_one(ann, AgentRxTraceSource(data_dir), judge, EvalConfig(10, 1, mode="whole"))

    assert len(judge.prompts) == 1
    assert "[step 0]" in judge.prompts[0] and "[step 2]" in judge.prompts[0]
    assert (row["pred"], row["pred_valid"], row["gold"]) == (2, False, 2)


def test_resume_retries_errored_rows(data_dir, tmp_path):
    from alibi.evaluate import EvalConfig, run

    out = tmp_path / "runs.jsonl"
    out.write_text(
        json.dumps({"trace_id": "tau_retail_7", "split": "tau_retail", "error": "403"}) + "\n"
    )

    class Judge:
        def evaluate(self, state, questions):
            return {"critical_step": 2, "confidence": 1.0, "rationale": "r"}

    rows = run(
        load_annotations(data_dir),
        Judge,
        EvalConfig(10, 1, mode="whole"),
        out,
        AgentRxTraceSource(data_dir),
        workers=1,
    )
    assert len(rows) == 1 and "error" not in rows[0] and rows[0]["pred"] == 2


class RoutingJudge:
    """Answers both whole-trace and windowed questions; records which were asked."""

    def __init__(self):
        self.asked = []

    def evaluate(self, state, questions):
        ids = [q.id for q in questions]
        self.asked.append(ids)
        answers = {"rationale": "r", "confidence": 0.5, "critical_step": 2}
        answers.update({"anomaly_score": 0.0, "state_summary": "s", "precursor_score": 0.5})
        return {k: v for k, v in answers.items() if k in ids}


def test_hybrid_routes_short_trace_to_single_call(data_dir):
    from alibi.evaluate import EvalConfig, evaluate_one

    judge = RoutingJudge()
    [ann] = load_annotations(data_dir)
    row = evaluate_one(
        ann, AgentRxTraceSource(data_dir), judge, EvalConfig(10, 1, route_threshold=10_000)
    )
    assert row["route"] == "whole" and len(judge.asked) == 1


def test_hybrid_routes_long_trace_to_windowed(data_dir):
    from alibi.evaluate import EvalConfig, evaluate_one

    judge = RoutingJudge()
    [ann] = load_annotations(data_dir)
    row = evaluate_one(
        ann, AgentRxTraceSource(data_dir), judge, EvalConfig(10, 1, route_threshold=1)
    )
    assert row["route"] == "windowed" and len(judge.asked) > 1
    assert row["pred"] in range(3)


def test_settings_window_defaults_to_reliable_length(monkeypatch):
    from alibi.config import load_settings

    monkeypatch.setattr("alibi.config.load_dotenv", lambda: None)
    monkeypatch.delenv("ALIBI_JUDGE_WINDOW_TOKENS", raising=False)
    monkeypatch.setenv("ALIBI_JUDGE_RELIABLE_TOKENS", "8000")
    s = load_settings()
    assert (s.judge_reliable_tokens, s.judge_window_tokens) == (8000, 8000)
