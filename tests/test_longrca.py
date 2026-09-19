import json

from alibi.backward import is_agent_step
from alibi.sources.longrca import LongRCATraceSource, load_annotations


def write_rows(tmp_path):
    rows = [
        {
            "question_ID": "swe_bench_pro__001",
            "history": [
                {"step": 0, "content": "fix the bug", "role": "user", "name": "human"},
                {"step": 1, "content": "ls", "role": "assistant", "name": "ActionAgent"},
                {"step": 2, "content": "a.py", "role": "user", "name": "Computer_terminal"},
                {"step": 3, "content": "edit a.py", "role": "assistant", "name": "ActionAgent"},
            ],
            "mistake_agent": "ActionAgent",
            "mistake_step": 3,
            "mistake_reason": "edited the wrong file",
        },
        {
            "question_ID": "vitabench__007",
            "history": [
                {"step": 0, "content": "book", "role": "assistant", "name": "Action_Expert"},
                {"step": 1, "content": "ok", "role": "user", "name": "Computer_terminal"},
            ],
            "mistake_agent": "Action_Expert",
            "mistake_step": 0,
            "mistake_reason": "wrong date",
        },
    ]
    path = tmp_path / "longrca-full.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_annotations_carry_subset_label_and_reason(tmp_path):
    anns = load_annotations(write_rows(tmp_path))
    assert [a.trace_id for a in anns] == ["swe_bench_pro/001", "vitabench/007"]
    assert [a.split for a in anns] == ["swe_bench_pro", "vitabench"]
    assert anns[0].critical_step == 3
    assert anns[0].category == "ActionAgent"
    assert anns[0].failure_summary == "edited the wrong file"


def test_subset_filter(tmp_path):
    anns = load_annotations(write_rows(tmp_path), subsets=("vitabench",))
    assert [a.trace_id for a in anns] == ["vitabench/007"]


def test_terminal_output_is_a_tool_step_not_the_agents_own(tmp_path):
    steps = LongRCATraceSource(write_rows(tmp_path)).get_trace("swe_bench_pro/001")
    assert [s.type for s in steps] == ["user", "assistant", "tool", "assistant"]
    assert steps[3].name == "ActionAgent" and steps[3].inputs == {"content": "edit a.py"}
    assert not is_agent_step(steps[2]) and is_agent_step(steps[3])


def test_cli_loads_longrca_benchmark(tmp_path):
    import argparse

    from alibi.cli import _load_benchmark

    args = argparse.Namespace(
        benchmark="longrca", data_dir=str(write_rows(tmp_path)), subset="swe_bench_pro"
    )
    anns, source = _load_benchmark(args)
    assert [a.trace_id for a in anns] == ["swe_bench_pro/001"]
    assert len(source.get_trace("swe_bench_pro/001")) == 4
