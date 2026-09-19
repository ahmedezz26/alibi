import json

from alibi.sources.trajerrbench import TrajErrBenchTraceSource, load_annotations


def write(root, src, name, steps, critical, etype="reason.WrongChoice"):
    d = root / "en" / src
    d.mkdir(parents=True, exist_ok=True)
    messages = [
        {
            "step": s,
            "role": role,
            "name": role,
            "content": f"{role}-{s}",
            "judgable": role == "assistant",
        }
        for s, role in steps
    ]
    meta = {
        "task_description": "t",
        "annotation": {"critical_error_step": critical, "critical_error_type": etype},
    }
    (d / f"{name}.json").write_text(json.dumps({"messages": messages, "metadata": meta}))


def test_labels_match_message_step_ids_not_positions(tmp_path):
    write(
        tmp_path, "tau2bench", "a", [(0, "system"), (1, "user"), (3, "assistant"), (4, "tool")], 3
    )
    write(tmp_path, "swebenchpro", "b", [(0, "system"), (1, "assistant")], 1, None)

    anns = {a.trace_id: a for a in load_annotations(tmp_path)}
    a = anns["tau2bench/a"]
    assert (a.split, a.critical_step, a.category) == ("tau2bench", 2, "reason.WrongChoice")
    assert anns["swebenchpro/b"].category == "unknown"
    assert [x.trace_id for x in load_annotations(tmp_path, subsets=("swebenchpro",))] == [
        "swebenchpro/b"
    ]


def test_trace_steps_keep_role_and_content(tmp_path):
    write(tmp_path, "tau2bench", "a", [(0, "system"), (1, "user"), (3, "assistant")], 3)
    steps = TrajErrBenchTraceSource(tmp_path).get_trace("tau2bench/a")
    assert [(s.step_id, s.type, s.inputs["content"]) for s in steps] == [
        ("0", "system", "system-0"),
        ("1", "user", "user-1"),
        ("3", "assistant", "assistant-3"),
    ]
