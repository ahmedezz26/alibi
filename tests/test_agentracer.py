import json

from alibi.sources.agentracer import AgenTracerTraceSource, load_annotations


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_labels_match_step_ids_not_positions(tmp_path):
    one_based = [{"step": i, "role": "assistant", "content": f"a{i}"} for i in (1, 2, 3)]
    zero_based = [{"step": i, "role": "user", "name": "coder", "content": f"c{i}"} for i in (0, 1)]
    write(
        tmp_path / "agentic/test.jsonl",
        [{"history": one_based, "mistake_step": "2", "error_source": "Injection Error"}],
    )
    write(tmp_path / "coding/test.jsonl", [{"history": zero_based, "mistake_step": 1}])
    write(tmp_path / "math/test.jsonl", [])

    agentic, coding = load_annotations(tmp_path)
    assert (agentic.trace_id, agentic.critical_step, agentic.category) == (
        "agentic/test/0",
        1,
        "Injection Error",
    )
    assert (coding.split, coding.critical_step) == ("coding", 1)

    steps = AgenTracerTraceSource(tmp_path).get_trace("coding/test/0")
    assert [(s.step_id, s.name, s.inputs["content"]) for s in steps] == [
        ("0", "coder", "c0"),
        ("1", "coder", "c1"),
    ]
