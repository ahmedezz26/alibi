import json

from alibi.cli import select_ids
from alibi.types import Annotation


def ann(i):
    return Annotation(f"t/{i}", "test", critical_step=0, category="c")


def test_select_ids_keeps_listed_traces_in_file_order(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"ids": ["t/3", "t/1"]}))
    assert [a.trace_id for a in select_ids([ann(1), ann(2), ann(3)], path)] == ["t/3", "t/1"]


def test_select_ids_accepts_plain_list_and_rejects_unknown(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["t/1", "t/9"]))
    try:
        select_ids([ann(1)], path)
    except ValueError as e:
        assert "t/9" in str(e)
    else:
        raise AssertionError("unknown id accepted")
