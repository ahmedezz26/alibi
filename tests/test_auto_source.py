import json

import pytest

from alibi.sources.auto import load_steps


def test_json_file_is_read_with_the_json_adapter(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"type": "user", "inputs": {"q": "hi"}}]))
    assert load_steps(str(p))[0].inputs == {"q": "hi"}


def test_jsonl_file_is_read_as_a_claude_code_session(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}))
    assert load_steps(str(p))[0].type == "user"


def test_a_missing_file_is_reported_as_such_not_sent_to_langsmith(tmp_path):
    """A path that does not exist must not fall through to the LangSmith adapter."""
    with pytest.raises(FileNotFoundError, match="no such trace file"):
        load_steps(str(tmp_path / "nope" / "session.jsonl"))
