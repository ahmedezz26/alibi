import json

from alibi import cli


def trace_file(tmp_path, n=3):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"type": "user", "inputs": {"q": "hi"}}] * n))
    return str(p)


def test_short_trace_prints_the_gate_message_without_a_judge(tmp_path, capsys, monkeypatch):
    def no_judge(*_args, **_kwargs):
        raise AssertionError("a gated trace must not build a judge")

    monkeypatch.setattr(cli, "make_judge", no_judge)
    assert cli.main(["diagnose", trace_file(tmp_path)]) == 0
    assert "direct read" in capsys.readouterr().out


def test_json_output_is_a_diagnosis(tmp_path, capsys):
    assert cli.main(["diagnose", trace_file(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gated"] is True and out["suspects"] == []


def test_long_trace_needs_the_jev_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-tokens", "0"]) == 2
    assert "ALIBI_JUDGE_BACKEND=typesafe" in capsys.readouterr().err
