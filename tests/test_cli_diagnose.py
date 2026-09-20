import json
import subprocess
import sys

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


def test_max_tokens_overrides_the_cost_ceiling(tmp_path, capsys):
    """The ceiling is overridable from the command line, not only through the environment."""
    trace = trace_file(tmp_path, n=3)
    assert cli.main(["diagnose", trace, "--min-trace-tokens", "0", "--max-trace-tokens", "1"]) == 0
    out = capsys.readouterr().out
    assert "ceiling" in out and "ALIBI_MAX_TRACE_TOKENS" in out


def test_a_missing_trace_file_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["diagnose", str(tmp_path / "gone.jsonl")]) == 2
    assert "no such trace file" in capsys.readouterr().err


def test_an_empty_trace_needs_no_judge(tmp_path, capsys, monkeypatch):
    """diagnose() answers an empty trace for free, so the CLI must not demand a backend."""

    def no_judge(*_a, **_k):
        raise AssertionError("an empty trace must not build a judge")

    monkeypatch.setattr(cli, "make_judge", no_judge)
    p = tmp_path / "empty.json"
    p.write_text("[]")
    assert cli.main(["diagnose", str(p), "--min-trace-tokens", "0"]) == 0
    assert "no steps" in capsys.readouterr().out


def test_a_malformed_trace_file_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert cli.main(["diagnose", str(p)]) == 2
    assert "could not read" in capsys.readouterr().err


def test_the_flag_spellings_from_0_1_3_still_work(tmp_path, capsys):
    """--min-tokens/--max-tokens shipped in 0.1.3; scripts using them must keep working."""
    trace = trace_file(tmp_path, n=3)
    assert cli.main(["diagnose", trace, "--min-tokens", "0", "--max-tokens", "1"]) == 0
    assert "ceiling" in capsys.readouterr().out


def test_the_backend_message_names_every_variable_it_needs(tmp_path, capsys, monkeypatch):
    """Following the message must be enough: the paid-model guard is the third variable."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    err = capsys.readouterr().err
    for var in ("ALIBI_JUDGE_BACKEND=typesafe", "TYPESAFE_API_KEY", "ALIBI_ALLOW_PAID_MODELS=1"):
        assert var in err


def test_a_missing_key_is_a_clean_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_a_bad_timestamp_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "ts.json"
    p.write_text(json.dumps([{"type": "user", "timestamp": "not-a-date", "inputs": {}}]))
    assert cli.main(["diagnose", str(p)]) == 2
    assert "could not read" in capsys.readouterr().err


def test_module_entry_point_runs(tmp_path):
    """`python -m alibi.cli` must define every handler before main() dispatches."""
    out = subprocess.run(
        [sys.executable, "-m", "alibi.cli", "diagnose", trace_file(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "direct read" in out.stdout


def test_long_trace_needs_the_jev_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    assert "ALIBI_JUDGE_BACKEND=typesafe" in capsys.readouterr().err
