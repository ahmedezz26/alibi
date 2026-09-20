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
    assert cli.main(["diagnose", trace, "--min-tokens", "0", "--max-tokens", "1"]) == 0
    out = capsys.readouterr().out
    assert "ceiling" in out and "ALIBI_MAX_TRACE_TOKENS" in out


def test_a_missing_trace_file_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["diagnose", str(tmp_path / "gone.jsonl")]) == 2
    assert "no such trace file" in capsys.readouterr().err


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
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-tokens", "0"]) == 2
    assert "ALIBI_JUDGE_BACKEND=typesafe" in capsys.readouterr().err
