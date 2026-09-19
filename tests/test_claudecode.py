import json

from alibi.sources.claudecode import ClaudeCodeTraceSource


def write(tmp_path, entries):
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return p


def test_messages_tool_calls_and_results_become_steps(tmp_path):
    p = write(
        tmp_path,
        [
            {"type": "mode", "mode": "default"},
            {"type": "user", "message": {"role": "user", "content": "fix the test"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "Running the tests."},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "1 failed"}],
                },
            },
        ],
    )
    steps = ClaudeCodeTraceSource().get_trace(str(p))
    assert [(s.type, s.name) for s in steps] == [
        ("user", None),
        ("assistant", None),
        ("assistant", "Bash"),
        ("tool", None),
    ]
    assert steps[2].inputs == {"command": "pytest"}
    assert steps[3].inputs == {"content": "1 failed"}


def test_unknown_lines_are_skipped(tmp_path):
    p = write(tmp_path, [{"type": "attachment"}, {"type": "user", "message": {"content": "hi"}}])
    assert len(ClaudeCodeTraceSource().get_trace(str(p))) == 1
