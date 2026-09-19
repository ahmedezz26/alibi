import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    return json.loads((ROOT / rel).read_text())


def test_marketplace_points_at_the_plugin_directory():
    m = load(".claude-plugin/marketplace.json")
    assert m["name"] == "alibi"
    [p] = m["plugins"]
    assert p["name"] == "alibi" and (ROOT / p["source"] / ".claude-plugin/plugin.json").is_file()


def test_plugin_manifest_and_mcp_server():
    assert load("plugin/.claude-plugin/plugin.json")["name"] == "alibi"
    server = load("plugin/.mcp.json")["mcpServers"]["alibi"]
    assert server["command"] == "uvx"
    assert "agent-alibi" in server["args"]  # the PyPI distribution, not a git build
    assert server["env"]["TYPESAFE_API_KEY"] == "${TYPESAFE_API_KEY}"


def test_skill_has_frontmatter_and_no_emojis():
    text = (ROOT / "plugin/skills/diagnose-trace/SKILL.md").read_text()
    assert text.startswith("---\n") and "description:" in text.split("---")[1]
    assert all(ord(ch) < 0x2600 for ch in text)
