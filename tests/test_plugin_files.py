import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_MINOR = "0.1"
NEXT_MINOR = "0.2"


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
    # the whole command, in order: a pinned PyPI distribution, then the console script
    assert server["args"] == ["--from", f"agent-alibi>=0.1,<{NEXT_MINOR}", "alibi-mcp"]
    assert server["env"]["TYPESAFE_API_KEY"] == "${TYPESAFE_API_KEY}"


def test_plugin_version_matches_the_distribution():
    """The plugin ships from git while the server ships from PyPI; the two must not drift."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert load("plugin/.claude-plugin/plugin.json")["version"] == project["version"]
    major, minor, _ = project["version"].split(".")
    assert f"{major}.{minor}" == CURRENT_MINOR, "bump the plugin's --from pin with the version"


def test_skill_has_frontmatter_and_no_emojis():
    text = (ROOT / "plugin/skills/diagnose-trace/SKILL.md").read_text()
    assert text.startswith("---\n") and "description:" in text.split("---")[1]
    assert all(ord(ch) < 0x2600 for ch in text)
