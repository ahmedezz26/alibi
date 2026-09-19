from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL = ("pays personally", "budget-constrained", "Jev credit left", "credit left")


def test_license_is_mit():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")


def test_readme_pitch_and_no_emojis():
    text = (ROOT / "README.md").read_text()
    assert "We check whether your agent has an alibi for what it did." in text
    assert all(ord(ch) < 0x2600 for ch in text)


def test_no_personal_budget_remarks_in_public_docs():
    for rel in ("README.md", "CLAUDE.md", "docs/research-log.md"):
        text = (ROOT / rel).read_text()
        assert not any(p in text for p in PERSONAL), rel
