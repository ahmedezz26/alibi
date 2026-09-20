import os

import pytest

from alibi import config


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Settings come from defaults unless a test sets them.

    ``load_settings`` reads ``.env`` through python-dotenv, and this repo has a real one, so
    without this a developer's ALIBI_MIN_TRACE_TOKENS could silently decide what the gate
    tests assert.
    """
    for key in list(os.environ):
        if key.startswith(("ALIBI_", "TYPESAFE_", "OPENROUTER_", "LANGSMITH_", "HF_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
