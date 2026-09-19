import json
from types import SimpleNamespace

import pytest

from alibi.config import DEFAULT_JUDGE_MODEL, Settings, load_settings
from alibi.judges import make_judge
from alibi.judges.openrouter import OpenRouterJudge, build_schema
from alibi.types import Question

QUESTIONS = [
    Question("anomalous", "Does this window show a mistake?", "boolean"),
    Question("score", "Anomaly score 0-1.", "number"),
    Question("state", "Updated state summary.", "string"),
]


class FakeClient:
    def __init__(self, content, usage=None):
        self.requests = []
        self._response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage or SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=0.0001),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


def test_schema_is_strict_and_typed():
    schema = build_schema(QUESTIONS)
    assert schema["required"] == ["anomalous", "score", "state"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["score"]["type"] == "number"


def test_evaluate_parses_answers_and_records_usage():
    answers = {"anomalous": True, "score": 0.8, "state": "agent retried the same tool"}
    client = FakeClient(json.dumps(answers))
    judge = OpenRouterJudge(client=client, model="some/model")

    assert judge.evaluate("prior state\nchunk", QUESTIONS) == answers
    request = client.requests[0]
    assert request["model"] == "some/model"
    assert request["response_format"]["type"] == "json_schema"
    assert judge.calls[0].cost == 0.0001
    assert judge.calls[0].prompt_tokens == 100


def test_evaluate_rejects_missing_answers():
    judge = OpenRouterJudge(client=FakeClient(json.dumps({"anomalous": False})), model="m")
    with pytest.raises(ValueError, match="omitted"):
        judge.evaluate("s", QUESTIONS)


def test_settings_default_model(monkeypatch):
    monkeypatch.setattr("alibi.config.load_dotenv", lambda: None)
    monkeypatch.delenv("ALIBI_JUDGE_MODEL", raising=False)
    assert (
        load_settings().judge_model == DEFAULT_JUDGE_MODEL == "deepseek/deepseek-v4-flash-0731:free"
    )
    monkeypatch.setenv("ALIBI_JUDGE_MODEL", "other/model")
    assert load_settings().judge_model == "other/model"


def test_make_judge_requires_key():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_judge(Settings(openrouter_api_key=None, judge_model="m"))


def test_make_judge_uses_configured_model():
    judge = make_judge(Settings(openrouter_api_key="sk-test", judge_model="x/y:free"))
    assert judge.model == "x/y:free"


def test_make_judge_refuses_paid_model_by_default():
    with pytest.raises(RuntimeError, match="paid"):
        make_judge(Settings(openrouter_api_key="sk-test", judge_model="openai/gpt-5"))
    allowed = Settings(
        openrouter_api_key="sk-test", judge_model="openai/gpt-5", allow_paid_models=True
    )
    assert make_judge(allowed).model == "openai/gpt-5"


def test_reasoning_setting_is_sent_as_extra_body():
    judge = make_judge(
        Settings(openrouter_api_key="sk-test", judge_model="x/y:free", judge_reasoning="off")
    )
    client = FakeClient(json.dumps({"anomalous": True, "score": 0.1, "state": "s"}))
    judge._client = client
    judge.evaluate("s", QUESTIONS)
    assert client.requests[0]["extra_body"] == {"reasoning": {"enabled": False}}
    with pytest.raises(RuntimeError, match="ALIBI_JUDGE_REASONING"):
        make_judge(Settings(openrouter_api_key="k", judge_model="x:free", judge_reasoning="max"))
