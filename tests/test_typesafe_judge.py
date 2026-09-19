from types import SimpleNamespace

import pytest
from langchain_typesafe import Choice, Noul, Score

from alibi.config import Settings
from alibi.judges import make_judge
from alibi.judges.typesafe import TypeSafeJudge
from alibi.types import Question

QUESTIONS = [
    Question("mistake", "Step 3 is the agent's own mistake.", "noul", {"true": "t", "false": "f"}),
    Question("health", "How is the agent doing so far?", "score", ["fine", "minor", "failed"]),
    Question("pick", "Most suspicious step?", "choice", {"3": "[step 3]", "4": "[step 4]"}),
]


def response():
    return SimpleNamespace(
        model="jev-1.13.0",
        answers={
            "mistake": SimpleNamespace(type="noul", noul=0.8),
            "health": SimpleNamespace(type="score", score=1.5, confidence=0.7),
            "pick": SimpleNamespace(
                type="choice", choice="4", probabilities={"3": 0.2, "4": 0.8}, confidence=0.6
            ),
        },
        usage=SimpleNamespace(input_tokens=2000, output_tokens=10),
    )


class FakeFactory:
    def __init__(self):
        self.built, self.states = [], []

    def __call__(self, questions, model):
        self.built.append((questions, model))
        return SimpleNamespace(invoke=lambda state: self.states.append(state) or response())


def test_converts_questions_to_typesafe_primitives():
    factory = FakeFactory()
    TypeSafeJudge(model="jev-x", classifier_factory=factory).evaluate("state", QUESTIONS)

    questions, model = factory.built[0]
    assert model == "jev-x"
    assert isinstance(questions["mistake"], Noul)
    assert questions["mistake"].criteria.true == "t"
    assert isinstance(questions["health"], Score) and questions["health"].criteria[-1] == "failed"
    assert isinstance(questions["pick"], Choice) and set(questions["pick"].criteria) == {"3", "4"}
    assert factory.states == ["state"]


def test_answers_are_probabilities_and_score_is_scaled_to_unit_interval():
    judge = TypeSafeJudge(model="jev-x", classifier_factory=FakeFactory())
    answers = judge.evaluate("state", QUESTIONS)

    assert answers["mistake"] == 0.8
    assert answers["health"] == pytest.approx(0.75)  # level 1.5 of 0..2
    assert answers["pick"] == {
        "choice": "4",
        "probabilities": {"3": 0.2, "4": 0.8},
        "confidence": 0.6,
    }


def test_records_cost_from_input_tokens():
    judge = TypeSafeJudge(model="jev-x", price_per_mtok=0.042, classifier_factory=FakeFactory())
    judge.evaluate("state", QUESTIONS)
    call = judge.calls[0]
    assert (call.model, call.prompt_tokens) == ("jev-1.13.0", 2000)
    assert call.cost == pytest.approx(2000 * 0.042e-6)


def test_rejects_free_text_questions():
    judge = TypeSafeJudge(model="jev-x", classifier_factory=FakeFactory())
    with pytest.raises(ValueError, match="free text"):
        judge.evaluate("s", [Question("why", "Explain.", "string")])


def test_make_judge_typesafe_needs_key_and_paid_approval():
    base = dict(openrouter_api_key=None, judge_model="m", judge_backend="typesafe")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        make_judge(Settings(**base, allow_paid_models=True))
    with pytest.raises(RuntimeError, match="paid"):
        make_judge(Settings(**base, typesafe_api_key="k"))
    judge = make_judge(Settings(**base, typesafe_api_key="k", allow_paid_models=True))
    assert isinstance(judge, TypeSafeJudge)
