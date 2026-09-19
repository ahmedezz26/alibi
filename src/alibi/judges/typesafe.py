"""Judge backend for TypeSafe's System One models (Jev), via ``langchain-typesafe``.

Jev answers typed questions with probabilities and cannot write free text, so only
``noul``, ``score`` and ``choice`` questions are accepted. Answers are returned as plain
values so pipeline code stays backend-agnostic:

- noul   -> probability the statement is true, in [0, 1]
- score  -> expected level scaled to [0, 1] (0 = first level, 1 = last level)
- choice -> {"choice": option, "probabilities": {option: p}, "confidence": c}
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from alibi.judges.base import CallRecord
from alibi.types import Question

ClassifierFactory = Callable[[dict[str, Any], str], Any]


def _default_factory(api_key: str | None) -> ClassifierFactory:
    def build(questions: dict[str, Any], model: str) -> Any:
        from langchain_typesafe import TypeSafeClassifier

        kwargs = {"api_key": api_key} if api_key else {}
        return TypeSafeClassifier(questions=questions, model=model, **kwargs)

    return build


def to_typesafe(question: Question) -> Any:
    from langchain_typesafe import Choice, Noul, NoulCriteria, Score

    if question.answer_type == "noul":
        criteria = NoulCriteria(**question.criteria) if question.criteria else None
        return Noul(instructions=question.prompt, criteria=criteria)
    if question.answer_type == "score":
        return Score(instructions=question.prompt, criteria=list(question.criteria))
    if question.answer_type == "choice":
        return Choice(instructions=question.prompt, criteria=dict(question.criteria))
    raise ValueError(
        f"question {question.id!r} is {question.answer_type!r}; System One judges answer "
        "noul/score/choice and cannot write free text"
    )


class TypeSafeJudge:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        price_per_mtok: float = 0.0,
        classifier_factory: ClassifierFactory | None = None,
    ) -> None:
        self.model = model
        self.reasoning = None  # no reasoning knob; kept for comparable run records
        self._price = price_per_mtok
        self._factory = classifier_factory or _default_factory(api_key)
        self.calls: list[CallRecord] = []

    def evaluate(self, state: str, questions: list[Question]) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one question is required")
        converted = {q.id: to_typesafe(q) for q in questions}
        started = time.perf_counter()
        response = self._factory(converted, self.model).invoke(state)
        latency = time.perf_counter() - started

        tokens = response.usage.input_tokens
        self.calls.append(
            CallRecord(
                model=response.model,
                latency_s=latency,
                prompt_tokens=tokens,
                completion_tokens=response.usage.output_tokens,
                cost=(tokens or 0) * self._price / 1e6,
            )
        )
        return {q.id: _value(q, response.answers[q.id]) for q in questions}


def _value(question: Question, answer: Any) -> Any:
    if question.answer_type == "noul":
        return answer.noul
    if question.answer_type == "score":
        return answer.score / (len(question.criteria) - 1)
    return {
        "choice": answer.choice,
        "probabilities": dict(answer.probabilities),
        "confidence": answer.confidence,
    }
