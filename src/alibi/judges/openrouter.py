"""Judge backend over OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import json
import time
from typing import Any

from alibi.judges.base import CallRecord
from alibi.types import Question

SYSTEM_PROMPT = (
    "You are auditing a window of an AI agent's execution trace. "
    "Answer every question about the window using only the evidence shown. "
    "Respond with JSON matching the provided schema."
)


def build_schema(questions: list[Question]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {q.id: {"type": q.answer_type, "description": q.prompt} for q in questions},
        "required": [q.id for q in questions],
        "additionalProperties": False,
    }


def build_user_prompt(state: str, questions: list[Question]) -> str:
    listed = "\n".join(f"- {q.id} ({q.answer_type}): {q.prompt}" for q in questions)
    return f"{state}\n\nQuestions:\n{listed}"


class OpenRouterJudge:
    def __init__(self, client: Any, model: str, reasoning: dict[str, Any] | None = None) -> None:
        self._client = client
        self.model = model
        self.reasoning = reasoning
        self._extra_body = {"reasoning": reasoning} if reasoning else None
        self.calls: list[CallRecord] = []

    def evaluate(self, state: str, questions: list[Question]) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one question is required")
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, questions)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_answers",
                    "strict": True,
                    "schema": build_schema(questions),
                },
            },
            extra_body=self._extra_body,
        )
        latency = time.perf_counter() - started

        usage = response.usage
        self.calls.append(
            CallRecord(
                model=self.model,
                latency_s=latency,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                cost=getattr(usage, "cost", None),  # OpenRouter-specific usage field
            )
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"judge {self.model} returned an empty response")
        answers = json.loads(content)
        missing = [q.id for q in questions if q.id not in answers]
        if missing:
            raise ValueError(f"judge {self.model} omitted answers for {missing}")
        return answers
