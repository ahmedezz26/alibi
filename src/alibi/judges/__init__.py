"""Judge construction. The only place that imports concrete judge backends."""

from __future__ import annotations

from alibi.config import Settings, load_settings
from alibi.judges.base import CallRecord, Judge

__all__ = ["CallRecord", "Judge", "make_judge"]


def make_judge(settings: Settings | None = None) -> Judge:
    settings = settings or load_settings()
    if settings.judge_backend == "typesafe":
        return _make_typesafe(settings)
    if settings.judge_backend != "openrouter":
        raise RuntimeError(
            f"ALIBI_JUDGE_BACKEND must be openrouter|typesafe, got {settings.judge_backend!r}"
        )
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
    if not settings.judge_model.endswith(":free") and not settings.allow_paid_models:
        raise RuntimeError(
            f"judge model {settings.judge_model!r} is paid; use a ':free' model or set "
            "ALIBI_ALLOW_PAID_MODELS=1 (only with the user's explicit approval)"
        )

    from openai import OpenAI

    from alibi.judges.openrouter import OpenRouterJudge

    client = OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        max_retries=settings.judge_max_retries,
    )
    return OpenRouterJudge(
        client=client, model=settings.judge_model, reasoning=_reasoning(settings.judge_reasoning)
    )


def _make_typesafe(settings: Settings) -> Judge:
    if not settings.typesafe_api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (see .env.example)")
    if not settings.allow_paid_models:  # TypeSafe has no free tier
        raise RuntimeError(
            "the TypeSafe judge is paid; set ALIBI_ALLOW_PAID_MODELS=1 "
            "(only with the user's explicit approval)"
        )

    from alibi.judges.typesafe import TypeSafeJudge

    return TypeSafeJudge(
        model=settings.typesafe_model,
        api_key=settings.typesafe_api_key,
        price_per_mtok=settings.typesafe_price_per_mtok,
    )


def _reasoning(level: str) -> dict | None:
    if not level:
        return None
    if level == "off":
        return {"enabled": False}
    if level in ("low", "medium", "high"):
        return {"effort": level}
    raise RuntimeError(f"ALIBI_JUDGE_REASONING must be off|low|medium|high, got {level!r}")
