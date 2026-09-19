"""Single source of runtime configuration. Read settings here, nowhere else."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Longest trace the judge still localizes reliably in one call. Measured, not the advertised
# context: gpt-5-nano on AgentRx collapsed past ~10K tokens (1/23 exact) despite a 400K window.
DEFAULT_JUDGE_RELIABLE_TOKENS = 10_000
DEFAULT_TYPESAFE_MODEL = "jev-latest"
# TypeSafe list price (2026-09): input tokens only, output is free.
DEFAULT_TYPESAFE_PRICE_PER_MTOK = 0.042
# The method only beats a single whole-trace read on long traces (wins measured at 59K and
# 120K tokens; no edge at 38K). Shorter traces are gated: one direct read is enough.
DEFAULT_MIN_TRACE_TOKENS = 50_000
# Jev bills per input token and every token is read about twice (forward, then the look-back),
# so a very long trace is refused rather than silently spent. 250K tokens is about $0.02.
DEFAULT_MAX_TRACE_TOKENS = 250_000


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    judge_model: str
    langsmith_api_key: str | None = None
    hf_token: str | None = None
    openrouter_base_url: str = OPENROUTER_BASE_URL
    # Traces up to this many tokens get one whole-trace call; longer ones are windowed.
    judge_reliable_tokens: int = DEFAULT_JUDGE_RELIABLE_TOKENS
    # Window size for long traces; defaults to the reliable length.
    judge_window_tokens: int = DEFAULT_JUDGE_RELIABLE_TOKENS
    # Minimum trace length to use the windowed V2 method; shorter traces get one whole-trace read.
    min_trace_tokens: int = DEFAULT_MIN_TRACE_TOKENS
    max_trace_tokens: int = DEFAULT_MAX_TRACE_TOKENS
    # SDK-level retries with backoff (honours Retry-After); free models are rate limited.
    judge_max_retries: int = 8
    # Spend guard: only ":free" judge models unless explicitly allowed.
    allow_paid_models: bool = False
    # "" = model default; "off" disables thinking; "low"/"medium"/"high" set the effort.
    judge_reasoning: str = ""
    # "openrouter" (free-form LLM judge) or "typesafe" (Jev, a System One judge).
    judge_backend: str = "openrouter"
    typesafe_api_key: str | None = None
    typesafe_model: str = DEFAULT_TYPESAFE_MODEL
    typesafe_price_per_mtok: float = DEFAULT_TYPESAFE_PRICE_PER_MTOK


def load_settings() -> Settings:
    load_dotenv()
    reliable = int(os.environ.get("ALIBI_JUDGE_RELIABLE_TOKENS") or DEFAULT_JUDGE_RELIABLE_TOKENS)
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        judge_model=os.environ.get("ALIBI_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
        langsmith_api_key=os.environ.get("LANGSMITH_API_KEY") or None,
        hf_token=os.environ.get("HF_TOKEN") or None,
        judge_reliable_tokens=reliable,
        judge_window_tokens=int(os.environ.get("ALIBI_JUDGE_WINDOW_TOKENS") or reliable),
        min_trace_tokens=int(os.environ.get("ALIBI_MIN_TRACE_TOKENS") or DEFAULT_MIN_TRACE_TOKENS),
        max_trace_tokens=int(os.environ.get("ALIBI_MAX_TRACE_TOKENS") or DEFAULT_MAX_TRACE_TOKENS),
        judge_max_retries=int(os.environ.get("ALIBI_JUDGE_MAX_RETRIES") or 8),
        allow_paid_models=os.environ.get("ALIBI_ALLOW_PAID_MODELS") == "1",
        judge_reasoning=(os.environ.get("ALIBI_JUDGE_REASONING") or "").strip().lower(),
        judge_backend=(os.environ.get("ALIBI_JUDGE_BACKEND") or "openrouter").strip().lower(),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY") or None,
        typesafe_model=os.environ.get("ALIBI_TYPESAFE_MODEL") or DEFAULT_TYPESAFE_MODEL,
        typesafe_price_per_mtok=float(
            os.environ.get("ALIBI_TYPESAFE_PRICE_PER_MTOK") or DEFAULT_TYPESAFE_PRICE_PER_MTOK
        ),
    )
