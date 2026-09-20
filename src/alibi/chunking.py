"""Split a trace into overlapping, token-budgeted windows for the judge."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from alibi.types import Chunk, Step

TokenCounter = Callable[[str], int]


_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
# json.dumps renders non-ASCII (e.g. Chinese) as \uXXXX; measured on Jev at ~4.75 real
# tokens per escape, far more than 6 chars / 4. Undercounting overflowed Jev's 32K limit.
TOKENS_PER_ESCAPE = 5


_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_DIGIT = re.compile(r"[0-9]")


def approx_tokens(text: str) -> int:
    """Cheap estimate: ~4 ASCII chars per token, but 1 token per digit and per non-ASCII
    char, 5 per \\uXXXX escape (measured on Jev: ~1.01 per Chinese char, ~1 per char of
    numeric tool output). Deliberately errs high. Swap in a real tokenizer when it matters."""
    escapes = len(_UNICODE_ESCAPE.findall(text))
    rest = _UNICODE_ESCAPE.sub("", text)
    non_ascii = len(_NON_ASCII.findall(rest))
    digits = len(_DIGIT.findall(rest))
    other = len(rest) - non_ascii - digits
    return max(1, other // 4 + digits + non_ascii + TOKENS_PER_ESCAPE * escapes)


def render_step(position: int, step: Step) -> str:
    parts = [f"[step {position}] {step.type}" + (f" {step.name}" if step.name else "")]
    parts.append(f"inputs: {json.dumps(step.inputs, default=str, ensure_ascii=False)}")
    if step.outputs is not None:
        parts.append(f"outputs: {json.dumps(step.outputs, default=str, ensure_ascii=False)}")
    if step.error:
        parts.append(f"error: {step.error}")
    return "\n".join(parts)


def trace_tokens(steps: Sequence[Step], count_tokens: TokenCounter = approx_tokens) -> int:
    return sum(count_tokens(render_step(i, s)) for i, s in enumerate(steps))


def clip_step(text: str, max_tokens: int, count_tokens: TokenCounter = approx_tokens) -> str:
    """Shrink one rendered step to fit ``max_tokens`` by keeping its head and tail.

    Real traces can hold a single tool output far larger than any judge's input limit
    (seen: ~265K tokens of numbers). Leaves text alone if the counter can't be satisfied.
    """
    if count_tokens(text) <= max_tokens:
        return text
    keep = int(max_tokens * 0.9) * 4  # chars, ~4 per token
    for _ in range(8):
        half = max(1, keep // 2)
        omitted = len(text) - 2 * half
        clipped = f"{text[:half]} …[{omitted} chars omitted]… {text[-half:]}"
        if count_tokens(clipped) <= max_tokens:
            return clipped
        if count_tokens(clipped) >= count_tokens(text):
            return text  # counter doesn't shrink with length (e.g. a fixed-cost test counter)
        keep //= 2
    return clipped


def chunk_trace(
    steps: Sequence[Step],
    max_tokens: int = 28_000,
    overlap_tokens: int = 2_000,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Chunk]:
    """Greedily pack whole steps into windows of at most ``max_tokens``.

    Each new window starts by repeating trailing steps of the previous one, up to
    ``overlap_tokens``, so a failure moment isn't split across a hard boundary.
    A single step larger than ``max_tokens`` is clipped (head and tail kept) and becomes
    its own chunk.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    rendered = [render_step(i, s) for i, s in enumerate(steps)]
    costs = [count_tokens(r) for r in rendered]
    for i, cost in enumerate(costs):
        if cost > max_tokens:
            rendered[i] = clip_step(rendered[i], max_tokens, count_tokens)
            costs[i] = count_tokens(rendered[i])

    chunks: list[Chunk] = []
    start = 0
    while start < len(steps):
        end, used = start, 0
        while end < len(steps) and (end == start or used + costs[end] <= max_tokens):
            used += costs[end]
            end += 1
        chunks.append(
            Chunk(
                index=len(chunks),
                steps=tuple(steps[start:end]),
                start=start,
                end=end,
                text="\n\n".join(rendered[start:end]),
                tokens=used,
            )
        )
        if end == len(steps):
            break
        # Walk back from `end` to take overlap, but always make forward progress.
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap + costs[next_start - 1] <= overlap_tokens:
            next_start -= 1
            overlap += costs[next_start]
        start = next_start
    return chunks
