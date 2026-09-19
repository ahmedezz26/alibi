"""TraceSource + annotations for the AgenTracer data release (github.com/bingreeky/AgenTracer).

Expected layout: ``data/agentracer/agentracer-data-v1.0.0/{domain}/{split}.jsonl``, one failed
trajectory per line with a ``history`` list and a single ``mistake_step``. Step numbering is
1-based in ``agentic`` and 0-based elsewhere, so the label is matched against each history
item's ``step`` field rather than assumed to be a position.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from alibi.types import Annotation, Step

DEFAULT_DATA_DIR = Path("data/agentracer/agentracer-data-v1.0.0")
DOMAINS = ("agentic", "coding", "math")


@cache
def _rows(data_dir: Path, domain: str, split: str) -> tuple[dict, ...]:
    path = data_dir / domain / f"{split}.jsonl"
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def _trace_id(domain: str, split: str, index: int) -> str:
    return f"{domain}/{split}/{index}"


def load_annotations(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    domains: tuple[str, ...] = DOMAINS,
    split: str = "test",
) -> list[Annotation]:
    data_dir = Path(data_dir)
    out = []
    for domain in domains:
        for i, row in enumerate(_rows(data_dir, domain, split)):
            step_ids = [str(h.get("step")) for h in row["history"]]
            out.append(
                Annotation(
                    trace_id=_trace_id(domain, split, i),
                    split=domain,
                    critical_step=step_ids.index(str(row["mistake_step"])),
                    category=str(row.get("error_source") or "unknown"),
                    failure_summary=str(row.get("mistake_reason") or ""),
                )
            )
    return out


class AgenTracerTraceSource:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)

    def get_trace(self, trace_id: str) -> list[Step]:
        domain, split, index = trace_id.split("/")
        row = _rows(self._data_dir, domain, split)[int(index)]
        return [_to_step(h) for h in row["history"]]


def _to_step(item: dict) -> Step:
    return Step(
        step_id=str(item.get("step")),
        type=item.get("role") or "unknown",
        timestamp=None,
        inputs={"content": item.get("content") or ""},
        name=item.get("name"),
    )
