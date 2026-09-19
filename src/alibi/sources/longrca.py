"""TraceSource + annotations for LongRCA Bench (arXiv 2608.15242, HF CLoud5-real/longrca-bench).

1,140 observed (non-injected) failed multi-agent trajectories with a human label for the
earliest decisive root-cause step. Expected file: ``data/longrca/longrca-full.jsonl`` (one row
per trajectory, converted from the release's ``longrca-full.parquet``). Steps are 0-based;
the label is matched against each history item's ``step`` field. Terminal output arrives
as ``role=user, name=Computer_terminal`` and is mapped to a ``tool`` step.
No licence is stated upstream: cite the paper and don't reshare the data.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from alibi.types import Annotation, Step

DEFAULT_DATA_DIR = Path("data/longrca")
FILE_NAME = "longrca-full.jsonl"
SUBSETS = ("swe_bench_pro", "terminal_bench_2", "travelplanner", "vitabench", "webarena_verified")
TOOL_NAMES = frozenset({"Computer_terminal"})


@cache
def _rows(data_dir: Path) -> dict[str, dict]:
    lines = (data_dir / FILE_NAME).read_text().splitlines()
    rows = (json.loads(line) for line in lines if line.strip())
    return {_trace_id(r["question_ID"]): r for r in rows}


def _trace_id(question_id: str) -> str:
    subset, number = question_id.rsplit("__", 1)
    return f"{subset}/{number}"


def load_annotations(
    data_dir: str | Path = DEFAULT_DATA_DIR, subsets: tuple[str, ...] = SUBSETS
) -> list[Annotation]:
    out = []
    for trace_id, row in _rows(Path(data_dir)).items():
        subset = trace_id.split("/")[0]
        if subset not in subsets:
            continue
        step_ids = [h["step"] for h in row["history"]]
        out.append(
            Annotation(
                trace_id=trace_id,
                split=subset,
                critical_step=step_ids.index(row["mistake_step"]),
                category=str(row.get("mistake_agent") or "unknown"),
                failure_summary=str(row.get("mistake_reason") or ""),
            )
        )
    return out


class LongRCATraceSource:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)

    def get_trace(self, trace_id: str) -> list[Step]:
        return [_to_step(h) for h in _rows(self._data_dir)[trace_id]["history"]]


def _to_step(item: dict) -> Step:
    name = item.get("name")
    kind = "tool" if name in TOOL_NAMES else (item.get("role") or "unknown")
    return Step(
        step_id=str(item.get("step")),
        type=kind,
        timestamp=None,
        inputs={"content": item.get("content") or ""},
        name=name,
    )
