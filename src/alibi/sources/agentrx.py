"""TraceSource + annotations for the AgentRx benchmark (microsoft/AgentRx on Hugging Face).

Files are expected in ``data/agentrx/``: ``{split}.jsonl`` holds annotations and
``{split}_dataset.jsonl`` holds trajectories (see SPLITS). AgentRx step numbers are
1-based and contiguous; ``Step.step_id`` keeps them, trace positions are 0-based.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from alibi.types import Annotation, Step

DEFAULT_DATA_DIR = Path("data/agentrx")

# split -> (annotation file, trajectory file, prefix joining annotation ids to trajectory ids)
SPLITS = {
    "tau_retail": ("tau_retail.jsonl", "tau_retail_dataset.jsonl", "tau_retail_"),
    "magentic_one": ("magentic_one.jsonl", "magentic_dataset.jsonl", ""),
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@cache
def _trajectories(data_dir: Path, split: str) -> dict[str, dict]:
    rows = _read_jsonl(data_dir / SPLITS[split][1])
    return {row["trajectory_id"]: row for row in rows}


def load_annotations(
    data_dir: str | Path = DEFAULT_DATA_DIR, splits: tuple[str, ...] = tuple(SPLITS)
) -> list[Annotation]:
    data_dir = Path(data_dir)
    out = []
    for split in splits:
        ann_file, _, prefix = SPLITS[split]
        for row in _read_jsonl(data_dir / ann_file):
            root = next(
                f for f in row["failures"] if f["failure_id"] == row["root_cause_failure_id"]
            )
            out.append(
                Annotation(
                    trace_id=prefix + row["trajectory_id"],
                    split=split,
                    critical_step=root["step_number"] - 1,
                    category=root["failure_category"],
                    failure_summary=row["failure_summary"],
                )
            )
    return out


class AgentRxTraceSource:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)

    def get_trace(self, trace_id: str) -> list[Step]:
        for split in SPLITS:
            row = _trajectories(self._data_dir, split).get(trace_id)
            if row is not None:
                return [_to_step(s) for s in sorted(row["steps"], key=lambda s: s["index"])]
        raise KeyError(f"trajectory {trace_id!r} not found in {self._data_dir}")


def _to_step(raw: dict) -> Step:
    subs = sorted(raw["substeps"], key=lambda s: s["sub_index"])
    return Step(
        step_id=str(raw["index"]),
        type=subs[0]["role"] if subs else "unknown",
        timestamp=None,
        inputs={"content": "\n\n".join(s["content"] or "" for s in subs)},
    )
