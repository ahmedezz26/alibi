"""TrajErrBench (THU-KEG/TrajDebug, EMNLP 2026 Findings): real failed agent runs with a
hand-labelled critical error step. 400 tau2-bench (customer service, short) and 86
SWE-Bench Pro (coding, long: median ~51K tokens) trajectories. English edition by default.

Labels are message ``step`` ids (not positions); ``critical_step`` is the 0-based
position of that message. Code/annotations are MIT; trajectories keep upstream terms.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from alibi.types import Annotation, Step

DEFAULT_DATA_DIR = Path("data/trajdebug/data/trajerrbench")
SUBSETS = ("tau2bench", "swebenchpro")


@lru_cache(maxsize=512)
def _read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _files(data_dir: Path, subset: str, lang: str) -> list[Path]:
    return sorted((data_dir / lang / subset).glob("*.json"))


def load_annotations(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    subsets: tuple[str, ...] = SUBSETS,
    lang: str = "en",
) -> list[Annotation]:
    out = []
    for subset in subsets:
        for f in _files(Path(data_dir), subset, lang):
            d = _read(str(f))
            label = d["metadata"]["annotation"]
            ids = [m["step"] for m in d["messages"]]
            out.append(
                Annotation(
                    trace_id=f"{subset}/{f.stem}",
                    split=subset,
                    critical_step=ids.index(label["critical_error_step"]),
                    category=str(label.get("critical_error_type") or "unknown"),
                )
            )
    return out


class TrajErrBenchTraceSource:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR, lang: str = "en") -> None:
        self._data_dir = Path(data_dir)
        self._lang = lang

    def get_trace(self, trace_id: str) -> list[Step]:
        subset, stem = trace_id.split("/", 1)
        d = _read(str(self._data_dir / self._lang / subset / f"{stem}.json"))
        return [
            Step(
                step_id=str(m["step"]),
                type=m.get("role") or "unknown",
                timestamp=None,
                inputs={"content": m.get("content") or ""},
                name=m.get("name") if m.get("name") != m.get("role") else None,
            )
            for m in d["messages"]
        ]
