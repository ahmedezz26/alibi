from datetime import datetime
from types import SimpleNamespace

from alibi.sources.langsmith import LangSmithTraceSource


def run(id, dotted, run_type="tool", error=None):
    return SimpleNamespace(
        id=id,
        dotted_order=dotted,
        start_time=datetime(2026, 1, 1),
        run_type=run_type,
        inputs={"q": id},
        outputs={"a": id},
        error=error,
        name=f"n{id}",
    )


class FakeClient:
    def __init__(self, runs):
        self.runs, self.kwargs = runs, None

    def list_runs(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.runs)


def test_get_trace_orders_by_dotted_order_and_maps_fields():
    client = FakeClient([run("b", "1.2"), run("a", "1.1", "llm"), run("c", "1.3", error="boom")])
    steps = LangSmithTraceSource(client=client, project_name="p").get_trace("t1")

    assert client.kwargs == {"trace_id": "t1", "project_name": "p"}
    assert [s.step_id for s in steps] == ["a", "b", "c"]
    assert steps[0].type == "llm"
    assert steps[2].error == "boom"
    assert steps[1].outputs == {"a": "b"}
