from datetime import date
from pathlib import Path

from ctagent.graph import ReviewDecision
from ctagent.llm import FakeLLM
from ctagent.review import Workspace, decide, open_graph, pending, process, review_payload
from test_graph import CTX, EVENT, good


def test_morning_batch_then_review_survives_reopening(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    second = EVENT.model_copy(update={"episode_id": "ep2", "er_visits_6m": 0, "acute": False})
    graph, tracker = open_graph(ws, FakeLLM(good()), lambda e: CTX)
    assert process(graph, [EVENT, second]) == 2
    queue = pending(tracker)
    assert [q["episode_id"] for q in queue] == ["ep1", "ep2"]  # high tier first
    assert queue[0]["tier"] == "high" and queue[1]["tier"] == "standard"

    # the app restarts: reopen the same workspace
    graph2, tracker2 = open_graph(ws, FakeLLM(good()), lambda e: CTX)
    payload = review_payload(graph2, "ep1")
    assert payload is not None and payload["call_script"]["opening"] == "Hi"
    assert decide(graph2, "ep1", ReviewDecision(decision="approve", reviewer="coord")) == "approved"
    assert review_payload(graph2, "ep1") is None
    assert [q["episode_id"] for q in pending(tracker2)] == ["ep2"]
    assert tracker2.due_check_ins(date(2024, 4, 30))[0][0] == "ep1"
