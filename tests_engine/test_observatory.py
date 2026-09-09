"""The observatory transports real, finite, replayable engine telemetry."""

import json
import threading
from types import SimpleNamespace

import pytest

import compare_server as server
from magnum_opus_v2.journal import CognitionJournal


@pytest.fixture
def dashboard(monkeypatch):
    journal = CognitionJournal()
    messages = ["An actual spontaneous message"]

    def drain():
        copy = list(messages)
        messages.clear()
        return copy

    engine = SimpleNamespace(
        profile=SimpleNamespace(model_name="fixture-model"), device="cpu",
        _last_causes={"turn": 1, "chosen_future": {"goodness": 0.2}},
        snapshot=lambda: {"bus": {"tick_count": 5},
                          "executive": {"wall_seconds_since_speech": float("inf")}},
        journal=journal, drain_autonomous_messages=drain,
        maybe_log_emotion=lambda payload: None, _history_lock=threading.Lock(),
        chat_history=[{"role": "user", "content": "Hello"},
                      {"role": "assistant", "content": "Hi"}],
    )
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "engine_history", [])
    return server.app.test_client(), engine


def first_event(client, headers=None):
    response = client.get("/api/stream", headers=headers or {}, buffered=False)
    frame = next(iter(response.response)).decode()
    response.close()
    line = next(line for line in frame.splitlines() if line.startswith("data: "))
    payload = json.loads(line[6:], parse_constant=lambda value: pytest.fail(f"Invalid JSON number {value}"))
    return frame, payload


def test_default_observatory_keeps_comparison_and_face_routes(dashboard):
    client, _ = dashboard
    assert b"Mind Observatory" in client.get("/").data
    assert b"raw-messages" in client.get("/compare").data
    assert client.get("/face").status_code == 200
    assert client.get("/static/observatory.js").status_code == 200
    assert client.get("/static/observatory.css").status_code == 200


def test_initial_telemetry_is_valid_json_even_before_speech(dashboard):
    client, _ = dashboard
    payload = client.get("/api/status").get_json()
    assert payload["executive"]["wall_seconds_since_speech"] is None
    assert payload["runtime"]["model"] == "fixture-model"
    frame, payload = first_event(client)
    assert frame.startswith("id: ")
    assert payload["executive"]["wall_seconds_since_speech"] is None
    assert payload["last_reply_context"]["turn"] == 1


def test_spontaneous_speech_is_available_to_multiple_observers(dashboard):
    client, engine = dashboard
    _, first = first_event(client)
    _, second = first_event(client)
    assert first["autonomous"] == ["An actual spontaneous message"]
    assert second["autonomous"] == []
    for payload in (first, second):
        replies = [e for e in payload["journal"] if e["kind"] == "autonomous_reply"]
        assert len(replies) == 1
        assert replies[0]["payload"]["reply"] == "An actual spontaneous message"
    assert len(engine.journal.ring) == 1


def test_reconnect_replays_only_events_after_cursor(dashboard):
    client, engine = dashboard
    for n in range(5):
        engine.journal.emit("future_considered", word=f"future {n}")
    _, payload = first_event(client, {"Last-Event-ID": "3"})
    assert [e["id"] for e in payload["journal"]] == [4, 5, 6]


def test_conversation_restoration_does_not_trigger_generation(dashboard):
    client, engine = dashboard
    payload = client.get("/api/conversation").get_json()
    assert payload["messages"] == engine.chat_history
    assert payload["session_id"]
