"""The cognition journal — an honest, time-ordered stream of what the
mind did, so a person can watch it think. Modeled on the organism's
Pulse: every entry emitted at a real site, none decorative."""

from magnum_opus_v2.journal import CognitionJournal
from magnum_opus_v2.regions.memory import Memory
from magnum_opus_v2.regions.subconscious import Candidate

import torch


def test_journal_since_and_latest():
    j = CognitionJournal(capacity=8)
    assert j.latest_id() == 0
    assert j.since(0) == []
    j.emit("future_considered", turn=1, word="fall", risk=0.8)
    j.emit("word_chosen", turn=1, word="fall")
    assert j.latest_id() == 2
    evs = j.since(0)
    assert [e["kind"] for e in evs] == ["future_considered", "word_chosen"]
    assert evs[0]["payload"]["word"] == "fall"
    assert evs[0]["turn"] == 1
    # delta semantics: nothing new since the latest id
    assert j.since(j.latest_id()) == []
    # a later reader gets only what came after its cursor
    j.emit("reply_emitted", turn=1, reply="be careful")
    assert [e["kind"] for e in j.since(2)] == ["reply_emitted"]


def test_journal_ring_is_bounded():
    j = CognitionJournal(capacity=4)
    for i in range(10):
        j.emit("emotion_snapshot", dominant="calm", value=i / 10)
    # only the last 4 survive, ids keep climbing (honest, no reuse)
    evs = j.since(0)
    assert len(evs) == 4
    assert evs[-1]["id"] == 10


def test_memory_recent_flags_false_memories():
    mem = Memory(device="cpu")
    mem.pool.append(Candidate(vec=torch.randn(8), source="memory",
                              confidence=1.0, meta={"tag": "heard:hi",
                                                    "importance": 1.0}))
    mem.pool.append(Candidate(vec=torch.randn(8), source="confab",
                              confidence=0.5,
                              meta={"tag": "confabulated", "importance": 0.4}))
    recent = mem.recent(10)
    assert len(recent) == 2
    false = [r for r in recent if r["false"]]
    assert len(false) == 1 and false[0]["tag"] == "confabulated"
    assert false[0]["confidence"] == 0.5
