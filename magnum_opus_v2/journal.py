"""
CognitionJournal — the engine's honest event feed.

A time-ordered ring of what the mind actually did, so a person can watch
it think as a STREAM instead of only a re-rendered snapshot. Modeled on
the organism's Pulse (primordium/events/pulse.py): every entry is emitted
at a real computation site, carrying real payload. Nothing decorative is
ever appended — the timeline UI renders these 1:1.

Kinds emitted (each at its real site):
  future_considered   — a scored imagined future (speculative.py)
  word_chosen         — the future/word that won a turn (engine.converse)
  emotion_snapshot    — a meaningful change in the emotional blend
  forecast_opened     — an imagined future became a due-dated bet (forecast)
  forecast_resolved   — that bet came due: hit / miss / expired (forecast)
  gate_fired          — the alignment gate acted (withhold/second_thought)
  reply_emitted       — a turn completed, grouping the events above it

The ring is in-memory per session; a fresh SSE connection starts at
latest_id() so the timeline shows the present, never an invented past.
"""

import threading
import time
from collections import deque
from typing import List


class CognitionJournal:
    def __init__(self, capacity: int = 1024):
        self.ring: deque = deque(maxlen=capacity)
        self._next_id = 1
        self._lock = threading.Lock()

    def emit(self, kind: str, turn: int = 0, **payload) -> None:
        with self._lock:
            self.ring.append({
                "id": self._next_id,
                "ts": round(time.time(), 3),
                "kind": kind,
                "turn": int(turn),
                "payload": payload,
            })
            self._next_id += 1

    def since(self, last_id: int, limit: int = 120) -> List[dict]:
        with self._lock:
            out = [e for e in self.ring if e["id"] > last_id]
        return out[-limit:]

    def latest_id(self) -> int:
        with self._lock:
            return self._next_id - 1
