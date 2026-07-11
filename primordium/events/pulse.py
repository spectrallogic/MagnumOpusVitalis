"""
Pulse — the honest event feed.

Every record here is emitted at a real call site, carrying real metadata
(tensor shapes, loss values, counts). The data-block UI renders these
1:1 — a block on screen IS a computation that happened. There are no
decorative spawns, by contract; the test suite checks that every kind
observed in a live run corresponds to a real emitter.
"""

import threading
import time
from collections import deque
from typing import List, Optional

ZONES = ("WORLD", "SENSE", "CORE", "SELF", "EXPRESS", "KEEP", "SCAFFOLD")


class Pulse:
    def __init__(self, capacity: int = 512):
        self.ring: deque = deque(maxlen=capacity)
        self._next_id = 1
        self._lock = threading.Lock()

    def emit(self, kind: str, zone_from: str, zone_to: str,
             **meta) -> None:
        with self._lock:
            self.ring.append({
                "id": self._next_id,
                "ts": round(time.time(), 3),
                "kind": kind,
                "zone_from": zone_from,
                "zone_to": zone_to,
                "meta": meta,
            })
            self._next_id += 1

    def since(self, last_id: int, limit: int = 64) -> List[dict]:
        with self._lock:
            out = [e for e in self.ring if e["id"] > last_id]
        return out[-limit:]

    def latest_id(self) -> int:
        with self._lock:
            return self._next_id - 1
