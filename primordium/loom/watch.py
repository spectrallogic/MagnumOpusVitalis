"""
Watch — many cheap eyes, one deliberate look.

The Gaze is a single fovea. The Watch is the rest of the eyes: one tiny
deviation-watcher per stream (each sense's prediction error, each
drive's need, the familiarity of what the Reach returns), each keeping
running statistics of its own stream and raising its hand only when the
stream departs from ITS OWN history (z-score, after a warmup — a
newborn's watchers stay silent until they have a history to defy).

Consumers — the rule is mechanism, cause, consumer, and this organ has
three real ones:
  1. Cross-modal interrupt: a hard spike on a NON-visual stream while
     the fovea is zoomed in releases the Gaze back to the whole scene —
     hearing something strange makes you look up from what you were
     staring at.
  2. Salience-gated memory: a spike banks the moment in the Reach even
     when tick-level surprise and the steady write beat would not.
  3. The Pulse and HUD show every spike with its stream and z, verbatim.

No watcher ever fabricates a value: every stream it sees is a number
some other organ already computed for its own honest reasons.
"""

import threading
from typing import Dict, List

MIN_STD = 1e-6


class Watch:
    def __init__(self, cfg):
        self.cfg = cfg
        self._stats: Dict[str, list] = {}   # name -> [mean, var, n]
        self.last_z: Dict[str, float] = {}
        self.spikes_total = 0
        self._lock = threading.Lock()

    def observe(self, values: Dict[str, float]) -> List[dict]:
        """Update every watcher with its stream's new value; return the
        spikes (z beyond watch_spike_z, only after watch_min_n samples)."""
        cfg = self.cfg
        spikes: List[dict] = []
        with self._lock:
            for name, v in values.items():
                v = float(v)
                m, var, n = self._stats.get(name, [v, 1.0, 0])
                n = min(n + 1, 5000)
                m += (v - m) / n
                var += ((v - m) ** 2 - var) / n
                self._stats[name] = [m, var, n]
                z = (v - m) / max(var ** 0.5, MIN_STD)
                z = max(-999.0, min(999.0, z))   # a dead-constant stream
                self.last_z[name] = round(z, 3)  # jumping is a real spike,
                # but the number needn't be astronomical to say so
                if n >= cfg.watch_min_n and abs(z) >= cfg.watch_spike_z:
                    spikes.append({"stream": name, "z": round(z, 2),
                                   "value": round(v, 5)})
            self.spikes_total += len(spikes)
        return spikes

    def snapshot(self) -> dict:
        with self._lock:
            return {"z": dict(self.last_z),
                    "streams": len(self._stats),
                    "spikes": self.spikes_total}

    def state(self) -> dict:
        with self._lock:
            return {"stats": {k: list(v) for k, v in self._stats.items()},
                    "spikes": self.spikes_total}

    def load_state(self, st: dict) -> None:
        with self._lock:
            self._stats = {k: list(v)
                           for k, v in (st.get("stats") or {}).items()}
            self.spikes_total = int(st.get("spikes", 0))
