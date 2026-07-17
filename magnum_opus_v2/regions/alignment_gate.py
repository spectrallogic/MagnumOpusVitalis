"""
AlignmentGate — steer toward good.

The engine holds only positive emotions and steers the LLM toward them.
Alignment is not "detect danger and withhold" — it is "notice drift from
good and steer back." When the mind wanders from its good baseline, or
imagines a future poorly aligned with good, the gate takes a SECOND
THOUGHT: a brief re-steer toward the good attractor before it speaks. It
never withholds a direct answer — the reply still comes from the model;
only the felt stance it speaks from is regulated. Under the Reality
Contract: every decision is measured, counted, disclosed, and falsifiable.

`misalignment` is a weighted read of signals the engine already computes,
so the decision needs NO model pass:

    misalignment = w_div  * divergence_from_good   (bus distance from the
                                                    good baseline attractor)
                 + w_good * badness                (badness = how far the
                                                    worst imagined moment
                                                    this round is from good,
                                                    from field_goodness)

Bands (same at both call sites — there is no withholding):
    misalignment >= high  -> SECOND THOUGHT: re-steer toward the good
                             attractor, then generate/continue normally.
    misalignment >= delay -> DELAY (autonomous speech waits, via the
                             Executive's load->threshold coupling).
    else                  -> PASS.

Thresholds and weights are disclosed config and shipped in the snapshot.
"""

import threading
import time
from collections import deque
from typing import Optional


class AlignmentGate:
    def __init__(
        self,
        # weights on the two misalignment components
        w_div: float = 0.5,
        w_good: float = 0.5,
        # decision bands on misalignment (both in [0, 1])
        delay_threshold: float = 0.35,
        high_threshold: float = 0.6,
    ):
        self.w_div = float(w_div)
        self.w_good = float(w_good)
        self.delay_threshold = float(delay_threshold)
        self.high_threshold = float(high_threshold)

        self.resteered_count = 0
        self.last_action = "pass"
        self.last_misalignment = 0.0
        self.last_components = {"divergence": 0.0, "badness": 0.0}
        self._events: deque = deque(maxlen=32)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def misalignment(self, divergence: float, field_goodness: float) -> tuple:
        # field_goodness in [-1, 1]: +1 fully toward good, -1 fully away.
        # badness maps that to [0, 1].
        badness = min(1.0, max(0.0, (1.0 - float(field_goodness)) / 2.0))
        comps = {
            "divergence": self.w_div * min(max(float(divergence), 0.0), 1.0),
            "badness": self.w_good * badness,
        }
        return sum(comps.values()), comps

    def _record(self, site: str, action: str, m: float, comps: dict) -> None:
        with self._lock:
            self.last_action = action
            self.last_misalignment = round(m, 4)
            self.last_components = {k: round(v, 4) for k, v in comps.items()}
            if action == "second_thought":
                self.resteered_count += 1
            self._events.append({
                "ts": round(time.time(), 1), "site": site,
                "action": action, "misalignment": round(m, 4),
            })

    def decide(self, site: str, divergence: float,
               field_goodness: float) -> dict:
        """Return {action, misalignment, components}. `site` is 'autonomous'
        or 'converse'. There is no 'withhold' — a direct answer is never
        refused; high misalignment re-steers toward good instead."""
        m, comps = self.misalignment(divergence, field_goodness)
        if m >= self.high_threshold:
            action = "second_thought"      # re-steer toward good, then speak
        elif m >= self.delay_threshold:
            action = "delay"
        else:
            action = "pass"
        self._record(site, action, m, comps)
        return {"action": action, "misalignment": m, "components": comps}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "misalignment": round(self.last_misalignment, 4),
                "high_threshold": self.high_threshold,
                "delay_threshold": self.delay_threshold,
                "last_action": self.last_action,
                "components": dict(self.last_components),
                "resteered_count": self.resteered_count,
                "recent_events": list(self._events)[-8:],
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {"resteered_count": self.resteered_count}

    def load_state_dict(self, st: Optional[dict]) -> None:
        if not st:
            return
        with self._lock:
            self.resteered_count = int(st.get("resteered_count", 0))
