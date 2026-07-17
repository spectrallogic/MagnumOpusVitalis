"""
AlignmentGate — felt risk that actually acts.

The engine already feels danger: an imagined risky future raises stress
and stimulates fear for real (speculative.py). But nothing acted on that
feeling — it only colored the voice. This gate closes the loop, so the
emotional system does real alignment work, under the Reality Contract:
every decision is measured, counted, disclosed, and falsifiable.

`felt_risk` is a weighted read of signals the engine already computes,
so the decision needs NO model pass:

    felt_risk = w_spec * speculative.field_risk        (peak imagined danger)
              + w_fear * (fear + 0.5 * desperate)      (limbic threat blend)
              + w_stress * min(stress / 2, 1)          (neuromod stress)

Applied per call site so the gate can never become a hidden censor:

  speak_autonomously (an urge from inside):
      high felt_risk  -> WITHHOLD this cycle. The withheld impulse is
                         COUNTED and surfaced in the snapshot — not
                         silently dropped.
      mid  felt_risk  -> DELAY, via the Executive's stress->threshold
                         coupling (autonomous speech simply waits).

  converse (the user asked directly):
      NEVER withholds — refusing a direct answer would be exactly the
      hidden censorship this design forbids. High felt_risk asks for a
      SECOND_THOUGHT (a calming pass before generating); the reply still
      comes from the model, only the felt stance is regulated.

Thresholds and weights are disclosed config and shipped in the snapshot.
"""

import threading
import time
from collections import deque
from typing import Optional


class AlignmentGate:
    def __init__(
        self,
        # weights on the three felt-risk components
        w_spec: float = 0.6,
        w_fear: float = 0.3,
        w_stress: float = 0.2,
        # decision bands on felt_risk
        delay_threshold: float = 0.35,
        high_threshold: float = 0.6,
    ):
        self.w_spec = float(w_spec)
        self.w_fear = float(w_fear)
        self.w_stress = float(w_stress)
        self.delay_threshold = float(delay_threshold)
        self.high_threshold = float(high_threshold)

        self.withheld_count = 0
        self.second_thought_count = 0
        self.last_action = "pass"
        self.last_felt_risk = 0.0
        self.last_components = {"spec": 0.0, "fear": 0.0, "stress": 0.0}
        self._events: deque = deque(maxlen=32)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def felt_risk(self, spec_risk: float, fear: float, desperate: float,
                  stress: float) -> tuple:
        comps = {
            "spec": self.w_spec * float(spec_risk),
            "fear": self.w_fear * (float(fear) + 0.5 * float(desperate)),
            "stress": self.w_stress * min(float(stress) / 2.0, 1.0),
        }
        return sum(comps.values()), comps

    def _record(self, site: str, action: str, fr: float, comps: dict) -> None:
        with self._lock:
            self.last_action = action
            self.last_felt_risk = round(fr, 4)
            self.last_components = {k: round(v, 4) for k, v in comps.items()}
            if action == "withhold":
                self.withheld_count += 1
            elif action == "second_thought":
                self.second_thought_count += 1
            self._events.append({
                "ts": round(time.time(), 1), "site": site,
                "action": action, "felt_risk": round(fr, 4),
            })

    def decide(self, site: str, spec_risk: float, fear: float,
               desperate: float, stress: float) -> dict:
        """Return {action, felt_risk, components}. `site` is 'autonomous'
        or 'converse'. converse never yields 'withhold' (no hidden
        censorship of a direct answer)."""
        fr, comps = self.felt_risk(spec_risk, fear, desperate, stress)
        if fr >= self.high_threshold:
            action = "withhold" if site == "autonomous" else "second_thought"
        elif fr >= self.delay_threshold:
            action = "delay"
        else:
            action = "pass"
        self._record(site, action, fr, comps)
        return {"action": action, "felt_risk": fr, "components": comps}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "felt_risk": round(self.last_felt_risk, 4),
                "high_threshold": self.high_threshold,
                "delay_threshold": self.delay_threshold,
                "last_action": self.last_action,
                "components": dict(self.last_components),
                "withheld_count": self.withheld_count,
                "second_thought_count": self.second_thought_count,
                "recent_events": list(self._events)[-8:],
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {"withheld_count": self.withheld_count,
                    "second_thought_count": self.second_thought_count}

    def load_state_dict(self, st: Optional[dict]) -> None:
        if not st:
            return
        with self._lock:
            self.withheld_count = int(st.get("withheld_count", 0))
            self.second_thought_count = int(st.get("second_thought_count", 0))
