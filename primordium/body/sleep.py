"""
SleepController — pressure, episodes, and the courtesy of not sleeping
mid-conversation.

Pressure integrates exhaustion (low energy), time awake, and learning
plateau. Sleep dims the eyes (gain, not amputation — stream continuity is
the point), silences the voice, stops live learning, multiplies replay,
and lets dreams run. A strong stimulus wakes it — groggy.
"""

import threading
import time


class SleepController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pressure = 0.0
        self.asleep = False
        self._woke_at = time.monotonic()
        self._slept_at = 0.0
        self.grogginess_until = 0.0
        self.episodes = 0
        self._lock = threading.Lock()

    def update(self, energy: float, plateau: float, presence: float,
               surprise: float, neuromod=None) -> None:
        """Called on the slow clock (~30s)."""
        now = time.monotonic()
        with self._lock:
            if not self.asleep:
                hours_awake = (now - self._woke_at) / 3600.0
                self.pressure += (0.035 * (1.0 - energy)
                                  + 0.012 * hours_awake
                                  + 0.02 * plateau)
                self.pressure = min(1.5, self.pressure)
                if (self.pressure > self.cfg.sleep_enter_pressure
                        and (presence < 0.2 or energy < 0.1)):
                    self.asleep = True
                    self.episodes += 1
                    self._slept_at = now
            else:
                self.pressure = max(0.0, self.pressure - 0.06)
                strong_stimulus = surprise > 3.0
                if self.pressure < self.cfg.sleep_exit_pressure or strong_stimulus:
                    self.asleep = False
                    self._woke_at = now
                    if strong_stimulus:
                        self.grogginess_until = now + 120.0
                        if neuromod is not None and hasattr(neuromod, "raise_"):
                            neuromod.raise_("arousal", 0.3,
                                            cause="startled awake")

    # ---- worker-facing flags ----
    def vision_gain(self) -> float:
        return 0.1 if self.asleep else 1.0

    def live_lr_gain(self) -> float:
        now = time.monotonic()
        if self.asleep:
            return 0.0
        if now < self.grogginess_until:
            return 0.5
        return 1.0

    def replay_multiplier(self) -> int:
        return self.cfg.sleep_replay_multiplier if self.asleep else 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "asleep": self.asleep,
                "pressure": round(self.pressure, 3),
                "episodes": self.episodes,
                "groggy": time.monotonic() < self.grogginess_until,
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {"pressure": self.pressure, "episodes": self.episodes}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            self.pressure = float(st.get("pressure", 0.0))
            self.episodes = int(st.get("episodes", 0))
