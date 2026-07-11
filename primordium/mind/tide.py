"""
Tide — global modulation, named by what it does, moved by what happened.

The audit that birthed this file found a four-channel layer inherited
from the engine: real mechanism, borrowed pharmacology. The names
implied a biology the code never implemented, and one channel (the
threat-reactivity one) modulated almost nothing here. Replaced by three
slow scalars named for their FUNCTION, each raised only by measured
events, each carrying a cause log the UI shows verbatim:

- arousal  — raised by surprise spikes, startle, situation shifts.
             Gates plasticity (Yerkes-Dodson), exploration temperature
             (typing, painting, babble), and bus noise.
- reward   — raised by drive-error reduction and milestones. Gates the
             live learning rate and loosens the urge to express.
- calm     — raised by sustained stability. Damps bus velocity and
             mood decay.

The drift baselines and rate below are authored homeostatic controller
constants, disclosed as such (the same standard the SleepController
meets) — nothing here claims to be discovered.

SUBSTRATE BOUNDARY: since the engine's own channels went functional
(the Second Reckoning), both minds speak one API — arousal/reward/calm
properties, bump(), and the gain helpers below. The engine has a fourth
channel, `stress`; the Tide deliberately does not. Its hasattr-guarded
consumers (stress_gain) simply skip, and bump("stress") is a safe
no-op — the honest hole where a channel that never earned its keep here
used to be. The hollowness test pins all of this.
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional

CHANNELS = ("arousal", "reward", "calm")


class Tide:
    def __init__(self, cfg=None):
        self.baselines = {
            "arousal": getattr(cfg, "tide_baseline_arousal", 0.1),
            "reward": getattr(cfg, "tide_baseline_reward", 0.2),
            "calm": getattr(cfg, "tide_baseline_calm", 0.5),
        }
        self.drift_speed = getattr(cfg, "tide_drift", 0.15)
        self._v: Dict[str, float] = dict(self.baselines)
        self._log: Dict[str, deque] = {c: deque(maxlen=12) for c in CHANNELS}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # the honest API: functional names, mandatory causes
    # ------------------------------------------------------------------
    @property
    def arousal(self) -> float:
        return self._v["arousal"]

    @property
    def reward(self) -> float:
        return self._v["reward"]

    @property
    def calm(self) -> float:
        return self._v["calm"]

    def raise_(self, channel: str, delta: float, cause: str) -> None:
        """Every rise names its cause. That is the whole point."""
        if channel not in CHANNELS:
            return
        with self._lock:
            self._v[channel] = min(2.0, max(0.0, self._v[channel] + delta))
            self._log[channel].append(
                {"t": round(time.time(), 1), "cause": str(cause)[:60],
                 "delta": round(float(delta), 4)})

    def drift_one_tick(self) -> None:
        with self._lock:
            for c in CHANNELS:
                self._v[c] += (self.baselines[c] - self._v[c]) \
                    * self.drift_speed

    def snapshot(self) -> dict:
        with self._lock:
            return {c: round(self._v[c], 4) for c in CHANNELS}

    def causes(self) -> Dict[str, List[dict]]:
        with self._lock:
            return {c: list(self._log[c]) for c in CHANNELS}

    def state(self) -> dict:
        with self._lock:
            return {c: self._v[c] for c in CHANNELS}

    def load_state(self, st: dict) -> None:
        with self._lock:
            for c in CHANNELS:
                if c in st:
                    self._v[c] = float(st[c])

    # ------------------------------------------------------------------
    # SUBSTRATE BOUNDARY — the engine speaks the same functional names
    # now. bump() is the engine's write verb (raise_ with a substrate
    # cause); unknown channels — the engine's `stress` — are safe no-ops.
    # No stress channel, no stress_gain: the honest hole.
    # ------------------------------------------------------------------
    def bump(self, name: str, delta: float) -> None:
        self.raise_(name, delta, cause=f"substrate:{name}")

    def arousal_gain(self, scale: float) -> float:
        return 1.0 + self._v["arousal"] * scale

    def calm_damp(self, scale: float) -> float:
        return max(0.05, 1.0 - self._v["calm"] * scale * 0.5)

    def reward_boost(self, scale: float) -> float:
        return 1.0 + self._v["reward"] * scale

    def reward_drop(self, scale: float) -> float:
        return max(0.0, 1.0 - self._v["reward"] * scale)


class TideRegion:
    """Slow-clock keeper: drift toward the disclosed baselines, and let
    calm accrue only from MEASURED stillness — sustained low divergence
    and low velocity on the bus, the same signals the old updater used,
    minus the borrowed emotion-name coupling."""

    name = "tide"
    clock = "slow"

    def __init__(self, tide: Tide, cfg=None):
        self.tide = tide
        self.calm_bump = getattr(cfg, "tide_calm_bump", 0.3)
        self._still_streak = 0

    def step(self, bus, neuromod: object, dt: float) -> Optional[object]:
        self.tide.drift_one_tick()
        try:
            div = float(bus.divergence_from_baseline())
            vel = float(bus.velocity.norm())
        except Exception:  # noqa: BLE001
            return None
        if div < 0.1 and vel < 0.15:
            self._still_streak += 1
            if self._still_streak >= 2:
                self.tide.raise_("calm", self.calm_bump,
                                 cause=f"stillness x{self._still_streak} "
                                       f"(div {div:.2f}, vel {vel:.2f})")
        else:
            self._still_streak = 0
        return None
