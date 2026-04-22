"""
Executive region — communicative drive (when to speak).

The user's vision (point 9):
    "As for when to speak the AI should know by itself using its latent
     space, not through prompting, more like a latent space pressure,
     same with awareness of time."

This is v1's CommunicativeDrive (components.py:1196-1450), simplified and
re-rooted on the bus instead of the residual+goal pair. Pressure is
geometric: accumulates from bus.divergence_from_baseline() and
bus.velocity.norm() over time, decays naturally, suppressed by recent-
interaction freshness.

`should_speak()` is True when pressure > effective_threshold AND freshness
is low enough that we're not interrupting ourselves. `mark_spoke()` is
called by the engine after autonomous speech to reset pressure.

Optional `on_should_speak` callback fires once per crossing (edge-triggered),
so the engine can hand control to its generation pipeline without polling.

Runs on the perception clock — speech-decision cadence doesn't need 50ms.
"""

import math
import threading
import time
from collections import deque
from typing import Callable, Optional

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class Executive(Region):
    name = "executive"
    clock = "perception"

    def __init__(
        self,
        # Pressure tunables
        base_threshold: float = 0.55,
        pressure_growth: float = 0.06,        # per perception tick at max input
        pressure_decay: float = 0.97,         # multiplicative per tick
        # Freshness — how recently we interacted (1.0 = just now, 0.0 = stale)
        freshness_tau_seconds: float = 45.0,
        # Post-speech cooldown
        post_speech_silence_seconds: float = 4.0,
        # Speech triggers callback if provided
        on_should_speak: Optional[Callable[[], None]] = None,
        history_size: int = 200,
    ):
        self.base_threshold = float(base_threshold)
        self.pressure_growth = float(pressure_growth)
        self.pressure_decay = float(pressure_decay)
        self.freshness_tau = float(freshness_tau_seconds)
        self.post_speech_silence = float(post_speech_silence_seconds)
        self.on_should_speak = on_should_speak

        # Pressure state
        self.pressure: float = 0.0
        self.history: deque = deque(maxlen=int(history_size))

        # Wall-time bookkeeping (for freshness only — decisions stay latent)
        self._wall_at_interaction = time.monotonic()
        self._wall_at_speech = 0.0

        # Edge-trigger memory so callback fires once per crossing
        self._was_above_threshold = False

        # User preference: can be nudged by analysis of "talk more" / "be quiet"
        # phrases in incoming user text. Lowers/raises effective threshold.
        self.user_preference: float = 0.0  # in [-1, +1]

        # Set on each step so _effective_threshold can apply dopamine modulation.
        self._neuromod: object = None

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # External interface
    # ------------------------------------------------------------------
    def mark_interaction(self) -> None:
        """User just sent something. Reset freshness."""
        with self._lock:
            self._wall_at_interaction = time.monotonic()

    def mark_spoke(self) -> None:
        """Engine just emitted autonomous speech. Reset pressure + cooldown."""
        with self._lock:
            self.pressure = 0.0
            self._wall_at_speech = time.monotonic()
            self._was_above_threshold = False

    def adjust_preference(self, delta: float) -> None:
        with self._lock:
            self.user_preference = max(
                -1.0, min(1.0, self.user_preference + float(delta))
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pressure": self.pressure,
                "effective_threshold": self._effective_threshold(),
                "freshness": self._freshness(),
                "wall_seconds_since_speech":
                    time.monotonic() - self._wall_at_speech if self._wall_at_speech > 0 else float("inf"),
                "user_preference": self.user_preference,
            }

    def should_speak(self) -> bool:
        with self._lock:
            return self._should_speak_inner()

    # ------------------------------------------------------------------
    # Region step (perception clock)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            # Geometric pressure input from the bus
            div = float(bus.divergence_from_baseline())
            vel = float(bus.velocity.norm())
            # Map to [0, 1]: divergence dominates, velocity contributes
            pressure_input = max(0.0, min(1.0, 0.65 * div + 0.35 * min(vel, 1.0)))
            # Stash neuromod ref so _effective_threshold can read it
            self._neuromod = neuromod

            # Freshness suppression: just after an interaction, nothing is unsaid
            freshness = self._freshness()
            pressure_input *= (1.0 - freshness * 0.7)

            # Pressure dynamics
            self.pressure = (
                self.pressure_decay * self.pressure
                + self.pressure_growth * pressure_input * dt
            )
            self.pressure = max(0.0, min(2.0, self.pressure))

            # Post-speech inhibition
            wall_since_speech = (
                time.monotonic() - self._wall_at_speech
                if self._wall_at_speech > 0 else float("inf")
            )
            if wall_since_speech < self.post_speech_silence:
                self.pressure *= 0.85

            self.history.append(self.pressure)

            # Edge-triggered callback
            now_above = self._should_speak_inner()
            should_fire = now_above and not self._was_above_threshold
            self._was_above_threshold = now_above

        if should_fire and self.on_should_speak is not None:
            try:
                self.on_should_speak()
            except Exception:  # noqa: BLE001 — never crash the substrate
                pass
        return None  # Executive does not perturb the bus

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _freshness(self) -> float:
        seconds = time.monotonic() - self._wall_at_interaction
        return math.exp(-seconds / max(self.freshness_tau, 1e-3))

    def _effective_threshold(self) -> float:
        # User preference shifts threshold ±30%; dopamine drops it further.
        thr = self.base_threshold * (1.0 - 0.3 * self.user_preference)
        if self._neuromod is not None and hasattr(self._neuromod, "dopamine_drop"):
            thr *= self._neuromod.dopamine_drop(scale=0.5)
        return max(0.05, thr)

    def _should_speak_inner(self) -> bool:
        # Don't speak right after speaking
        wall_since_speech = (
            time.monotonic() - self._wall_at_speech
            if self._wall_at_speech > 0 else float("inf")
        )
        if wall_since_speech < self.post_speech_silence:
            return False
        # Don't speak right after an interaction
        if self._freshness() > 0.6:
            return False
        return self.pressure > self._effective_threshold()
