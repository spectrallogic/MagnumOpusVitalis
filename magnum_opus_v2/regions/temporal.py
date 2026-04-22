"""
Temporal region — subjective time perception.

Wraps v1's TemporalEngine but reads its signals from the bus directly
rather than from a sea of bookkeeping in the engine. Subjective time is
derived from:

  - residual_norm: bus.state.norm()  (how much steering accumulated)
  - emotional_distance: bus.divergence_from_baseline()
  - interaction_freshness: 1 - exp(-time_since_interaction / freshness_tau)
  - steps_since_interaction: bus.tick_count - tick_at_last_interaction

Runs on the perception clock (~200ms) since subjective-time updates don't
need to fire every 50ms. Returns a temporal-direction perturbation if the
profile has temporal_recency / temporal_urgency vectors.

`mark_interaction()` is called when a user message arrives, to reset the
"freshness" clock. Without that signal, freshness decays exponentially.
"""

import math
import threading
import time
from typing import Dict, Optional

import torch

from magnum_opus.components import InternalTimeSignals, TemporalEngine
from magnum_opus.config import EngineConfig

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class Temporal(Region):
    name = "temporal"
    clock = "perception"

    def __init__(
        self,
        temporal_vectors: Dict[str, torch.Tensor],  # subset of profile.vectors
        device: str = "cpu",
        steering_strength: float = 1.0,
        freshness_tau_seconds: float = 60.0,
        config: Optional[EngineConfig] = None,
    ):
        # Filter to only temporal_* vectors. If none, the region still runs
        # subjective-time bookkeeping but emits no perturbation.
        self._temporal_vecs: Dict[str, torch.Tensor] = {
            n: v.to(device).float()
            for n, v in temporal_vectors.items()
            if n.startswith("temporal_")
        }
        self._engine = TemporalEngine(
            temporal_vectors=self._temporal_vecs,
            config=config,
        )
        self.device = device
        self.steering_strength = float(steering_strength)
        self.freshness_tau = float(freshness_tau_seconds)

        self._tick_at_interaction = 0
        self._wall_at_interaction = time.monotonic()
        self._lock = threading.Lock()
        self._last_subjective: float = 0.0

    # ------------------------------------------------------------------
    # External interface
    # ------------------------------------------------------------------
    def mark_interaction(self, bus: LatentBus) -> None:
        """Called when a user message lands. Resets freshness reference."""
        with self._lock:
            signals = self._gather_signals(bus)
            self._engine.mark_interaction(signals)
            self._tick_at_interaction = bus.tick_count
            self._wall_at_interaction = time.monotonic()

    def subjective_elapsed(self, bus: LatentBus) -> float:
        with self._lock:
            return self._engine.subjective_elapsed(self._gather_signals(bus))

    def snapshot(self, bus: LatentBus) -> dict:
        with self._lock:
            sig = self._gather_signals(bus)
            return {
                "subjective_elapsed": self._engine.subjective_elapsed(sig),
                "pace": self._engine.pace(),
                "steps_since_interaction": sig.steps_since_interaction,
                "interaction_freshness": sig.interaction_freshness,
                "wall_seconds_since_interaction":
                    time.monotonic() - self._wall_at_interaction,
            }

    # ------------------------------------------------------------------
    # Region step
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            self._engine.tick()
            sig = self._gather_signals(bus)
            self._last_subjective = self._engine.subjective_elapsed(sig)
            if not self._temporal_vecs:
                return None
            steering = self._engine.compute_steering(sig)
        if steering is None:
            return None
        return (steering * self.steering_strength).to(bus.device)

    # ------------------------------------------------------------------
    # Internal: derive InternalTimeSignals from bus state
    # ------------------------------------------------------------------
    def _gather_signals(self, bus: LatentBus) -> InternalTimeSignals:
        seconds_since = time.monotonic() - self._wall_at_interaction
        freshness = math.exp(-seconds_since / max(self.freshness_tau, 1e-3))
        return InternalTimeSignals(
            residual_norm=float(bus.state.norm()),
            residual_norm_at_last_interaction=self._engine.residual_norm_at_interaction,
            emotional_distance=float(bus.divergence_from_baseline()),
            interaction_freshness=float(freshness),
            memory_avg_importance=0.0,
            memory_avg_importance_at_last_interaction=
                self._engine.importance_at_interaction,
            steps_since_interaction=int(bus.tick_count - self._tick_at_interaction),
        )
