"""
Memory region — episodic latent traces + false-memory confabulation.

Two clocks:

  perception (~200ms): when the bus has been "interesting" since last capture
                       (high divergence, high velocity, or just-stimulated),
                       capture bus.state as a memory trace. Decay importance
                       of all stored memories.

  slow (~30s): generate false memories by interpolating two real memories
               with noise. Mark with confidence < 1.0. They live in the
               same pool as real memories so the subconscious can't
               distinguish them — that's the point.

The region holds a `pool: List[Candidate]` that MemorySampler reads from
directly. Capacity-bounded (drops lowest-importance entries when full).

Memory does NOT emit a perturbation directly to the bus — its influence
is indirect, through the SubconsciousStack's MemorySampler. That keeps
memory's effect on the substrate emergent rather than imposed.

Run on perception clock; we add a separate Region instance for slow-clock
confabulation so each clock has one well-defined responsibility.
"""

import threading
import time
from typing import List, Optional

import numpy as np
import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.regions.subconscious import Candidate, MemorySampler


class Memory(Region):
    """Captures bus traces and decays importance. Perception clock."""

    name = "memory"
    clock = "perception"

    def __init__(
        self,
        device: str = "cpu",
        capacity: int = 200,
        # Capture if velocity OR divergence exceeds these (since last capture).
        capture_velocity_threshold: float = 0.4,
        capture_divergence_threshold: float = 0.15,
        # Importance decays per tick (perception ~200ms; over 30s = 150 ticks).
        importance_decay: float = 0.005,
        # Cooldown — don't capture more than once per N seconds.
        capture_cooldown_seconds: float = 1.0,
    ):
        self.device = device
        self.capacity = int(capacity)
        self.cap_vel = float(capture_velocity_threshold)
        self.cap_div = float(capture_divergence_threshold)
        self.importance_decay = float(importance_decay)
        self.cooldown = float(capture_cooldown_seconds)

        # The pool MemorySampler reads from. Live reference — appended in place.
        self.pool: List[Candidate] = []
        # Parallel arrays (importance/age) — kept on Candidate.meta for simplicity.

        self._last_capture_wall = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # External interface — engine calls this when a user message arrives,
    # or anywhere a "salience event" should write a memory regardless of
    # whether the threshold is met.
    # ------------------------------------------------------------------
    def force_capture(self, bus: LatentBus, importance: float = 1.0,
                      tag: Optional[str] = None) -> None:
        with self._lock:
            self._capture(bus.state, importance=importance, tag=tag)

    def make_sampler(self) -> MemorySampler:
        """Return a MemorySampler bound to this pool. SubconsciousStack uses it."""
        return MemorySampler(self.pool, device=self.device)

    def snapshot(self) -> dict:
        with self._lock:
            n = len(self.pool)
            if n == 0:
                return {"size": 0, "avg_importance": 0.0, "n_false": 0}
            imps = [c.meta["importance"] for c in self.pool if c.meta]
            n_false = sum(1 for c in self.pool if c.confidence < 1.0)
            return {
                "size": n,
                "avg_importance": float(np.mean(imps)) if imps else 0.0,
                "max_importance": float(max(imps)) if imps else 0.0,
                "n_false": n_false,
            }

    # ------------------------------------------------------------------
    # Region step (perception clock)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            self._decay_importance()
            now = time.monotonic()
            if now - self._last_capture_wall >= self.cooldown:
                vel_n = float(bus.velocity.norm())
                div = float(bus.divergence_from_baseline())
                if vel_n >= self.cap_vel or div >= self.cap_div:
                    importance = max(min(vel_n + div, 2.0), 0.1)
                    self._capture(bus.state, importance=importance, tag="auto")
                    self._last_capture_wall = now
        return None  # Memory does not perturb the bus directly

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _capture(self, state: torch.Tensor, importance: float,
                 tag: Optional[str]) -> None:
        v = state.detach().clone().to(self.device)
        c = Candidate(
            vec=v, source="memory", confidence=1.0,
            meta={"importance": float(importance), "age": 0,
                  "tag": tag, "false": False},
        )
        self.pool.append(c)
        self._evict_if_over_capacity()

    def _decay_importance(self) -> None:
        for c in self.pool:
            if not c.meta:
                continue
            c.meta["importance"] = max(
                0.0, c.meta["importance"] - self.importance_decay
            )
            c.meta["age"] = c.meta.get("age", 0) + 1

    def _evict_if_over_capacity(self) -> None:
        if len(self.pool) <= self.capacity:
            return
        # Drop the lowest-importance entries until at capacity.
        self.pool.sort(key=lambda c: (c.meta or {}).get("importance", 0.0))
        excess = len(self.pool) - self.capacity
        del self.pool[:excess]


class FalseMemoryConfabulator(Region):
    """
    Slow-clock companion to Memory. Periodically synthesizes a false memory
    by interpolating two random real memories and adding noise. Marks with
    confidence < 1.0. Lives in the same pool the subconscious samples from.

    Bound at construction to a Memory instance so it shares the pool.
    """

    name = "memory_confabulator"
    clock = "slow"

    def __init__(
        self,
        memory: Memory,
        per_tick_count: int = 1,         # how many false memories per slow tick
        noise_strength: float = 0.15,    # how much noise to add to interpolated vec
        confidence_range: tuple = (0.4, 0.85),
        max_false_fraction: float = 0.25,  # cap false memories at this fraction of pool
    ):
        self._memory = memory
        self.per_tick_count = int(per_tick_count)
        self.noise_strength = float(noise_strength)
        self.confidence_range = confidence_range
        self.max_false_fraction = float(max_false_fraction)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._memory._lock:  # share the lock — we mutate the same pool
            pool = self._memory.pool
            # Need at least two real memories to interpolate.
            real = [c for c in pool if c.confidence >= 1.0]
            if len(real) < 2:
                return None
            # Cap on false-memory fraction
            n_false = sum(1 for c in pool if c.confidence < 1.0)
            if pool and (n_false / len(pool)) >= self.max_false_fraction:
                return None
            for _ in range(self.per_tick_count):
                a, b = np.random.choice(len(real), 2, replace=False)
                va = real[a].vec.float()
                vb = real[b].vec.float()
                alpha = float(np.random.uniform(0.2, 0.8))
                interp = alpha * va + (1.0 - alpha) * vb
                noise = (
                    torch.randn_like(interp)
                    * self.noise_strength
                    / (interp.shape[-1] ** 0.5)
                    * float(interp.norm() + 1e-6)
                )
                vec = interp + noise
                conf = float(np.random.uniform(*self.confidence_range))
                pool.append(Candidate(
                    vec=vec, source="memory", confidence=conf,
                    meta={
                        "importance": 0.5 * (
                            (real[a].meta or {}).get("importance", 0.5)
                            + (real[b].meta or {}).get("importance", 0.5)
                        ),
                        "age": 0,
                        "tag": "confabulated",
                        "false": True,
                        "alpha": alpha,
                    },
                ))
            self._memory._evict_if_over_capacity()
        return None
