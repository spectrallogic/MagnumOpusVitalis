"""
Consolidation — the sleep-work of the mind. Slow clock (~30s).

Human memory doesn't persist by storage; it persists by REHEARSAL —
important experiences are replayed, replay strengthens abstraction, and
repeated abstraction becomes disposition. This region closes that loop:

  REPLAY   — each slow tick, the highest-importance memory traces are
             re-observed by the AbstractionLadder. What mattered gets
             taught again; what didn't fades (importance decay is already
             running on the perception clock).

  DISTILL  — every few ticks, the most-lived-in concept at the ladder's
             deepest unlocked level is installed as a bus attractor: a
             standing pull on the substrate. Repeated experience literally
             becomes a place the mind tends to return to. Old dispositions
             rotate out via the bus's attractor cap.

No text, no gradients — vectors teaching vectors.
"""

from typing import Optional

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class Consolidation(Region):
    name = "consolidation"
    clock = "slow"

    def __init__(
        self,
        memory,                     # Memory region — replay source
        abstraction,                # AbstractionLadder — replay target
        replay_count: int = 3,      # traces re-taught per slow tick
        distill_every: int = 4,     # slow ticks between disposition installs
        attractor_weight: float = 0.25,
        min_importance: float = 0.4,   # below this a trace isn't worth rehearsing
    ):
        self.memory = memory
        self.abstraction = abstraction
        self.replay_count = int(replay_count)
        self.distill_every = int(distill_every)
        self.attractor_weight = float(attractor_weight)
        self.min_importance = float(min_importance)

        self._ticks = 0
        self.last_replayed = 0
        self.distilled_total = 0

    def snapshot(self) -> dict:
        return {
            "last_replayed": self.last_replayed,
            "distilled_total": self.distilled_total,
        }

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        # --- REPLAY: rehearse what still matters
        with self.memory._lock:  # noqa: SLF001 — shared pool by design
            # EPISTEMIC GUARD (ADR-002): only OBSERVED traces may teach
            # the ladder or distill into bus attractors. Confabulations
            # kept flowing here unmarked for a long time — a false
            # memory could become a permanent disposition (Era-6 audit).
            scored = [
                (float((c.meta or {}).get("importance", 0.0)), c.vec)
                for c in self.memory.pool
                if (c.meta or {}).get("epistemic", "observed") == "observed"
                and c.confidence >= 1.0
            ]
        scored.sort(key=lambda t: -t[0])
        replayed = 0
        for imp, vec in scored[: self.replay_count]:
            if imp < self.min_importance:
                break
            try:
                self.abstraction.observe_external(vec, neuromod)
                replayed += 1
            except Exception:  # noqa: BLE001 — never crash the substrate
                continue
        self.last_replayed = replayed

        # --- DISTILL: the most-lived concept becomes a disposition
        self._ticks += 1
        if self._ticks % self.distill_every == 0:
            concept = self.abstraction.dominant_concept()
            if concept is not None and concept.norm() > 1e-6:
                bus.add_attractor(concept.to(bus.device), weight=self.attractor_weight)
                self.distilled_total += 1
        return None
