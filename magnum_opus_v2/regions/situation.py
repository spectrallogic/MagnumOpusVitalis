"""
SituationModel — the Now.

The engine had a transcript tail and per-message percepts, but no
persistent representation of WHAT IS HAPPENING RIGHT NOW — the thing a
mind actually predicts futures from. This region holds it, in two forms:

  VECTOR    — a latent situation state (mid-layer space). Each message's
              percept either ASSIMILATES into it (more of the same
              situation) or, when it doesn't resonate with the current
              situation, triggers a SCENE SHIFT: the old now is largely
              replaced, and arousal spikes (orienting response).
              Decisions — recall, novelty, the bus pull — use this.

  NARRATIVE — one present-tense sentence the MODEL ITSELF writes after
              each turn ("The user is driving at night beside a cliff").
              This is not hardcoded language: it is the model rendering
              its own understanding, and it exists because imagining the
              world's future requires a world-framed seed. Speculation
              consumes it to predict world events and the user's next
              action, not just the next words of the conversation.

Situations PERSIST: between messages the Now stays, its confidence
decaying with staleness (a situation you heard about two minutes ago is
probably still true; one from an hour ago barely is). While confident,
the region gently pulls the bus toward the situation vector — the felt
pressure of being somewhere, aware of it.
"""

import threading
import time
from typing import Optional

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class SituationModel(Region):
    name = "situation"
    clock = "perception"

    def __init__(
        self,
        device: str = "cpu",
        assimilation: float = 0.30,      # EMA weight for same-situation updates
        shift_threshold: float = 0.35,   # cosine below this = scene change
        shift_carry: float = 0.25,       # how much of the old now survives a shift
        confidence_tau_seconds: float = 150.0,  # staleness half-feel
        pull_gain: float = 0.045,        # bus pull toward the situation
        narrative_blend: float = 0.5,    # weight of narrative embedding in vec
    ):
        self.device = device
        self.assimilation = float(assimilation)
        self.shift_threshold = float(shift_threshold)
        self.shift_carry = float(shift_carry)
        self.confidence_tau = float(confidence_tau_seconds)
        self.pull_gain = float(pull_gain)
        self.narrative_blend = float(narrative_blend)

        self.vec: Optional[torch.Tensor] = None
        self.narrative: Optional[str] = None
        self.last_sim: float = 1.0
        self.shift_count: int = 0
        self.just_shifted: bool = False
        self._updated_wall: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # External interface (engine calls these)
    # ------------------------------------------------------------------
    def observe_percept(self, h: torch.Tensor, neuromod: object = None) -> bool:
        """A new message's percept arrives. Assimilate or shift.
        Returns True if the scene changed."""
        if h is None:
            return False
        v = h.detach().float().to(self.device)
        if v.norm() < 1e-6:
            return False
        with self._lock:
            self._updated_wall = time.monotonic()
            if self.vec is None:
                self.vec = v.clone()
                self.last_sim = 1.0
                self.just_shifted = True
                return True
            sim = float(torch.dot(
                v / v.norm(), self.vec / (self.vec.norm() + 1e-8),
            ))
            self.last_sim = sim
            if sim < self.shift_threshold:
                # Scene change: the old now mostly dissolves.
                self.vec = self.shift_carry * self.vec + (1 - self.shift_carry) * v
                self.shift_count += 1
                self.just_shifted = True
                shifted = True
            else:
                self.vec = (1 - self.assimilation) * self.vec + self.assimilation * v
                self.just_shifted = False
                shifted = False
        # Orienting response: a changed world is arousing.
        if shifted and neuromod is not None and hasattr(neuromod, "bump"):
            neuromod.bump("arousal", 0.15)
        return shifted

    def set_narrative(self, text: Optional[str],
                      narrative_vec: Optional[torch.Tensor] = None) -> None:
        """The model's own one-sentence rendering of the now, refreshed
        after each turn. Its embedding refines the situation vector."""
        with self._lock:
            self.narrative = (text or "").strip() or None
            self._updated_wall = time.monotonic()
            if narrative_vec is not None and narrative_vec.norm() > 1e-6:
                nv = narrative_vec.detach().float().to(self.device)
                if self.vec is None:
                    self.vec = nv.clone()
                else:
                    b = self.narrative_blend
                    self.vec = (1 - b) * self.vec + b * nv

    def confidence(self) -> float:
        """How much the Now can still be trusted (staleness decay)."""
        if self.vec is None:
            return 0.0
        stale = time.monotonic() - self._updated_wall
        return float(torch.exp(torch.tensor(-stale / self.confidence_tau)))

    def current_vec(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self.vec.detach().clone() if self.vec is not None else None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "narrative": self.narrative,
                "confidence": round(self.confidence(), 3),
                "last_sim": round(self.last_sim, 3),
                "shifts": self.shift_count,
                "seconds_since_update": (
                    round(time.monotonic() - self._updated_wall, 1)
                    if self._updated_wall > 0 else None
                ),
            }

    # ------------------------------------------------------------------
    # Region step (perception clock): being situated
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            if self.vec is None:
                return None
            conf = self.confidence()
            if conf < 0.05:
                return None
            pull = (self.vec.to(bus.device) - bus.state) * self.pull_gain * conf
        return pull
