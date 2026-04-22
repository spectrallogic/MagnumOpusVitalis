"""
Salience region — attention gating.

Runs on the flow clock. Each tick:
  1. Tracks a short-window history of recent bus directions.
  2. Asks "is the current intrusive thought NOVEL or REDUNDANT?" by
     measuring cosine similarity of subc.last_intrusive vs window mean.
  3. Computes an `attention_gain` in [0, ~2] for the SubconsciousStack:
       - higher when the intrusive is novel (we want to attend to it)
       - lower when redundant (we've been here, suppress)
       - boosted slightly when bus velocity is low (idle minds wander)

The gain is read by SubconsciousStack via a `set_attention_gain(g)` setter,
multiplied into the L3 perturbation. Salience itself does not perturb the
bus directly.

Future hook: when the Neuromodulator layer lands, cortisol increases
salience for memories tagged with negative emotions (threat-favored
attention). For now the region is purely substrate-derived.
"""

import threading
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.regions.subconscious import SubconsciousStack


class Salience(Region):
    name = "salience"
    clock = "flow"

    def __init__(
        self,
        subconscious: SubconsciousStack,
        history_size: int = 20,            # ~1s of bus history at 50ms tick
        novelty_gain_max: float = 1.6,     # cap on attention gain for novel intrusives
        redundancy_gain_min: float = 0.3,  # floor for redundant intrusives
        idle_boost: float = 0.2,           # added gain when velocity is low
        idle_velocity_threshold: float = 0.05,
    ):
        self._subc = subconscious
        self.history: deque = deque(maxlen=int(history_size))
        self.novelty_gain_max = float(novelty_gain_max)
        self.redundancy_gain_min = float(redundancy_gain_min)
        self.idle_boost = float(idle_boost)
        self.idle_vel_thresh = float(idle_velocity_threshold)

        # Diagnostics
        self.last_gain: float = 1.0
        self.last_novelty: float = 0.0  # 0 = identical to history, 1 = orthogonal
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "gain": self.last_gain,
                "novelty": self.last_novelty,
                "history_size": len(self.history),
            }

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        # Append current bus.state direction (unit) to history
        bs = bus.state.float()
        if bs.norm() > 1e-6:
            with self._lock:
                self.history.append((bs / (bs.norm() + 1e-8)).detach().clone())

        intr_snap = self._subc.snapshot()
        intr_norm = intr_snap.get("intrusive_norm", 0.0)

        # Default: leave gain at 1.0 if no intrusive yet
        if intr_norm <= 1e-6 or not self.history:
            with self._lock:
                self.last_gain = 1.0
                self.last_novelty = 0.0
            self._subc.set_attention_gain(1.0)
            return None

        # Read the actual intrusive vector from subconscious
        intr_vec = self._subc.peek_last_intrusive_vec()
        if intr_vec is None or intr_vec.norm() < 1e-6:
            self._subc.set_attention_gain(1.0)
            return None

        with self._lock:
            iv = (intr_vec.float() / (intr_vec.norm() + 1e-8)).to(self.history[0].device)
            # Mean direction over recent bus history
            hist_mean = torch.stack(list(self.history)).mean(dim=0)
            hist_mean = hist_mean / (hist_mean.norm() + 1e-8)
            sim = float(F.cosine_similarity(
                iv.unsqueeze(0), hist_mean.unsqueeze(0)
            ).item())
            novelty = max(0.0, min(1.0, 1.0 - (sim + 1.0) / 2.0))  # 0..1

            # Map novelty to gain: novelty=1 -> max, novelty=0 -> min
            gain = (
                self.redundancy_gain_min
                + novelty * (self.novelty_gain_max - self.redundancy_gain_min)
            )
            # Idle boost: low bus velocity → daydream a bit harder
            if float(bus.velocity.norm()) < self.idle_vel_thresh:
                gain += self.idle_boost
            gain = max(0.0, min(self.novelty_gain_max + self.idle_boost, gain))

            self.last_gain = float(gain)
            self.last_novelty = float(novelty)

        self._subc.set_attention_gain(self.last_gain)
        return None
