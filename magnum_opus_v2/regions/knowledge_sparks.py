"""
KnowledgeSparks region — periodic emotion-biased token embedding intrusions.

Distinct from TokenEmbeddingSampler (which is part of the SubconsciousStack,
ambient): KnowledgeSparks runs on the expensive clock and probabilistically
"fires" a transient spark that decays over many flow ticks. It pushes a
specific concept (token embedding) into the bus directly, not through the
subconscious filter chain.

Think of it as: TokenEmbeddingSampler = ambient stream of word-shaped noise;
KnowledgeSparks = the occasional sudden "this concept just struck me."

Mechanism per expensive tick:
  1. With probability `fire_probability`, sample a token embedding from
     the model's vocab biased by current emotion.
  2. Set it as the active spark with magnitude `spark_strength`.
  3. Each subsequent flow tick (via Region.step run from flow clock by
     proxying), the spark contributes to the bus and decays by `decay`.

Implementation note: this region runs on the expensive clock for the
firing decision, but to inject the spark continuously into bus.velocity
it would need a flow-clock companion. Cleanest: spark fires by calling
bus.add_perturbation(vec * strength) once, then decays via successive
applies on subsequent flow ticks. We use a small flow-clock helper
internally.
"""

import random
import threading
from typing import Optional

import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.regions.limbic import Limbic


class KnowledgeSparks(Region):
    """
    Single Region with internal sub-step: when expensive clock fires, decide
    to spark; on every flow tick, emit the current decaying spark vector.

    To simplify the multi-clock setup we register two Region objects against
    different clocks (a "fire" half and a "decay" half) — both share state
    via a bound _SparkState object.
    """
    name = "knowledge_sparks"
    clock = "flow"  # the .step() that emits the perturbation runs on flow

    def __init__(
        self,
        model,
        limbic: Limbic,
        device: str = "cpu",
        fire_probability: float = 0.05,        # per expensive tick (~1.5s) ⇒ avg every ~30s
        spark_strength: float = 0.6,
        decay_per_flow_tick: float = 0.93,    # ~0.05^N over many ticks; lasts ~1s
        candidate_pool: int = 64,
    ):
        self.device = device
        self.fire_probability = float(fire_probability)
        self.spark_strength = float(spark_strength)
        self.decay = float(decay_per_flow_tick)
        self.candidate_pool = int(candidate_pool)
        self.limbic = limbic

        # Cache embedding matrix (same approach as TokenEmbeddingSampler)
        self.W = model.get_input_embeddings().weight.detach().to(device).float()

        # Live spark state
        self._spark_vec: Optional[torch.Tensor] = None
        self._fires_total: int = 0
        self._lock = threading.Lock()

    # --- companion expensive-clock region ---
    def fire_companion(self) -> "KnowledgeSparksFire":
        return KnowledgeSparksFire(self)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self._spark_vec is not None,
                "spark_norm": float(self._spark_vec.norm()) if self._spark_vec is not None else 0.0,
                "fires_total": self._fires_total,
            }

    # ------------------------------------------------------------------
    # Flow-clock step: emit current spark, then decay it.
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            if self._spark_vec is None:
                return None
            out = self._spark_vec.to(bus.device).clone()
            self._spark_vec = self._spark_vec * self.decay
            if self._spark_vec.norm() < 1e-3:
                self._spark_vec = None
        return out

    # ------------------------------------------------------------------
    # Internal: try to fire a new spark (called from companion)
    # ------------------------------------------------------------------
    def _maybe_fire(self) -> bool:
        if random.random() > self.fire_probability:
            return False
        # Bias by current emotion
        blend = self.limbic.snapshot()["blended"]
        bias = self._build_emotion_bias(blend)
        # Sample candidate pool, score by alignment, pick one
        vocab = self.W.shape[0]
        ids = torch.randint(0, vocab, (self.candidate_pool,), device=self.device)
        embs = self.W[ids]  # (pool, hidden)
        if bias is not None and bias.norm() > 1e-6:
            b = (bias / (bias.norm() + 1e-8)).to(self.device)
            scores = embs @ b
            probs = F.softmax(scores * 4.0, dim=0)
            pick = torch.multinomial(probs, 1).item()
        else:
            pick = int(torch.randint(0, self.candidate_pool, (1,)).item())
        v = embs[pick]
        v = v / (v.norm() + 1e-8) * self.spark_strength
        with self._lock:
            self._spark_vec = v.detach().clone()
            self._fires_total += 1
        return True

    def _build_emotion_bias(self, blend: dict) -> Optional[torch.Tensor]:
        # Use Limbic's emotion vectors to build a bias direction
        accum = None
        for n, w in blend.items():
            v = self.limbic._emo_vecs.get(n)  # noqa: SLF001 — internal cache
            if v is None or w == 0.0:
                continue
            term = float(w) * v
            accum = term if accum is None else accum + term
        return accum


class KnowledgeSparksFire(Region):
    """Expensive-clock companion that calls _maybe_fire on the parent."""
    name = "knowledge_sparks_fire"
    clock = "expensive"

    def __init__(self, parent: KnowledgeSparks):
        self._parent = parent

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        try:
            self._parent._maybe_fire()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        return None
