"""
SubconsciousStack — layered subconscious from the user's vision (point 4):

    "The subconscious is composed of multiple layers. The bottom layer is
     a sea of noise and previous knowledge represented as organized latent
     space with some noise infused, some random false memories could be
     present as well. As data flows continuously like a stream of latent
     space, it gets filtered per layer, each layer tries to relate it to
     the current situation. The top layer shares what it figured out and
     then reveals it to the thinking conscious flowing layers where it goes
     into thinking. The subconscious kind of acts like an intrusive thought
     generator."

Stack of four layers, each implementing the same transform shape:
    (input_from_below, bus_state, layer_internal_state) -> output_directions

  L0 — Sea: Generates raw substrate. Random noise + samples from
            CandidateSamplers (token embeddings, memory pool incl. false
            memories) biased lightly by current emotion. Has no awareness
            of bus state. Output: a list of candidate vectors (the stream).

  L1 — Associative: Filters L0 against current bus state via cosine
            similarity. Keeps the directions in L0 that resonate with the
            moment. Drops the rest.

  L2 — Relational: Takes L1 candidates and relates them to current
            emotional vector and recent thought residual (= bus.velocity
            as a proxy for "what's been on my mind"). Re-weights.
            Occasionally promotes a low-resonance candidate to give the
            stream genuine surprise.

  L3 — Emergent: Picks the strongest candidate (with some interpolation
            between top-2 if they cluster), normalizes, scales by the
            stack's strength, and emits as the perturbation toward the
            bus. This is the "intrusive thought" that the conscious flow
            (= the substrate + steering hook) feels.

Information flows L0 -> L1 -> L2 -> L3 -> bus once per flow tick.
Salience (added later) gates how strongly L3's output is admitted —
without Salience, full strength is admitted.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


# ---------------------------------------------------------------------------
# Candidate samplers — pluggable sources of latent material for L0
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    vec: torch.Tensor                # the latent direction
    source: str                      # "noise" | "tokens" | "memory" | ...
    confidence: float = 1.0          # for false memories: < 1.0
    meta: Optional[dict] = None      # arbitrary tag, e.g. token_id


class CandidateSampler:
    """Subclass and implement sample()."""
    name = "sampler"

    def sample(
        self, n: int, bias: Optional[torch.Tensor] = None
    ) -> List[Candidate]:
        raise NotImplementedError


class NoiseSampler(CandidateSampler):
    """Gaussian noise vectors. Always available, doesn't need a model."""
    name = "noise"

    def __init__(self, hidden_dim: int, device: str = "cpu", magnitude: float = 1.0):
        self.hidden_dim = hidden_dim
        self.device = device
        self.magnitude = float(magnitude)

    def sample(self, n: int, bias: Optional[torch.Tensor] = None) -> List[Candidate]:
        out: List[Candidate] = []
        for _ in range(n):
            v = torch.randn(self.hidden_dim, device=self.device)
            v = v / (v.norm() + 1e-8) * self.magnitude
            out.append(Candidate(vec=v, source="noise"))
        return out


class TokenEmbeddingSampler(CandidateSampler):
    """
    Samples embeddings from the model's input embedding matrix. With bias,
    softmax-weights the sampling toward tokens whose embedding aligns with
    the bias direction. Mirrors v1's KnowledgeSparks logic but exposed as
    a sampler rather than a region.
    """
    name = "tokens"

    def __init__(
        self,
        embedding_matrix: torch.Tensor,  # (vocab, hidden_dim)
        device: str = "cpu",
        candidate_pool: int = 64,
        min_alpha_chars: int = 3,
        forbidden_token_ids: Optional[List[int]] = None,
    ):
        self.W = embedding_matrix.detach().to(device).float()
        self.device = device
        self.candidate_pool = int(candidate_pool)
        self.min_alpha_chars = int(min_alpha_chars)
        self.forbidden = set(forbidden_token_ids or [])

    def sample(self, n: int, bias: Optional[torch.Tensor] = None) -> List[Candidate]:
        vocab = self.W.shape[0]
        out: List[Candidate] = []
        for _ in range(n):
            # Draw a candidate pool, score by bias alignment, sample one.
            ids = torch.randint(
                0, vocab, (self.candidate_pool,), device=self.device,
            )
            embs = self.W[ids]  # (pool, hidden)
            if bias is not None and bias.norm() > 1e-6:
                b = bias.to(self.device).float()
                b = b / (b.norm() + 1e-8)
                scores = embs @ b
                # Softmax with mild temperature for some randomness.
                probs = F.softmax(scores * 4.0, dim=0)
                pick = torch.multinomial(probs, 1).item()
            else:
                pick = int(torch.randint(0, self.candidate_pool, (1,)).item())
            v = embs[pick]
            v = v / (v.norm() + 1e-8)
            out.append(Candidate(
                vec=v, source="tokens",
                meta={"token_id": int(ids[pick].item())},
            ))
        return out


class MemorySampler(CandidateSampler):
    """
    Samples from a memory pool. Pool is populated by the Memory region;
    this sampler is read-only. Each item is (vec, confidence) so false
    memories (confidence < 1.0) participate transparently — the
    subconscious can't tell which is real.
    """
    name = "memory"

    def __init__(self, pool: List[Candidate], device: str = "cpu"):
        # pool is a live list reference — Memory region appends to it.
        self.pool = pool
        self.device = device

    def sample(self, n: int, bias: Optional[torch.Tensor] = None) -> List[Candidate]:
        if not self.pool:
            return []
        if bias is None or bias.norm() < 1e-6:
            # uniform random sample
            picks = np.random.choice(len(self.pool), size=min(n, len(self.pool)),
                                     replace=True)
            return [self.pool[i] for i in picks]
        # Score by alignment, sample weighted.
        b = bias.to(self.device).float()
        b = b / (b.norm() + 1e-8)
        scores = []
        for c in self.pool:
            v = c.vec.to(self.device).float()
            v = v / (v.norm() + 1e-8)
            scores.append(float(torch.dot(v, b)))
        scores = np.asarray(scores)
        # confidence-weighted scores (false memories slightly suppressed
        # but not excluded — that's the point of the sea: subconscious
        # doesn't know what's true).
        confs = np.asarray([c.confidence for c in self.pool])
        weighted = scores * (0.5 + 0.5 * confs)
        # Softmax sampling
        ex = np.exp(weighted - weighted.max())
        probs = ex / ex.sum()
        picks = np.random.choice(len(self.pool), size=n, replace=True, p=probs)
        return [self.pool[i] for i in picks]


# ---------------------------------------------------------------------------
# The four-layer subconscious stack
# ---------------------------------------------------------------------------

class SubconsciousStack(Region):
    """One Region that runs all four layers per flow tick."""

    name = "subconscious_stack"
    clock = "flow"

    def __init__(
        self,
        hidden_dim: int,
        device: str = "cpu",
        samplers: Optional[List[CandidateSampler]] = None,
        # L0 controls
        l0_per_sampler_count: int = 4,        # how many to draw from each sampler
        l0_emotion_bias_strength: float = 0.5,  # how much to bias L0 by current emotion
        # L1 controls
        l1_keep_top_k: int = 8,               # how many of the L0 sea survive L1
        # L2 controls
        l2_keep_top_k: int = 3,
        l2_surprise_probability: float = 0.15,  # chance to promote a low-resonance candidate
        # L3 controls
        l3_perturbation_strength: float = 0.4,
        l3_interpolate_top_2: bool = True,
        # Optional emotion vector dict for biasing L0 + L2
        emotion_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.samplers = list(samplers or [NoiseSampler(hidden_dim, device, magnitude=1.0)])

        self.l0_per_sampler_count = int(l0_per_sampler_count)
        self.l0_emotion_bias_strength = float(l0_emotion_bias_strength)
        self.l1_keep_top_k = int(l1_keep_top_k)
        self.l2_keep_top_k = int(l2_keep_top_k)
        self.l2_surprise_probability = float(l2_surprise_probability)
        self.l3_perturbation_strength = float(l3_perturbation_strength)
        self.l3_interpolate_top_2 = bool(l3_interpolate_top_2)

        # Store emotion vectors so L0 can bias by current emotion and L2
        # can compare to "what does this mean for me right now."
        self._emo_vecs: Dict[str, torch.Tensor] = {}
        if emotion_vectors:
            for n, v in emotion_vectors.items():
                if n.startswith("temporal_"):
                    continue
                self._emo_vecs[n] = v.to(device).float()

        # Live-fed by external code (Limbic snapshot, set by SubconsciousStack
        # owner before each tick if desired). Defaults to a uniform tilt.
        self._emotion_blend: Optional[Dict[str, float]] = None

        # Last-tick diagnostic for the dashboard / verification
        self.last_intrusive: Optional[Candidate] = None
        self.last_l1_count: int = 0
        self.last_l2_count: int = 0
        self.last_was_surprise: bool = False

        # External attention gain (Salience writes this; we multiply L3
        # perturbation by it). Defaults to 1.0 — no gating.
        self._attention_gain: float = 1.0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # External interface
    # ------------------------------------------------------------------
    def add_sampler(self, sampler: CandidateSampler) -> None:
        with self._lock:
            self.samplers.append(sampler)

    def set_emotion_blend(self, blend: Dict[str, float]) -> None:
        """
        Push the current Limbic blended-emotion dict in. The stack uses it
        to bias L0 sampling and to compute L2 relational scores.
        Called by the engine each tick (or whenever the blend changes).
        """
        with self._lock:
            self._emotion_blend = dict(blend)

    def set_attention_gain(self, gain: float) -> None:
        """Salience calls this to amplify or suppress the L3 perturbation."""
        with self._lock:
            self._attention_gain = float(gain)

    def peek_last_intrusive_vec(self) -> Optional[torch.Tensor]:
        """Salience reads the actual vector to compute novelty against history."""
        with self._lock:
            if self.last_intrusive is None:
                return None
            return self.last_intrusive.vec.detach().clone()

    def snapshot(self) -> dict:
        with self._lock:
            intr = self.last_intrusive
            return {
                "l1_count": self.last_l1_count,
                "l2_count": self.last_l2_count,
                "surprise": self.last_was_surprise,
                "intrusive_source": intr.source if intr is not None else None,
                "intrusive_norm": float(intr.vec.norm()) if intr is not None else 0.0,
                "intrusive_confidence": intr.confidence if intr is not None else 0.0,
                "intrusive_meta": intr.meta if intr is not None else None,
            }

    # ------------------------------------------------------------------
    # Region step — runs L0 -> L1 -> L2 -> L3 once per flow tick
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            emotion_bias = self._compute_emotion_bias()
            l0 = self._layer_0_sea(emotion_bias)
            if not l0:
                self.last_intrusive = None
                self.last_l1_count = 0
                self.last_l2_count = 0
                self.last_was_surprise = False
                return None

            l1 = self._layer_1_associative(l0, bus.state)
            self.last_l1_count = len(l1)

            l2, was_surprise = self._layer_2_relational(l1, emotion_bias, bus.velocity)
            self.last_l2_count = len(l2)
            self.last_was_surprise = was_surprise

            chosen = self._layer_3_emergent(l2)
            self.last_intrusive = chosen
            if chosen is None:
                return None

            v = chosen.vec.to(bus.device).float()
            v = v / (v.norm() + 1e-8) * self.l3_perturbation_strength * self._attention_gain
            return v

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------
    def _compute_emotion_bias(self) -> Optional[torch.Tensor]:
        """A direction in latent space that summarizes 'how I feel right now',
        derived from the live emotion blend × emotion vectors."""
        blend = self._emotion_blend
        if not blend or not self._emo_vecs:
            return None
        accum = None
        for n, w in blend.items():
            v = self._emo_vecs.get(n)
            if v is None or w == 0.0:
                continue
            term = float(w) * v
            accum = term if accum is None else accum + term
        if accum is None:
            return None
        return accum

    def _layer_0_sea(self, emotion_bias: Optional[torch.Tensor]) -> List[Candidate]:
        """Generate raw substrate from all samplers. Lightly biased."""
        # Bias is the emotion vector × strength; samplers may use it.
        bias = (
            emotion_bias * self.l0_emotion_bias_strength
            if emotion_bias is not None else None
        )
        out: List[Candidate] = []
        for sampler in self.samplers:
            try:
                got = sampler.sample(self.l0_per_sampler_count, bias=bias)
                out.extend(got)
            except Exception:  # noqa: BLE001 — never crash the substrate
                continue
        return out

    def _layer_1_associative(
        self, l0: List[Candidate], bus_state: torch.Tensor,
    ) -> List[Tuple[Candidate, float]]:
        """Cosine-filter against bus.state. Keep top_k by resonance."""
        if bus_state.norm() < 1e-6:
            # Bus is at zero — nothing to resonate against. Return uniform random subset.
            picks = np.random.permutation(len(l0))[: self.l1_keep_top_k]
            return [(l0[i], 0.0) for i in picks]
        bs = bus_state.float()
        bs = bs / (bs.norm() + 1e-8)
        scored: List[Tuple[Candidate, float]] = []
        for c in l0:
            v = c.vec.to(bs.device).float()
            v = v / (v.norm() + 1e-8)
            score = float(torch.dot(v, bs))
            scored.append((c, score))
        scored.sort(key=lambda kv: -kv[1])
        return scored[: self.l1_keep_top_k]

    def _layer_2_relational(
        self,
        l1: List[Tuple[Candidate, float]],
        emotion_bias: Optional[torch.Tensor],
        bus_velocity: torch.Tensor,
    ) -> Tuple[List[Tuple[Candidate, float]], bool]:
        """
        Re-weight L1 candidates by alignment with emotion bias AND bus
        velocity (a proxy for "what's been on my mind"). Then either keep
        the top_k by combined score, OR with surprise_probability promote
        one low-resonance candidate (the spontaneous-novelty mechanism).
        Returns (final_list, was_surprise).
        """
        if not l1:
            return [], False
        emo = emotion_bias
        if emo is not None:
            emo = emo / (emo.norm() + 1e-8)
        vel = bus_velocity if bus_velocity.norm() > 1e-6 else None
        if vel is not None:
            vel = vel.float() / (vel.norm() + 1e-8)

        rescored: List[Tuple[Candidate, float]] = []
        for cand, l1_score in l1:
            v = cand.vec.float()
            v = v / (v.norm() + 1e-8)
            score = l1_score * 0.5
            if emo is not None:
                score += 0.3 * float(torch.dot(v, emo.to(v.device)))
            if vel is not None:
                score += 0.2 * float(torch.dot(v, vel.to(v.device)))
            rescored.append((cand, score))
        rescored.sort(key=lambda kv: -kv[1])

        was_surprise = False
        if (
            self.l2_surprise_probability > 0
            and len(rescored) > self.l2_keep_top_k
            and np.random.random() < self.l2_surprise_probability
        ):
            # Pull one low-resonance (bottom half) candidate up to the top.
            lower_half = rescored[len(rescored) // 2:]
            surprise = lower_half[np.random.randint(len(lower_half))]
            top = rescored[: self.l2_keep_top_k - 1]
            top.append((surprise[0], surprise[1] + 1.0))  # promote
            top.sort(key=lambda kv: -kv[1])
            return top, True

        return rescored[: self.l2_keep_top_k], was_surprise

    def _layer_3_emergent(
        self, l2: List[Tuple[Candidate, float]],
    ) -> Optional[Candidate]:
        """Pick the strongest. Optionally interpolate top-2 if they cluster."""
        if not l2:
            return None
        top = l2[0]
        if not self.l3_interpolate_top_2 or len(l2) < 2:
            return top[0]
        # If top-2 are close in score, average their vectors.
        s1, s2 = l2[0][1], l2[1][1]
        if s1 > 1e-6 and (s1 - s2) / abs(s1) < 0.2:
            v1 = l2[0][0].vec.float() / (l2[0][0].vec.norm() + 1e-8)
            v2 = l2[1][0].vec.float() / (l2[1][0].vec.norm() + 1e-8)
            blended = (v1 + v2) / 2.0
            # Inherit source from the stronger one but mark blended in meta.
            return Candidate(
                vec=blended,
                source=l2[0][0].source + "+" + l2[1][0].source,
                confidence=min(l2[0][0].confidence, l2[1][0].confidence),
                meta={"interpolated": True},
            )
        return top[0]
