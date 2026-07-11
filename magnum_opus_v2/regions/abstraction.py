"""
AbstractionLadder — developmental coarse-to-fine concept learning.

Design: like an infant, learn sky/ground before clouds and rocks —
abstract scaffolds first, details only when they are earned.

The ladder is a stack of online clustering levels over the stream of lived
latent states (bus.state samples on the perception clock):

    level 0:  2 concepts   — sky / ground
    level 1:  4 concepts   — clouds / dirt ...
    level 2:  8 concepts
    level 3: 16 concepts

Levels UNLOCK developmentally: a deeper level only starts learning after
enough experience has accumulated at the coarser ones. A newborn engine
literally cannot perceive fine distinctions yet — the same input that an
"adult" engine sorts into 16 concepts lands in one of 2 buckets at birth.

Learning is online k-means with a decaying learning rate — no gradients,
no backprop, no dataset. It adapts to any LLM in minutes of runtime,
which is the "quick and simple training" constraint.

Effects on the substrate:
  - GROUNDING: a gentle pull toward the best-matching concept at the
    deepest unlocked level. Perception is shaped by what is understood.
  - CURIOSITY: novelty (distance to the nearest known concept) bumps
    reward and reads as the felt sense of "this is new".

Interpretability: each concept centroid is labeled with the vocabulary
token whose embedding lies nearest to it (approximate — centroids live in
mid-layer space, embeddings in input space — but consistently revealing
in practice, especially for tied-embedding models like GPT-2).
"""

import threading
import time
from typing import List, Optional

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region

STAGE_NAMES = ["newborn", "infant", "child", "adolescent", "adult"]


class _Level:
    def __init__(self, n_slots: int, unlock_at: int):
        self.n_slots = n_slots
        self.unlock_at = unlock_at          # total observations needed to unlock
        self.centroids: List[torch.Tensor] = []
        self.counts: List[int] = []
        self.updates = 0

    def nearest(self, x: torch.Tensor) -> int:
        best, best_sim = 0, -2.0
        xn = x / (x.norm() + 1e-8)
        for i, c in enumerate(self.centroids):
            sim = float(torch.dot(xn, c / (c.norm() + 1e-8)))
            if sim > best_sim:
                best, best_sim = i, sim
        return best

    def observe(self, x: torch.Tensor, base_lr: float) -> int:
        """Update the level with one observation. Returns the assigned slot."""
        if len(self.centroids) < self.n_slots:
            # Concept formation: the first experiences BECOME the concepts.
            self.centroids.append(x.detach().clone())
            self.counts.append(1)
            return len(self.centroids) - 1
        idx = self.nearest(x)
        n = self.counts[idx]
        lr = max(base_lr / (1.0 + 0.02 * n), 0.002)
        self.centroids[idx] = self.centroids[idx] + lr * (x - self.centroids[idx])
        self.counts[idx] = n + 1
        self.updates += 1
        return idx


class AbstractionLadder(Region):
    name = "abstraction_ladder"
    clock = "perception"

    def __init__(
        self,
        hidden_dim: int,
        device: str = "cpu",
        embedding_matrix: Optional[torch.Tensor] = None,  # (vocab, hidden) for labels
        tokenizer=None,
        level_sizes: Optional[List[int]] = None,          # default [2, 4, 8, 16]
        unlock_schedule: Optional[List[int]] = None,      # observations to unlock each level
        base_lr: float = 0.15,
        grounding_gain: float = 0.06,     # pull toward the active concept
        novelty_reward: float = 0.05,   # reward per unit novelty spike
        min_state_norm: float = 0.05,     # ignore an empty bus
        label_refresh_seconds: float = 10.0,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.tokenizer = tokenizer
        self.W = (
            embedding_matrix.detach().to(device).float()
            if embedding_matrix is not None else None
        )

        sizes = level_sizes or [2, 4, 8, 16]
        # Developmental schedule: ~1min, ~4min, ~12min of experience at 5Hz.
        schedule = unlock_schedule or [0, 300, 1200, 3600]
        self.levels = [_Level(s, u) for s, u in zip(sizes, schedule)]

        self.base_lr = float(base_lr)
        self.grounding_gain = float(grounding_gain)
        self.novelty_reward = float(novelty_reward)
        self.min_state_norm = float(min_state_norm)

        self.observations = 0
        self.current_path: List[int] = []   # concept index per unlocked level
        self.novelty: float = 0.0           # distance to nearest deep concept
        self.novelty_ema: float = 0.0

        self._labels: List[List[str]] = [["?"] * lvl.n_slots for lvl in self.levels]
        self._labels_at = 0.0
        self.label_refresh = float(label_refresh_seconds)
        self._Wn: Optional[torch.Tensor] = None  # cached normalized vocab

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Ingestion — shared by interoception (bus state, via step) and
    # exteroception (world content, via observe_external)
    # ------------------------------------------------------------------
    def _ingest(self, x: torch.Tensor, neuromod: object) -> Optional[torch.Tensor]:
        """One observation through the ladder. Returns the deepest matched
        centroid (or None). Caller supplies a normalized-device tensor."""
        with self._lock:
            self.observations += 1
            path: List[int] = []
            deepest_centroid: Optional[torch.Tensor] = None
            for lvl in self.levels:
                if self.observations < lvl.unlock_at:
                    break
                idx = lvl.observe(x, self.base_lr)
                path.append(idx)
                deepest_centroid = lvl.centroids[idx]
            self.current_path = path

            if deepest_centroid is not None:
                sim = float(torch.dot(
                    x / (x.norm() + 1e-8),
                    deepest_centroid / (deepest_centroid.norm() + 1e-8),
                ))
                self.novelty = max(0.0, 1.0 - sim)
                self.novelty_ema = 0.95 * self.novelty_ema + 0.05 * self.novelty

        # Novelty spike above the running baseline = felt curiosity.
        spike = self.novelty - self.novelty_ema
        if spike > 0.05 and neuromod is not None and hasattr(neuromod, "bump"):
            neuromod.bump("reward", self.novelty_reward * spike / 0.05)
        return deepest_centroid

    def observe_external(self, vec: torch.Tensor, neuromod: object = None) -> None:
        """Exteroception: learn from a world-content vector (the model's
        hidden-state reading of a message, a reply, a replayed memory) —
        not from the engine's own mood."""
        x = vec.detach().float().to(self.device)
        if x.norm() < self.min_state_norm:
            return
        self._ingest(x, neuromod)

    def dominant_concept(self) -> Optional[torch.Tensor]:
        """The most-lived-in concept at the deepest unlocked level — what
        Consolidation distills into a long-term disposition."""
        with self._lock:
            best = None
            for lvl in self.levels:
                if self.observations < lvl.unlock_at or not lvl.centroids:
                    break
                i = int(max(range(len(lvl.counts)), key=lambda k: lvl.counts[k]))
                best = lvl.centroids[i]
            return best.detach().clone() if best is not None else None

    # ------------------------------------------------------------------
    # Region step (perception clock, ~200ms) — interoception
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        x = bus.state.detach().float().to(self.device)
        if x.norm() < self.min_state_norm:
            return None

        deepest_centroid = self._ingest(x, neuromod)

        # Grounding: perception drifts toward the concept it was filed under.
        if deepest_centroid is None or self.grounding_gain <= 0:
            return None
        pull = deepest_centroid.to(bus.device) - bus.state
        return pull * self.grounding_gain

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def developmental_stage(self) -> str:
        unlocked = sum(1 for lvl in self.levels if self.observations >= lvl.unlock_at
                       and lvl.centroids)
        return STAGE_NAMES[min(unlocked, len(STAGE_NAMES) - 1)]

    def _refresh_labels(self, centroids_by_level) -> None:
        """Runs OUTSIDE the region lock, on copies: the vocab-scale
        matmuls here once ran under the same lock the perception-clock
        step() needs, so a dashboard poll on a big-vocab model could
        stall interoception for hundreds of ms (Era-4 audit finding).
        Labels are diagnostics; a benign string race is acceptable."""
        if self.W is None or self.tokenizer is None:
            return
        now = time.monotonic()
        if now - self._labels_at < self.label_refresh:
            return
        self._labels_at = now
        # Normalize the vocab once — recomputing this every refresh would
        # allocate a fresh ~1GB tensor on large-vocab models.
        if self._Wn is None:
            self._Wn = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-8)
        Wn = self._Wn
        for li, centroids in enumerate(centroids_by_level):
            for ci, c in enumerate(centroids):
                cn = (c / (c.norm() + 1e-8)).to(self.device)
                sims = Wn @ cn
                # Take the nearest *readable* token — vocabularies are full
                # of control bytes and fragments that label nothing. Prefer
                # ASCII words so multilingual vocabs stay legible.
                word = "·"
                top = torch.topk(sims, k=24).indices.tolist()
                for require_ascii in (True, False):
                    found = None
                    for tok in top:
                        try:
                            cand = self.tokenizer.decode([int(tok)]).strip()
                        except Exception:  # noqa: BLE001
                            continue
                        if sum(ch.isalpha() for ch in cand) < 2:
                            continue
                        if require_ascii and not cand.isascii():
                            continue
                        found = cand
                        break
                    if found is not None:
                        word = found
                        break
                self._labels[li][ci] = word

    def snapshot(self) -> dict:
        # cheap copies under the lock; the expensive labeling outside it
        with self._lock:
            centroids_by_level = [
                [c.detach().clone() for c in lvl.centroids]
                for lvl in self.levels
            ]
            levels = []
            for li, lvl in enumerate(self.levels):
                unlocked = self.observations >= lvl.unlock_at
                levels.append({
                    "slots": lvl.n_slots,
                    "formed": len(lvl.centroids),
                    "unlocked": unlocked,
                    "unlock_at": lvl.unlock_at,
                    "updates": lvl.updates,
                    "counts": list(lvl.counts),
                })
            head = {
                "stage": self.developmental_stage(),
                "observations": self.observations,
                "current_path": list(self.current_path),
                "novelty": round(self.novelty, 4),
                "novelty_ema": round(self.novelty_ema, 4),
            }
        self._refresh_labels(centroids_by_level)
        for li, lvl_meta in enumerate(levels):
            lvl_meta["labels"] = list(
                self._labels[li][: lvl_meta["formed"]])
        head["levels"] = levels
        return head
