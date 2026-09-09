"""Bounded recursive imagination, with fast replay of selected hypotheses.

The expensive worker explores a beam of sampled event continuations.
Children receive the parent's generated context and residual change.
Discounted path utility combines token-chain confidence with an authored
positive-affect proxy. Neither score establishes truth or ethical value.

The winner perturbs the bus; the upper subconscious layer receives tagged
hypotheses, and plausible alternatives linger in the penumbra. Model calls
are isolated from the speech-feedback tap until explicit selection.
The shared model lock serializes generation and imagination. Budgets are
cooperative; an in-flight forward can delay a waiting generation request.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.regions.subconscious import SubconsciousStack
from magnum_opus_v2.imagination import search_futures

POSITIVE_EMOTIONS = ("joy", "trust", "calm", "curious")

# How a future FEELS is scored purely in LATENT SPACE: the rollout's own
# mid-layer hidden states are projected onto the profile's emotion vectors
# (directions extracted from the model's own geometry) relative to the
# neutral baseline. No surface words, no authored good/bad lexicon — a
# future's valence is read from the model's own geometry, whatever
# vocabulary it happens to use.


class SpeculativeFutures(Region):
    """Expensive-clock region: imagines candidate futures, scores them,
    commits the winner, retains near-misses in the penumbra."""

    name = "speculative_futures"
    clock = "expensive"

    def __init__(
        self,
        model,
        tokenizer,
        hook,                       # SteeringHook
        subconscious: SubconsciousStack,
        emotion_vectors: Dict[str, torch.Tensor],
        baseline_projections: Optional[Dict[str, float]] = None,
        memory=None,                # Memory region (optional, for past-repeats seed)
        limbic=None,                # optional (unused now: no fear-injection)
        device: str = "cpu",
        model_lock: Optional[threading.Lock] = None,
        n_futures: int = 4,
        rollout_tokens: int = 14,            # imagined depth: futures are PHRASES
        rollout_budget_ms: float = 250.0,    # wall-clock cap on one rollout
        chained_continuation_tokens: int = 8,  # WORLD mode reads its trajectory further
        imagination_strength: float = 1.2,   # candidate offset added to bus.state
        winner_strength: float = 0.5,        # perturbation magnitude of chosen future
        plausibility_floor: float = 0.05,    # min utility to survive into penumbra
        penumbra_gain: float = 0.08,         # how loud the unattended futures are
        w_probability: float = 0.4,
        w_benefit: float = 0.35,
        max_depth: int = 2,
        branching_factor: int = 2,
        beam_width: int = 2,
        max_nodes: int = 12,
        max_total_tokens: int = 128,
        round_budget_ms: float = 500.0,
        discount: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook = hook
        self.subc = subconscious
        self.memory = memory
        self.device = device
        self.model_lock = model_lock or threading.Lock()

        # Where imagination happens: a callable returning recent conversation
        # token ids (1-D LongTensor) or None. With context, silent rollouts
        # literally contain the current situation — say "cliff" and the
        # threat future "falling" becomes discoverable. Without it (or before
        # the first exchange), imagination is contextless from BOS.
        self.context_provider = None
        # Optional callable returning the situation vector (the model's
        # hidden-state reading of the last user message) — seeded as a
        # candidate: "what does this situation itself become?"
        self.situation_provider = None
        # Optional callable returning the Now narrative (one sentence from
        # SituationModel). With it, imagination runs in three MODES:
        #   speech — continue the conversation (as before)
        #   world  — "<now> What happens next:"       → world events
        #   user   — "<now> The user will probably"   → the user's next act
        self.situation_text_provider = None

        # Never name a future after template scaffolding.
        try:
            self._special_ids = set(int(t) for t in (tokenizer.all_special_ids or []))
        except Exception:  # noqa: BLE001
            self._special_ids = set()

        self.limbic = limbic
        self.n_futures = int(n_futures)
        self.rollout_tokens = int(rollout_tokens)
        self.rollout_budget_ms = float(rollout_budget_ms)
        self.chained_continuation_tokens = int(chained_continuation_tokens)
        self.imagination_strength = float(imagination_strength)
        self.winner_strength = float(winner_strength)
        self.plausibility_floor = float(plausibility_floor)
        self.penumbra_gain = float(penumbra_gain)
        self.w_p = float(w_probability)
        self.w_b = float(w_benefit)
        self.max_depth = int(max_depth)
        self.branching_factor = int(branching_factor)
        self.beam_width = int(beam_width)
        self.max_nodes = int(max_nodes)
        self.max_total_tokens = int(max_total_tokens)
        self.round_budget_ms = float(round_budget_ms)
        self.discount = float(discount)
        self.last_search = {}
        self.rollout_failures = 0
        self.last_rollout_error = None
        self.last_attempt_tokens = 0

        # Full per-emotion vectors + neutral baseline — the latent affect
        # reader for imagined futures (same convention as perception).
        self._emo_vecs: Dict[str, torch.Tensor] = {
            n: v.detach().float().to(device)
            for n, v in emotion_vectors.items()
            if not n.startswith("temporal_")
        }
        self._base_proj: Dict[str, float] = dict(baseline_projections or {})

        # Shared with the flow-clock penumbra companion
        self.penumbra: List[dict] = []  # {"vec","weight","word","utility"}
        self._penumbra_lock = threading.Lock()

        # imagined futures become accountable predictions
        from magnum_opus_v2.forecast import ForecastLedger
        self.ledger = ForecastLedger()

        # optional cognition journal (wired by the engine); guarded so the
        # region works standalone in tests.
        self.journal = None

        # Diagnostics for dashboard
        self.last_futures: List[dict] = []
        self.rounds_total = 0
        self.skipped_busy = 0
        self.over_budget = 0            # rollouts truncated by the wall-clock guard
        self.rollout_tokens_used = 0.0  # mean imagined depth actually reached
        # field_goodness: the worst-aligned imagined moment of the last
        # round (the trough), thread-safe so the alignment gate can read how
        # far the mind is drifting from good without a model pass. Starts
        # neutral (0.0 = neither good nor bad).
        self.field_goodness = 0.0
        self._lock = threading.Lock()

    def _composite(self, vectors: Dict[str, torch.Tensor], names) -> Optional[torch.Tensor]:
        parts = [vectors[n].float() for n in names if n in vectors]
        if not parts:
            return None
        v = torch.stack(parts).mean(dim=0)
        return (v / (v.norm() + 1e-8)).to(self.device)

    def penumbra_companion(self) -> "SpeculativePenumbra":
        return SpeculativePenumbra(self)

    def snapshot(self) -> dict:
        with self._lock:
            futures = [dict(f) for f in self.last_futures]
        with self._penumbra_lock:
            pen = [
                {"word": p["word"], "weight": round(p["weight"], 4),
                 "utility": round(p["utility"], 3)}
                for p in self.penumbra
            ]
        return {
            "futures": futures,
            "penumbra": pen,
            "rounds_total": self.rounds_total,
            "skipped_busy": self.skipped_busy,
            "field_goodness": round(self.field_goodness, 3),
            "over_budget": self.over_budget,
            "rollout_tokens_used": round(self.rollout_tokens_used, 1),
            "search": self.last_search,
            "rollout_failures": self.rollout_failures,
            "last_rollout_error": self.last_rollout_error,
            "score_semantics": {"probability": "token_chain_confidence",
                                "goodness": "positive_affect_proxy",
                                "utility": "discounted_path_heuristic"},
        }

    # ------------------------------------------------------------------
    # Region step (expensive clock, runs in executor — may take ~100ms)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        candidates = self._gather_candidates(bus)
        if not candidates:
            return None

        # Skip if another operation owns the model; do not queue behind it.
        if not self.model_lock.acquire(blocking=False):
            with self._lock:
                self.skipped_busy += 1
            return None
        w_g = self.w_b
        if neuromod is not None and hasattr(neuromod, "reward_boost"):
            w_g *= neuromod.reward_boost(scale=0.4)
        try:
            base_state = bus.state.detach().clone()
            seeds: Dict[str, torch.Tensor] = {}

            def rollout(seed, parent, deadline, remaining_tokens):
                source, direction, mode = seed
                if parent is None:
                    if mode not in seeds:
                        seeds[mode] = self._seed_for_mode(mode)
                    context = seeds[mode]
                else:
                    # The next imagined event depends on what actually emerged
                    # in the parent, both its tokens and its residual change.
                    context = parent["context_ids"]
                    direction = parent["vec"]
                result = self._imagine(base_state, direction, context, mode=mode,
                                       deadline=deadline, max_tokens=remaining_tokens)
                if result is not None:
                    result.update(source=source, mode=mode, name=result["phrase"])
                    return result
                return {"failed": True, "tokens_used": self.last_attempt_tokens}

            def evaluate(result, seed):
                p_cal = self.ledger.calibrated(result["probability"], seed[2])
                result["probability_cal"] = p_cal
                p = p_cal if p_cal is not None else result["probability"]
                return self.w_p * p + w_g * result["goodness"]

            search = search_futures(
                candidates[:self.n_futures], rollout, evaluate,
                max_depth=self.max_depth, branching_factor=self.branching_factor,
                beam_width=self.beam_width, max_nodes=self.max_nodes,
                max_tokens=self.max_total_tokens,
                budget_s=self.round_budget_ms / 1000.0, discount=self.discount,
            )
            scored = search.leaves
            with self._lock:
                self.last_search = {
                    "attempts": search.attempts, "nodes": len(search.nodes),
                    "tokens": search.tokens, "budget_exhausted": search.budget_exhausted,
                    "depth_reached": max((n["depth"] for n in search.nodes), default=0),
                    "tree": [{k: n[k] for k in ("id", "parent_id", "depth", "name",
                                                "utility", "local_utility", "epistemic_type")}
                             for n in search.nodes],
                }
        finally:
            self.model_lock.release()

        if not scored:
            return None

        winner, rest = scored[0], scored[1:]
        # Resolution belongs to fresh external perception, never an idle
        # reread of the same situation that seeded the forecast.
        self.ledger.record(scored, tick=bus.tick_count)
        self.subc.publish_futures(scored)

        # Retain plausible runners-up in the penumbra; discard the rest —
        # a promising future lingers in awareness (keyed on utility).
        with self._penumbra_lock:
            for f in rest:
                salience = f["utility"]
                if salience >= self.plausibility_floor:
                    self.penumbra.append({
                        "vec": f["vec"],
                        "weight": self.penumbra_gain * max(salience, 0.0),
                        "word": f["name"],
                        "utility": f["utility"],
                    })
            # Bounded awareness — only the strongest few linger.
            self.penumbra.sort(key=lambda p: -p["weight"])
            del self.penumbra[6:]

        # Selected positive-affect projections can raise the reward channel.
        # The trough feeds affect regulation; it is not an external harm score.
        field_goodness = float(min(f["goodness_min"] for f in search.nodes))
        self.field_goodness = field_goodness
        if neuromod is not None and hasattr(neuromod, "bump"):
            if winner["goodness"] > 0.15 and winner["utility"] > 0:
                neuromod.bump("reward", 0.08 * winner["goodness"])

        with self._lock:
            self.rounds_total += 1
            used = [f.get("tokens_used", 0) for f in search.nodes]
            self.rollout_tokens_used = float(np.mean(used)) if used else 0.0
            self.last_futures = [
                {
                    "source": f["source"],
                    "mode": f["mode"],
                    "word": f["name"],
                    "probability": round(f["probability"], 3),
                    "probability_cal": (round(f["probability_cal"], 3)
                                        if f.get("probability_cal")
                                        is not None else None),
                    "goodness": round(f["goodness"], 3),
                    "goodness_min": round(f["goodness_min"], 3),
                    "utility": round(f["utility"], 3),
                    "chosen": f is winner,
                    "id": f["id"], "parent_id": f["parent_id"],
                    "depth": f["depth"], "epistemic_type": "hypothesis",
                }
                for f in scored
            ]

        # Journal the winner + penumbra survivors (throttled to the few
        # that matter, not every raw candidate) so the timeline can show
        # what was imagined this round.
        if self.journal is not None:
            try:
                for f in ([winner] + rest[:3]):
                    self.journal.emit(
                        "future_considered", turn=bus.tick_count,
                        word=f["name"], mode=f["mode"],
                        probability=round(f["probability"], 3),
                        goodness=round(f["goodness"], 3),
                        goodness_min=round(f["goodness_min"], 3),
                        utility=round(f["utility"], 3),
                        chosen=(f is winner))
            except Exception:  # noqa: BLE001
                pass

        # The chosen future pulls the present toward it.
        v = winner["vec"].to(bus.device).float()
        v = v / (v.norm() + 1e-8)
        strength = self.winner_strength * float(np.clip(winner["utility"], 0.0, 1.0))
        if strength <= 1e-6:
            return None
        return v * strength

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _gather_candidates(self, bus: LatentBus) -> List[Tuple[str, torch.Tensor, str]]:
        """Candidates are (source, direction, mode): speech futures continue
        the conversation; world futures imagine the situation's next event;
        user futures imagine the user's next action."""
        out: List[Tuple[str, torch.Tensor, str]] = []

        # Subconscious L2 survivors — what's already bubbling up
        for cand, _score in self.subc.peek_l2_candidates()[:2]:
            v = cand.vec.to(self.device).float()
            if v.norm() > 1e-6:
                out.append((f"subconscious:{cand.source}", v / v.norm(), "speech"))

        # The situation itself — "what will the user do?"
        if self.situation_provider is not None:
            try:
                sit = self.situation_provider()
                if sit is not None:
                    s = sit.detach().float().to(self.device)
                    if s.norm() > 1e-6:
                        out.append(("situation", s / s.norm(), "user"))
            except Exception:  # noqa: BLE001
                pass

        # Velocity extrapolation — "if the world keeps going this way"
        vel = bus.velocity.detach().float()
        if vel.norm() > 1e-4:
            out.append(("trajectory", (vel / vel.norm()).to(self.device), "world"))

        # A memory trace — "what if the past repeats, out there".
        # Snapshot under the Memory lock: the pool is sorted/trimmed in
        # place on other threads and an unlocked index can go stale.
        if self.memory is not None and getattr(self.memory, "pool", None):
            mem_lock = getattr(self.memory, "_lock", None)
            if mem_lock is not None:
                with mem_lock:
                    pool = list(self.memory.pool)
            else:
                pool = list(self.memory.pool)
            if pool:
                c = pool[int(np.random.randint(len(pool)))]
                v = c.vec.to(self.device).float()
                if v.norm() > 1e-6:
                    out.append(("memory", v / v.norm(), "world"))

        # Wildcard — genuine unknown, anywhere
        w = torch.randn(bus.hidden_dim, device=self.device)
        out.append(("wildcard", w / (w.norm() + 1e-8),
                    ["speech", "world", "user"][int(np.random.randint(3))]))

        return out

    def _context_seed(self) -> torch.Tensor:
        """The stage imagination runs on: recent conversation plain-text
        tokens, or BOS before the first exchange."""
        if self.context_provider is not None:
            try:
                ids = self.context_provider()
                if ids is not None and ids.numel() > 0:
                    return ids[-64:].detach().reshape(1, -1).to(self.device)
            except Exception:  # noqa: BLE001
                pass
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
        return torch.tensor([[bos]], device=self.device)

    def _seed_for_mode(self, mode: str) -> torch.Tensor:
        """Build the imagination stage for a mode. World and user modes
        are framed on the Now narrative; without one they fall back to
        the conversation tail."""
        if mode in ("world", "user") and self.situation_text_provider is not None:
            try:
                sit = self.situation_text_provider()
            except Exception:  # noqa: BLE001
                sit = None
            if sit:
                frame = (f"{sit} What happens next:" if mode == "world"
                         else f"{sit} The user will probably")
                try:
                    ids = self.tokenizer(
                        frame, return_tensors="pt", truncation=True,
                        max_length=64, add_special_tokens=False,
                    )["input_ids"]
                    if ids.numel() > 0:
                        return ids.to(self.device)
                except Exception:  # noqa: BLE001
                    pass
        return self._context_seed()

    def _goodness_of(self, h: torch.Tensor) -> float:
        """Signed positive-affect projection proxy in [-1, 1].

        Dividing by the sum of absolute deltas discards magnitude; this is
        a heuristic, not a calibrated measure of preference or welfare.
        """
        if h is None or not self._emo_vecs:
            return 0.0
        pos_deltas = [
            float(torch.dot(h, self._emo_vecs[n])) - float(self._base_proj.get(n, 0.0))
            for n in POSITIVE_EMOTIONS if n in self._emo_vecs
        ]
        if not pos_deltas:
            return 0.0
        mag = sum(abs(d) for d in pos_deltas) + 1e-6
        return sum(pos_deltas) / mag

    def _imagine(
        self, base_state: torch.Tensor, direction: torch.Tensor,
        seed: Optional[torch.Tensor] = None, mode: str = "world",
        deadline: Optional[float] = None, max_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        """Sample one hypothetical event, with no direct feedback into the bus.

        Return the generated context and residual change so a child can
        condition on its parent. Token confidence and affect are heuristics;
        neither establishes truth, welfare, or external-world probability.
        """
        steer = base_state.to(self.device) + direction.to(self.device) * self.imagination_strength
        self.last_attempt_tokens = 0
        if seed is None:
            seed = self._context_seed()
        depth = self.rollout_tokens + (self.chained_continuation_tokens if mode == "world" else 0)
        if max_tokens is not None:
            depth = min(depth, max_tokens)
        # Reserve room for the entire event in models with absolute positions.
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is None:
            limit = getattr(self.model.config, "n_positions", 1024)
        depth = min(depth, max(0, int(limit) - 1))
        if depth < 1:
            return None
        seed = seed[:, -max(1, int(limit) - depth):]
        stop_at = time.monotonic() + self.rollout_budget_ms / 1000.0
        if deadline is not None:
            stop_at = min(stop_at, deadline)
        ids, logps, captured = [], [], []
        over = False
        try:
            with self.hook.isolated(steer, capture=True), torch.no_grad():
                if time.monotonic() >= stop_at:
                    self.over_budget += 1
                    return None
                out = self.model(seed, use_cache=True)
                # Subtract this event's own context boundary, rather than
                # steering directly with an unrelated input embedding.
                context_h = self.hook.captured_states[-1][0, -1].float()
                self.hook.clear()
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                for _ in range(depth):
                    if time.monotonic() >= stop_at:
                        over = True
                        break
                    logp = F.log_softmax(logits, dim=-1)
                    probs = F.softmax(logits / 0.9, dim=-1)
                    top = torch.topk(probs, k=min(50, probs.numel()))
                    pick = int(torch.multinomial(top.values / top.values.sum(), 1).item())
                    tok = int(top.indices[pick])
                    if tok in self._special_ids:
                        break
                    ids.append(tok)
                    self.last_attempt_tokens = len(ids)
                    logps.append(float(logp[tok]))
                    out = self.model(torch.tensor([[tok]], device=self.device),
                                     past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1].float()
                captured = [c[0, -1].detach().float().to(self.device)
                            for c in self.hook.captured_states]
        except Exception as exc:
            self.rollout_failures += 1
            self.last_rollout_error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            if over:
                self.over_budget += 1
        if not ids or not captured:
            return None
        predicted_state = torch.stack(captured).mean(dim=0)
        delta = predicted_state - context_h.to(self.device)
        if not torch.isfinite(delta).all():
            self.rollout_failures += 1
            self.last_rollout_error = "nonfinite rollout state"
            return None
        # A degenerate residual change has no inferred direction; retain
        # the proposal as an explicitly hypothetical fallback.
        vec = delta if delta.norm() > 1e-6 else direction.to(self.device)
        vec = vec / (vec.norm() + 1e-8)
        phrase = " ".join(self.tokenizer.decode(ids, skip_special_tokens=True).split())[:96] or "…"
        per = [self._goodness_of(h) for h in captured]
        return {
            "probability": float(np.exp(np.mean(logps))),
            "goodness": float(np.mean(per)), "goodness_min": float(np.min(per)),
            "phrase": phrase, "tokens_used": len(ids), "over_budget": over,
            "vec": vec, "predicted_state": predicted_state,
            "context_ids": torch.cat((seed, torch.tensor([ids], device=self.device)), dim=1),
        }


class SpeculativePenumbra(Region):
    """Flow-clock companion: emits the retained-but-unattended futures at
    low gain, decaying over seconds. This is the 'aware of it but not
    thinking about it' bandwidth."""

    name = "speculative_penumbra"
    clock = "flow"

    def __init__(self, parent: SpeculativeFutures, decay_per_tick: float = 0.985):
        self._parent = parent
        self.decay = float(decay_per_tick)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._parent._penumbra_lock:  # noqa: SLF001 — shared by design
            pen = self._parent.penumbra
            if not pen:
                return None
            out = torch.zeros(bus.hidden_dim, device=bus.device)
            for p in pen:
                out = out + p["vec"].to(bus.device).float() * p["weight"]
                p["weight"] *= self.decay
            self._parent.penumbra = [p for p in pen if p["weight"] > 1e-3]
        if out.norm() < 1e-6:
            return None
        return out
