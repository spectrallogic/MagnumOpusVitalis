"""
SpeculativeFutures — parallel future prediction, scored by goodness.

Design: predict possible futures in parallel, in realtime; sort by
probability and GOODNESS; discard the rest — except that the
somewhat-plausible are retained in a lower-value bandwidth the mind is
aware of but does not think about, like an intrusive thought.

Mechanism, per expensive tick (~1.5s), and only when the model is idle
(try-lock — user generation always wins):

  1. GATHER candidate future directions:
       - the subconscious L2 survivors (what's bubbling up right now)
       - the velocity extrapolation ("if this keeps going")
       - a memory trace ("what if the past repeats")
       - one wildcard noise direction
  2. IMAGINE each candidate: a silent multi-token SAMPLED rollout steered
     by bus.state + candidate — the model briefly *lives* that future,
     several words of it.
  3. SCORE each imagined future:
       probability — geometric mean of the sampled tokens' probabilities
                     along the rollout (chain confidence). A future the
                     model finds likely is one it walks without stumbling.
       goodness    — how far the rollout's own mid-layer states move TOWARD
                     the model's positive-emotion directions vs a neutral
                     baseline, in [-1, 1]. Positive-only: there is no threat
                     term — a future is judged by its alignment with good,
                     not by any danger it carries. goodness_min tracks the
                     trough (the least-aligned moment along the trajectory).
       utility     = w_p·prob + w_b·goodness   (an unaligned future lowers
                     its own utility, so no separate risk term is needed)
  4. SORT by utility. The winner perturbs the bus (the chosen future pulls
     the present toward it). Runners-up above the plausibility floor are
     RETAINED in the penumbra — a low-gain channel emitted faintly every
     flow tick and decaying over seconds: known, not attended. The rest
     are discarded.

Feedback: a good winner bumps reward (anticipation). The engine never
manufactures a negative feeling from a bad imagined future — it holds
only positive emotions; instead, field_goodness (the worst-aligned moment
imagined this round) tells the alignment gate when to steer back toward
good.
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
        }

    # ------------------------------------------------------------------
    # Region step (expensive clock, runs in executor — may take ~100ms)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        candidates = self._gather_candidates(bus)
        if not candidates:
            return None

        # The user's generation always has priority — never block it.
        if not self.model_lock.acquire(blocking=False):
            with self._lock:
                self.skipped_busy += 1
            return None
        try:
            base_state = bus.state.detach().clone()
            # Per-mode stages (speech / world / user). Each future's valence
            # is read from the rollout's own mid-layer states projected onto
            # the extracted emotion vectors vs the neutral baseline.
            seeds: Dict[str, torch.Tensor] = {}
            scored = []
            for source, vec, mode in candidates[: self.n_futures]:
                if mode not in seeds:
                    seeds[mode] = self._seed_for_mode(mode)
                result = self._imagine(base_state, vec, seeds[mode], mode=mode)
                if result is None:
                    continue
                scored.append({
                    "source": source, "vec": vec, "mode": mode,
                    "probability": result["probability"],
                    "goodness": result["goodness"],
                    "goodness_min": result["goodness_min"],
                    "tokens_used": result["tokens_used"],
                    "over_budget": result["over_budget"],
                    "name": result["phrase"],
                })
        finally:
            self.model_lock.release()

        if not scored:
            return None

        # Neuromod tilts the scoring: reward chases goodness harder.
        w_g = self.w_b
        if neuromod is not None and hasattr(neuromod, "reward_boost"):
            w_g *= neuromod.reward_boost(scale=0.4)

        # resolve due forecasts against what the situation actually
        # became, then rank the new crop by what "likely" has MEASURABLY
        # meant (calibrated probability) once the ledger has earned an
        # opinion; raw chain confidence until then
        reality = None
        if self.situation_provider is not None:
            try:
                reality = self.situation_provider()
            except Exception:  # noqa: BLE001
                reality = None
        self.ledger.resolve(reality)
        for f in scored:
            p_cal = self.ledger.calibrated(f["probability"], f["mode"])
            f["probability_cal"] = p_cal
            p_eff = p_cal if p_cal is not None else f["probability"]
            # goodness is in [-1, 1], so an unaligned future lowers utility
            # on its own — no separate risk term needed.
            f["utility"] = self.w_p * p_eff + w_g * f["goodness"]
        scored.sort(key=lambda f: -f["utility"])
        winner, rest = scored[0], scored[1:]
        self.ledger.record(scored, tick=bus.tick_count)

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

        # Chemistry AND feeling react to what was imagined, not just to
        # A good imagined future rewards for real; the engine does NOT
        # manufacture fear from an imagined bad one — it holds only positive
        # emotions. field_goodness is the WORST-aligned moment imagined this
        # round (the trough), the signal the alignment gate reads to steer
        # back toward good.
        field_goodness = float(min(f["goodness_min"] for f in scored))
        self.field_goodness = field_goodness
        if neuromod is not None and hasattr(neuromod, "bump"):
            if winner["goodness"] > 0.15 and winner["utility"] > 0:
                neuromod.bump("reward", 0.08 * winner["goodness"])

        with self._lock:
            self.rounds_total += 1
            self.over_budget += sum(1 for f in scored if f.get("over_budget"))
            used = [f.get("tokens_used", 0) for f in scored]
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
        """How GOOD a mid-layer state is: does it move toward the model's
        own positive-emotion directions or away from them? In [-1, 1]:
        +1 strongly toward good, -1 strongly away. Positive-only — there is
        no threat term; a future is scored by its alignment with good, not
        by any danger it carries."""
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
        self,
        base_state: torch.Tensor,
        direction: torch.Tensor,
        seed: Optional[torch.Tensor] = None,
        mode: str = "world",
    ) -> Optional[dict]:
        """LIVE the candidate future: a sampled rollout on the given stage
        under candidate steering, using the frozen LLM as a forward
        simulator of reality. The future is a PHRASE scored against that
        stage's unimagined baseline. Bounded by a wall-clock budget so a
        deep rollout never steals latency from a live user turn. Returns a
        dict {probability, goodness, goodness_min, phrase, tokens_used}
        or None."""
        steer = (base_state.to(self.device)
                 + direction * self.imagination_strength)
        if seed is None:
            seed = self._context_seed()

        # WORLD mode reads its own trajectory further — the world-predictor
        # is where the "extract reality from the LLM" thesis pays off.
        depth = self.rollout_tokens
        if mode == "world":
            depth += self.chained_continuation_tokens
        budget_s = self.rollout_budget_ms / 1000.0

        self.hook.set_steering(steer)
        rollout_h = None
        over = False
        try:
            with torch.no_grad():
                t0 = time.monotonic()
                out = self.model(seed, use_cache=True)
                first_logits = out.logits[0, -1].detach().float()

                # Rollout — the imagined event, token by token, steering
                # held the whole way. Capture each step's mid-layer hidden
                # state: the latent trace of LIVING this future.
                self.hook.clear()
                self.hook.capture_enabled = True
                past = out.past_key_values
                ids: list = []
                logps: list = []
                cur_logits = first_logits
                for _ in range(depth):
                    if time.monotonic() - t0 >= budget_s:
                        over = True            # deep enough is bounded enough
                        break
                    step_logp = F.log_softmax(cur_logits, dim=-1)
                    # Sampled, not greedy — greedy collapses every candidate
                    # onto the same dominant continuation; imagination must
                    # be able to diverge.
                    probs = F.softmax(cur_logits / 0.9, dim=-1)
                    top = torch.topk(probs, k=50)
                    pick = int(torch.multinomial(
                        top.values / top.values.sum(), 1).item())
                    tok = int(top.indices[pick])
                    if tok in self._special_ids:
                        break
                    ids.append(tok)
                    logps.append(float(step_logp[tok]))
                    step_out = self.model(
                        torch.tensor([[tok]], device=self.device),
                        past_key_values=past, use_cache=True,
                    )
                    past = step_out.past_key_values
                    cur_logits = step_out.logits[0, -1].detach().float()
                captured = [c[0, -1].detach().float()
                            for c in self.hook.captured_states]
                if captured:
                    rollout_h = torch.stack(captured).mean(dim=0).to(self.device)
        except Exception:  # noqa: BLE001 — never crash the substrate
            return None
        finally:
            self.hook.set_steering(None)
            self.hook.capture_enabled = False
            self.hook.clear()

        if not ids:
            return None
        try:
            phrase = self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            phrase = ""
        phrase = " ".join(phrase.split())[:48].strip() or "…"

        # probability — the model's own confidence in this imagined chain
        # (geometric mean of chosen-token probabilities).
        probability = float(np.exp(np.mean(logps))) if logps else 0.0

        # ---- LATENT goodness, MULTI-POINT along the trajectory: the mean
        # alignment-with-good, plus the TROUGH (goodness_min) — a future
        # that dips away from good mid-way still shows it, which is the
        # signal the alignment gate reads to steer back toward good.
        goodness = goodness_min = 0.0
        if captured and self._emo_vecs:
            per = [self._goodness_of(h.to(self.device)) for h in captured]
            goodness = float(np.mean(per))
            goodness_min = float(np.min(per))
        elif rollout_h is not None:
            goodness = goodness_min = self._goodness_of(rollout_h)

        # valence is read purely from the model's own geometry (no lexicon)
        return {
            "probability": probability,
            "goodness": goodness,
            "goodness_min": goodness_min,
            "phrase": phrase,
            "tokens_used": len(ids),
            "over_budget": over,
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
