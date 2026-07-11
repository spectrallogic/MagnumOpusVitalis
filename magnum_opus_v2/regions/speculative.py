"""
SpeculativeFutures — parallel future prediction with probability /
benefit / risk sorting.

Design: predict possible futures in parallel, in realtime; sort by
probability, benefit, and risk; discard the rest — except that the
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
       benefit     — 70% latent (rollout hidden states projected on the
                     profile's positive-emotion vectors against a neutral
                     baseline) + 30% lexicon logit signal (LEXICON_WEIGHT),
                     boosted by reward.
       risk        — same 70/30 blend against the threat composite
                     (fear/anger/disgust/desperate), amplified by stress.
       utility     = w_p·prob + w_b·benefit − w_r·risk
  4. SORT by utility. The winner perturbs the bus (the chosen future pulls
     the present toward it). Runners-up above the plausibility floor are
     RETAINED in the penumbra — a low-gain channel emitted faintly every
     flow tick and decaying over seconds: known, not attended. The rest
     are discarded.

Chemistry feedback: a high-benefit winner bumps reward (anticipation);
a high-risk field bumps stress (dread). Imagined futures have real
physiological consequences — like ours do.
"""

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.regions.subconscious import SubconsciousStack

POSITIVE_EMOTIONS = ("joy", "trust", "calm", "curious")
THREAT_EMOTIONS = ("fear", "anger", "disgust", "desperate", "sadness")

# How a future FEELS is scored in LATENT SPACE first: the rollout's own
# mid-layer hidden states are projected onto the profile's emotion vectors
# (directions extracted from the model's own geometry) relative to the
# neutral baseline. No surface words required — a future is threatening if
# it moves the model's state toward its own fear direction, whatever
# vocabulary it happens to use.
#
# A small lexicon channel remains as a SECONDARY signal (30%): the shift in
# next-token probability mass over emotionally charged words. It reads the
# model's beliefs, not the conversation's wording — its weakness is basket
# narrowness, which is why it no longer leads.
BENEFIT_WORDS = [
    "good", "great", "love", "happy", "wonderful", "safe", "hope",
    "joy", "beautiful", "peace", "friend", "warm", "success",
]
THREAT_WORDS = [
    "danger", "fear", "bad", "death", "pain", "angry", "hate",
    "terrible", "hurt", "afraid", "alone", "lost", "fail",
]
LEXICON_WEIGHT = 0.3   # latent projection carries the other 0.7


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
        limbic=None,                # optional — imagined risk frightens for real
        device: str = "cpu",
        model_lock: Optional[threading.Lock] = None,
        n_futures: int = 4,
        rollout_tokens: int = 6,             # imagined depth: futures are PHRASES
        imagination_strength: float = 1.2,   # candidate offset added to bus.state
        winner_strength: float = 0.5,        # perturbation magnitude of chosen future
        plausibility_floor: float = 0.05,    # min utility to survive into penumbra
        penumbra_gain: float = 0.08,         # how loud the unattended futures are
        w_probability: float = 0.4,
        w_benefit: float = 0.35,
        w_risk: float = 0.45,
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
        self.imagination_strength = float(imagination_strength)
        self.winner_strength = float(winner_strength)
        self.plausibility_floor = float(plausibility_floor)
        self.penumbra_gain = float(penumbra_gain)
        self.w_p = float(w_probability)
        self.w_b = float(w_benefit)
        self.w_r = float(w_risk)

        # Emotion composites for benefit/risk scoring
        self._benefit_dir = self._composite(emotion_vectors, POSITIVE_EMOTIONS)
        self._threat_dir = self._composite(emotion_vectors, THREAT_EMOTIONS)

        # Full per-emotion vectors + neutral baseline — the latent affect
        # reader for imagined futures (same convention as perception).
        self._emo_vecs: Dict[str, torch.Tensor] = {
            n: v.detach().float().to(device)
            for n, v in emotion_vectors.items()
            if not n.startswith("temporal_")
        }
        self._base_proj: Dict[str, float] = dict(baseline_projections or {})

        # Emotional lexicon token ids for logit-shift scoring (secondary)
        self._pos_ids = self._lexicon_ids(BENEFIT_WORDS)
        self._neg_ids = self._lexicon_ids(THREAT_WORDS)

        # Shared with the flow-clock penumbra companion
        self.penumbra: List[dict] = []  # {"vec","weight","word","utility"}
        self._penumbra_lock = threading.Lock()

        # Era 6: imagined futures become accountable predictions
        from magnum_opus_v2.forecast import ForecastLedger
        self.ledger = ForecastLedger()

        # Diagnostics for dashboard
        self.last_futures: List[dict] = []
        self.rounds_total = 0
        self.skipped_busy = 0
        self._lock = threading.Lock()

    def _composite(self, vectors: Dict[str, torch.Tensor], names) -> Optional[torch.Tensor]:
        parts = [vectors[n].float() for n in names if n in vectors]
        if not parts:
            return None
        v = torch.stack(parts).mean(dim=0)
        return (v / (v.norm() + 1e-8)).to(self.device)

    def _lexicon_ids(self, words) -> Optional[torch.Tensor]:
        ids = set()
        for w in words:
            for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                try:
                    toks = self.tokenizer.encode(form, add_special_tokens=False)
                except Exception:  # noqa: BLE001
                    continue
                if toks:
                    ids.add(int(toks[0]))
        if not ids:
            return None
        return torch.tensor(sorted(ids), device=self.device, dtype=torch.long)

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
            # Per-mode stages and per-mode baseline passes ("the present,
            # unimagined"). Every future is scored by what it CHANGES
            # relative to its own stage's baseline — otherwise the model's
            # large baseline response drowns the candidate signal.
            seeds: Dict[str, torch.Tensor] = {}
            bases: Dict[str, Optional[torch.Tensor]] = {}
            scored = []
            for source, vec, mode in candidates[: self.n_futures]:
                if mode not in seeds:
                    seeds[mode] = self._seed_for_mode(mode)
                    bases[mode] = self._silent_pass(base_state, seeds[mode])
                logp_base = bases[mode]
                if logp_base is None:
                    continue
                result = self._imagine(base_state, vec, logp_base, seeds[mode])
                if result is None:
                    continue
                prob, benefit, risk, phrase = result
                scored.append({
                    "source": source, "vec": vec, "mode": mode,
                    "probability": prob, "benefit": benefit, "risk": risk,
                    "name": phrase,
                })
        finally:
            self.model_lock.release()

        if not scored:
            return None

        # Neuromod tilts the scoring the way chemistry tilts ours:
        # reward chases benefit, stress magnifies risk.
        w_b, w_r = self.w_b, self.w_r
        if neuromod is not None:
            if hasattr(neuromod, "reward_boost"):
                w_b *= neuromod.reward_boost(scale=0.4)
            if hasattr(neuromod, "stress_gain"):
                w_r *= neuromod.stress_gain(scale=0.6)

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
            f["utility"] = (
                self.w_p * p_eff + w_b * f["benefit"] - w_r * f["risk"]
            )
        scored.sort(key=lambda f: -f["utility"])
        winner, rest = scored[0], scored[1:]
        self.ledger.record(scored, tick=bus.tick_count)

        # Retain plausible runners-up in the penumbra; discard the rest.
        with self._penumbra_lock:
            for f in rest:
                # Retention keys on SALIENCE, not just utility — a
                # threatening future lingers in awareness precisely
                # because it is threatening. That is what an intrusive
                # thought is.
                salience = max(f["utility"], 0.6 * f["risk"])
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
        # what happened — an imagined fall frightens for real.
        field_risk = float(max(f["risk"] for f in scored))
        if neuromod is not None and hasattr(neuromod, "bump"):
            if winner["benefit"] > 0.15 and winner["utility"] > 0:
                neuromod.bump("reward", 0.08 * winner["benefit"])
            avg_risk = float(np.mean([f["risk"] for f in scored]))
            if avg_risk > 0.08:
                neuromod.bump("stress", 0.18 * avg_risk)
        if self.limbic is not None and field_risk > 0.12:
            try:
                self.limbic.stimulate(
                    "fear", min(0.5, 0.6 * field_risk), neuromod=neuromod,
                )
            except Exception:  # noqa: BLE001
                pass

        with self._lock:
            self.rounds_total += 1
            self.last_futures = [
                {
                    "source": f["source"],
                    "mode": f["mode"],
                    "word": f["name"],
                    "probability": round(f["probability"], 3),
                    "probability_cal": (round(f["probability_cal"], 3)
                                        if f.get("probability_cal")
                                        is not None else None),
                    "benefit": round(f["benefit"], 3),
                    "risk": round(f["risk"], 3),
                    "utility": round(f["utility"], 3),
                    "chosen": f is winner,
                }
                for f in scored
            ]

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

    def _silent_pass(self, steer: torch.Tensor,
                     seed: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """One steered silent forward pass ON THE GIVEN STAGE. Returns
        next-token log-probs or None. The steering injects at the mid
        layer; its causal effect is read downstream in the logits."""
        if seed is None:
            seed = self._context_seed()
        self.hook.set_steering(steer.to(self.device))
        try:
            with torch.no_grad():
                out = self.model(seed)
        except Exception:  # noqa: BLE001 — never crash the substrate
            return None
        finally:
            self.hook.set_steering(None)

        logits = out.logits[0, -1].detach().float()
        return F.log_softmax(logits, dim=-1)

    def _imagine(
        self,
        base_state: torch.Tensor,
        direction: torch.Tensor,
        logp_base: torch.Tensor,
        seed: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[float, float, float, str]]:
        """LIVE the candidate future: a short sampled rollout on the given
        stage under candidate steering. The future is a PHRASE — an
        imagined continuation of the situation — scored against that
        stage's unimagined baseline. Returns (probability, benefit, risk,
        phrase) or None."""
        steer = (base_state.to(self.device)
                 + direction * self.imagination_strength)
        if seed is None:
            seed = self._context_seed()

        self.hook.set_steering(steer)
        rollout_h = None
        try:
            with torch.no_grad():
                out = self.model(seed, use_cache=True)
                first_logits = out.logits[0, -1].detach().float()
                logp = F.log_softmax(first_logits, dim=-1)

                # Rollout — the imagined event, token by token, steering
                # held the whole way. Capture each step's mid-layer hidden
                # state: the latent trace of LIVING this future.
                self.hook.clear()
                self.hook.capture_enabled = True
                past = out.past_key_values
                ids: list = []
                logps: list = []
                cur_logits = first_logits
                for _ in range(self.rollout_tokens):
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
                if self.hook.captured_states:
                    rollout_h = torch.stack([
                        c[0, -1].detach().float()
                        for c in self.hook.captured_states
                    ]).mean(dim=0).to(self.device)
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

        # ---- LATENT affect (primary): where did living this future move
        # the model's own mid-layer state, measured against its own emotion
        # directions and neutral baseline? Scale-invariant across models,
        # and requires no particular vocabulary from the future at all.
        lat_benefit = lat_risk = 0.0
        if rollout_h is not None and self._emo_vecs:
            deltas: Dict[str, float] = {}
            for n, v in self._emo_vecs.items():
                deltas[n] = (float(torch.dot(rollout_h, v))
                             - float(self._base_proj.get(n, 0.0)))
            pos = sum(max(0.0, deltas.get(n, 0.0)) for n in POSITIVE_EMOTIONS)
            neg = sum(max(0.0, deltas.get(n, 0.0)) for n in THREAT_EMOTIONS)
            tot = sum(abs(d) for d in deltas.values()) + 1e-6
            lat_benefit = (pos - neg) / tot
            lat_risk = neg / tot

        # ---- lexicon channel (secondary): belief shift over charged words
        shift = logp - logp_base
        lex_benefit = lex_risk = 0.0
        if self._pos_ids is not None:
            lex_benefit = float(np.tanh(3.0 * float(shift[self._pos_ids].mean())))
        if self._neg_ids is not None:
            lex_risk = float(np.tanh(3.0 * float(shift[self._neg_ids].mean())))

        benefit = (1.0 - LEXICON_WEIGHT) * lat_benefit + LEXICON_WEIGHT * lex_benefit
        risk = ((1.0 - LEXICON_WEIGHT) * lat_risk
                + LEXICON_WEIGHT * max(0.0, lex_risk))
        return probability, benefit, max(0.0, risk), phrase

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
