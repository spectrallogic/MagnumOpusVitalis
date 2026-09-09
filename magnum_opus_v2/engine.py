"""
V2Engine — thin orchestrator.

Assembles: LatentBus + NeuromodState + all regions + SteeringHook + driver +
FlowRunner. Provides a small public surface:

    engine = V2Engine.from_profile(model, tokenizer, profile, device=...)
    engine.start()                       # spin up flow runner
    engine.user_message("hi")            # mark interaction, light keyword stim
    text = engine.converse("hello")      # generate with live bus driving steering
    snap = engine.snapshot()             # everything at-a-glance
    engine.stop()

The engine itself does NOT orchestrate per-tick logic — that's the bus +
clocks. This class just wires regions together at construction and exposes
external entry points (user message, generate, snapshot).
"""

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import torch

from magnum_opus_v2.profile import ModelProfile
from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.config import V2Config
from magnum_opus_v2.flow import FlowRunner
from magnum_opus_v2.neuromod import NeuromodState, NeuromodulatorRegion
from magnum_opus_v2.steering_hook import SteeringHook, BusSteeringDriver
from magnum_opus_v2.regions import (
    Limbic, Temporal, SubconsciousStack, Memory, FalseMemoryConfabulator,
    Salience, Executive, DefaultMode, KnowledgeSparks,
    NoiseSampler, TokenEmbeddingSampler,
    SpeculativeFutures, AbstractionLadder, SelfModel, Consolidation,
    SituationModel,
)
from magnum_opus_v2.regions.alignment_gate import AlignmentGate
from magnum_opus_v2.journal import CognitionJournal


# No authored persona. The engine gives the frozen LLM a continuous felt
# state and steers it token-by-token; whatever character emerges is the
# LLM's own, surfaced by that steering — not a hand-written identity. A
# caller may still pass their own system_prompt explicitly, but the
# engine ships none.
DEFAULT_SYSTEM_PROMPT = None


# Emotion perception is latent-only: perceive_emotions() runs the user's
# message through the model and projects its hidden state onto the
# extracted emotion vectors. No keyword list — a message's felt content is
# read from the model's own semantic reading, never from authored words.


# Steered generation occasionally derails past its own turn and starts
# writing the OTHER side of the dialogue ("User: ..."). Never let that
# reach the ear, the caption, or — critically — the chat history, where
# it becomes a self-reinforcing script of the model talking to itself.
_ROLE_MARKER = re.compile(
    r"(?:\n|^)\s*(?:user|human|assistant|ai|vitalis)\s*[::]",
    re.IGNORECASE,
)
_LEADING_SELF_LABEL = re.compile(
    r"^\s*(?:vitalis|assistant|ai)\s*[::]\s*", re.IGNORECASE,
)


def _clean_reply(text: str) -> str:
    if not text:
        return text
    text = _LEADING_SELF_LABEL.sub("", text)
    m = _ROLE_MARKER.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


@dataclass
class V2Engine:
    bus: LatentBus
    neuromod: NeuromodState
    flow: FlowRunner
    hook: SteeringHook
    driver: BusSteeringDriver
    limbic: Limbic
    temporal: Temporal
    subc: SubconsciousStack
    memory: Memory
    confab: FalseMemoryConfabulator
    salience: Salience
    executive: Executive
    default_mode: Optional[DefaultMode]
    sparks: Optional[KnowledgeSparks]
    speculative: Optional[SpeculativeFutures]
    abstraction: Optional[AbstractionLadder]
    self_model: Optional[SelfModel]
    model: Any
    tokenizer: Any
    device: str = "cpu"
    model_lock: threading.Lock = field(default_factory=threading.Lock)
    _autonomous_messages: Deque[str] = field(default_factory=lambda: deque(maxlen=32))
    profile: Optional[ModelProfile] = None
    system_prompt: Optional[str] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    # converse() (caller thread) and autonomous speech (its own thread)
    # both mutate the history; the lock keeps turns from being lost.
    # RLock: mutation sites call _trim_history while already holding it.
    _history_lock: threading.RLock = field(default_factory=threading.RLock)
    max_history_turns: int = 8
    consolidation: Optional[Consolidation] = None
    situation: Optional[SituationModel] = None
    # steer toward good: when the mind drifts from its good baseline or
    # imagines a poorly-aligned future, it takes a "second thought" that
    # re-steers toward good before speaking. Never withholds (see
    # alignment_gate.py).
    alignment_gate: AlignmentGate = field(default_factory=AlignmentGate)
    # honest event feed — watch it think as a stream (see journal.py)
    journal: CognitionJournal = field(default_factory=CognitionJournal)
    # frozen causes of the last reply, for reply<-cause linking in the UI
    _last_causes: Optional[dict] = None
    # throttle for emotion_snapshot journal entries
    _last_emo_log: float = 0.0
    _last_dominant: Optional[str] = None
    # Latent think-before-speak passes per turn (Phase 5). 0 disables.
    rumination_steps: int = 2
    # Last conversation tokens — the stage on which imagination runs.
    _context_ids: Optional[Any] = None
    # Hidden-state reading of the last user message (the situation vector).
    _last_percept: Optional[Any] = None
    # What the situation reminded it of, for the dashboard.
    _last_recall: Optional[dict] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_profile(
        cls,
        model,
        tokenizer,
        profile: ModelProfile,
        device: str = "cpu",
        config: Optional[V2Config] = None,
        # Behavior switches
        enable_default_mode: bool = True,
        enable_knowledge_sparks: bool = True,
        enable_speculative: bool = True,
        enable_abstraction: bool = True,
        enable_self_model: bool = True,
        enable_consolidation: bool = True,
        enable_situation: bool = True,
        # Persona for chat-template models (ignored for base LMs like gpt2)
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        # Speech callback
        on_should_speak=None,
    ) -> "V2Engine":
        profile.validate_activation_site()
        cfg = config or V2Config(hidden_dim=profile.hidden_dim, device=device)

        bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)
        neuromod = NeuromodState()
        model_lock = threading.Lock()

        # Limbic + bus baseline. If the profile carries Mirror dynamics
        # (M1), the emotional constants — onset, decay, baselines, and the
        # cross-emotion interaction matrix — are the ones FITTED from this
        # model's own implied human trajectories, not the hand-authored
        # defaults in _dynamics.py.
        limbic_configs = None
        limbic_interactions = None
        if getattr(profile, "dynamics", None):
            from magnum_opus_v2._dynamics import EmotionConfig
            limbic_configs = {
                n: EmotionConfig(
                    onset_rate=d["onset_rate"],
                    decay_rate=d["decay_rate"],
                    baseline=d["baseline"],
                )
                for n, d in profile.dynamics.get("emotions", {}).items()
            }
            limbic_interactions = {
                (s, t): f
                for s, t, f in profile.dynamics.get("interactions", [])
            }
        # The engine HOLDS only positive/neutral emotions (alignment: it
        # only ever feels and steers toward good states). Negative vectors
        # are still extracted and used for PERCEPTION (perceive_emotions),
        # but the Limbic — the held/steered state — contains only these.
        from magnum_opus_v2._dynamics import HELD_EMOTIONS
        held = [n for n in profile.vectors
                if n in HELD_EMOTIONS and not n.startswith("temporal_")]
        limbic = Limbic(
            profile.vectors, device=device, steering_strength=1.0,
            emotion_names=held,
            configs=limbic_configs, interactions=limbic_interactions,
        )
        bus.set_baseline(limbic.baseline_vector(), weight=1.0)

        temporal = Temporal(profile.vectors, device=device, steering_strength=0.3)

        # Memory
        memory = Memory(device=device, capacity=200)
        confab = FalseMemoryConfabulator(memory, per_tick_count=1)
        # subjective time's fifth signal: how vividly memory holds
        temporal.memory_provider = memory.snapshot

        # SubconsciousStack with three samplers: noise, token-embedding, memory.
        # ONE shared fp32 copy of the embedding matrix — samplers, sparks and
        # the abstraction ladder all reuse it (a 3B model's vocab is ~1GB in
        # fp32; three private copies would not fit alongside the model).
        embedding_matrix = model.get_input_embeddings().weight.detach().to(device).float()
        forbidden = [
            tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id,
        ]
        forbidden = [f for f in forbidden if f is not None]
        samplers = [
            NoiseSampler(profile.hidden_dim, device=device, magnitude=1.0),
            TokenEmbeddingSampler(
                embedding_matrix, device=device,
                candidate_pool=64, forbidden_token_ids=forbidden,
            ),
            memory.make_sampler(),
        ]
        subc = SubconsciousStack(
            hidden_dim=profile.hidden_dim, device=device, samplers=samplers,
            l0_per_sampler_count=4, l0_emotion_bias_strength=0.5,
            l1_keep_top_k=8, l2_keep_top_k=3,
            l2_surprise_probability=0.15,
            l3_perturbation_strength=0.4,
            emotion_vectors=profile.vectors,
        )

        salience = Salience(subc, history_size=20)
        executive = Executive(
            base_threshold=0.55,
            pressure_growth=0.40,
            pressure_decay=0.985,
            on_should_speak=on_should_speak,
        )

        # Steering hook + driver
        hook = SteeringHook()
        hook.attach(model, profile.target_layer)
        driver = BusSteeringDriver(bus, steering_strength=1.0)

        # Optional model-touching regions
        default_mode = None
        if enable_default_mode:
            default_mode = DefaultMode(
                model=model, tokenizer=tokenizer,
                hook=hook, driver=driver,
                limbic=limbic,
                emotion_vectors=profile.vectors,
                device=device,
                idle_drift_amplitude=0.04,
                model_lock=model_lock,
            )

        sparks = None
        sparks_fire = None
        if enable_knowledge_sparks:
            sparks = KnowledgeSparks(
                model=model, limbic=limbic, device=device,
                fire_probability=0.05,
                spark_strength=0.6,
                embedding_matrix=embedding_matrix,
            )
            sparks_fire = sparks.fire_companion()

        # Parallel future prediction (probability/benefit/risk) + penumbra
        speculative = None
        penumbra = None
        if enable_speculative:
            speculative = SpeculativeFutures(
                model=model, tokenizer=tokenizer, hook=hook,
                subconscious=subc, emotion_vectors=profile.vectors,
                baseline_projections=profile.baseline.projections,
                memory=memory, limbic=limbic,
                device=device, model_lock=model_lock,
                n_futures=cfg.spec.n_futures,
                rollout_tokens=cfg.spec.rollout_tokens,
                rollout_budget_ms=cfg.spec.rollout_budget_ms,
                chained_continuation_tokens=cfg.spec.chained_continuation_tokens,
                max_depth=cfg.spec.max_depth,
                branching_factor=cfg.spec.branching_factor,
                beam_width=cfg.spec.beam_width,
                max_nodes=cfg.spec.max_nodes,
                max_total_tokens=cfg.spec.max_total_tokens,
                round_budget_ms=cfg.spec.round_budget_ms,
                discount=cfg.spec.discount,
            )
            penumbra = speculative.penumbra_companion()

        # Developmental coarse-to-fine concept learning
        abstraction = None
        if enable_abstraction:
            abstraction = AbstractionLadder(
                hidden_dim=profile.hidden_dim, device=device,
                embedding_matrix=embedding_matrix, tokenizer=tokenizer,
            )

        # Identity, felt time, memory leakage
        self_model = None
        if enable_self_model:
            self_model = SelfModel(memory=memory, device=device)

        # Sleep-work: replay important traces into the ladder, distill
        # dominant concepts into standing dispositions (bus attractors)
        consolidation = None
        if enable_consolidation and abstraction is not None:
            consolidation = Consolidation(memory=memory, abstraction=abstraction)

        # The Now — persistent awareness of what is happening
        situation = None
        if enable_situation:
            situation = SituationModel(device=device)

        # Neuromodulator slow-clock region
        neuromod_region = NeuromodulatorRegion(neuromod=neuromod)

        regions = [
            limbic, temporal, subc, salience, executive,
            memory, confab, neuromod_region,
        ]
        if default_mode is not None:
            regions.append(default_mode)
        if sparks is not None:
            regions.append(sparks)
            regions.append(sparks_fire)
        if speculative is not None:
            regions.append(speculative)
            regions.append(penumbra)
        if abstraction is not None:
            regions.append(abstraction)
        if self_model is not None:
            regions.append(self_model)
        if consolidation is not None:
            regions.append(consolidation)
        if situation is not None:
            regions.append(situation)

        flow = FlowRunner(
            bus=bus, regions=regions,
            clock_config=cfg.clock,
            neuromod=neuromod,
            verbose_errors=cfg.verbose_errors,
        )

        inst = cls(
            bus=bus, neuromod=neuromod, flow=flow,
            hook=hook, driver=driver,
            limbic=limbic, temporal=temporal, subc=subc,
            memory=memory, confab=confab,
            salience=salience, executive=executive,
            default_mode=default_mode, sparks=sparks,
            speculative=speculative, abstraction=abstraction,
            self_model=self_model,
            model=model, tokenizer=tokenizer, device=device,
            model_lock=model_lock,
            profile=profile, system_prompt=system_prompt,
            consolidation=consolidation, situation=situation,
        )

        # The cognition journal listens at the real emit sites.
        if speculative is not None:
            speculative.journal = inst.journal
            speculative.ledger.journal = inst.journal

        # Imagination runs on the live conversation — give speculation a
        # window onto the engine's context tail.
        if speculative is not None:
            speculative.context_provider = lambda: inst._context_ids
            speculative.situation_provider = lambda: (
                inst.situation.current_vec() if inst.situation else inst._last_percept
            )
            speculative.situation_text_provider = lambda: (
                inst.situation.narrative if inst.situation else None
            )

        # Traffic loop: while generating (and during silent passes), the
        # model's own thought periodically tugs the bus. Pre-steering
        # hidden state, small gain, clipped — thought colors feeling
        # without a runaway echo.
        def _thought_feedback(h_vec):
            try:
                v = h_vec.detach().float().to(device)
                delta = v - inst.bus.state
                n = float(delta.norm())
                if n > 4.0:
                    delta = delta * (4.0 / n)
                inst.bus.add_perturbation(delta * 0.02,
                                          source="thought_feedback")
            except Exception:  # noqa: BLE001 — never crash a forward pass
                pass
        hook.set_feedback(_thought_feedback, every=8)

        # If the user did not supply a speech callback, install a default that
        # generates autonomously and queues the result for the dashboard.
        # Generation runs in its own thread — the callback fires from the
        # perception clock, which must never block on a model pass.
        if on_should_speak is None:
            def _speak_worker():
                try:
                    text = inst.speak_autonomously()
                    inst._autonomous_messages.append(text)
                except Exception as e:  # noqa: BLE001 — never crash the
                    # substrate, but never die silently either: log it and
                    # release the urge so the pressure loop doesn't thrash
                    print(f"  [autonomous speech failed] {e}")
                    try:
                        inst.executive.mark_spoke()
                    except Exception:  # noqa: BLE001
                        pass

            def _default_speak():
                threading.Thread(
                    target=_speak_worker, name="v2-autonomous-speech", daemon=True,
                ).start()
            executive.on_should_speak = _default_speak

        return inst

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.flow.start()

    def stop(self, save_path=None) -> None:
        self.flow.stop()
        # a restart is a nap: snapshot AFTER the flow thread has joined,
        # so no tick races the save. Best-effort — never crash on exit.
        if save_path is not None:
            try:
                self.save_state(save_path)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Persistence — "a restart is a nap, not a death" (see persistence.py)
    # ------------------------------------------------------------------
    def save_state(self, path=None):
        from magnum_opus_v2 import persistence
        return persistence.save_engine(self, path)

    def load_state(self, path=None) -> bool:
        """Load a saved nap into this (built, not-yet-started) engine."""
        from magnum_opus_v2 import persistence
        return persistence.load_engine(self, path)

    def _turn(self) -> int:
        with self._history_lock:
            return len(self.chat_history) // 2

    def maybe_log_emotion(self, snap: Optional[dict] = None) -> None:
        """Emit an emotion_snapshot to the journal only on a MEANINGFUL
        change (dominant emotion flipped, or ≥1s elapsed) — honest, not
        spam. Called once per SSE tick by the server."""
        try:
            blend = (snap or {}).get("limbic", {}).get("blended") \
                or self.limbic.snapshot().get("blended", {})
            if not blend:
                return
            dominant = max(blend, key=lambda k: blend[k])
            now = time.monotonic()
            changed = dominant != self._last_dominant
            if not changed and (now - self._last_emo_log) < 1.0:
                return
            self._last_emo_log = now
            self._last_dominant = dominant
            self.journal.emit("emotion_snapshot", turn=self._turn(),
                              dominant=dominant,
                              value=round(float(blend[dominant]), 3))
        except Exception:  # noqa: BLE001
            pass

    def _capture_causes(self, prompt: str, reply: str, gate: dict) -> None:
        """Freeze the causal bundle of a reply and emit word_chosen +
        reply_emitted to the journal so the UI can link output to cause."""
        try:
            turn = self._turn()
            chosen = None
            if self.speculative is not None:
                for f in self.speculative.last_futures:
                    if f.get("chosen"):
                        chosen = {k: f.get(k) for k in
                                  ("word", "mode", "probability", "goodness",
                                   "depth", "id", "utility")}
                        break
            blend = self.limbic.snapshot().get("blended", {})
            dominant = max(blend, key=lambda k: blend[k]) if blend else None
            intr = self.subc.snapshot().get("intrusive_word")
            self._last_causes = {
                "turn": turn,
                "prompt": prompt[:80],
                "reply": reply[:200],
                "chosen_future": chosen,
                "dominant_emotion": dominant,
                "emotion": {k: round(float(v), 3) for k, v in blend.items()},
                "neuromod": self.neuromod.snapshot(),
                "intrusive": intr,
                "gate": {"action": gate.get("action"),
                         "misalignment": round(gate.get("misalignment", 0.0), 3)},
            }
            if chosen and chosen.get("word"):
                self.journal.emit("word_chosen", turn=turn,
                                  word=chosen["word"], mode=chosen.get("mode"))
            self.journal.emit("reply_emitted", turn=turn,
                              reply=reply[:80], dominant=dominant,
                              gate=gate.get("action"))
        except Exception:  # noqa: BLE001
            pass

    def _alignment_signals(self):
        """The cached scalars the alignment gate weighs — no model pass:
        how far the bus has drifted from its good baseline, and how far the
        worst imagined moment this round is from good."""
        divergence = float(self.bus.divergence_from_baseline())
        field_goodness = (float(getattr(self.speculative, "field_goodness", 0.0))
                          if self.speculative is not None else 0.0)
        return (divergence, field_goodness)

    def _calm_the_stance(self) -> None:
        """A brief perturbation toward the good baseline — re-steers the
        felt stance back toward good. The CONTENT still comes from the
        model; only the mood it speaks from is steadied."""
        try:
            calm = self.bus.attractors[0][0].to(self.bus.device)
            self.bus.add_perturbation((calm - self.bus.state) * 0.15,
                                      source="alignment_gate")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # External entry points
    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> Optional[torch.Tensor]:
        """The model's hidden-state reading of a piece of text — mean
        mid-layer activation, same layer + convention as extraction.py so
        it is comparable to profile vectors, baseline, and bus space."""
        if self.profile is None or not text:
            return None
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=96,
        ).to(self.device)
        from magnum_opus_v2.extraction import read_block_output
        with self.model_lock, self.hook.isolated():
            return read_block_output(self.model, enc, self.profile.target_layer)

    def perceive_emotions(self, text: str) -> Dict[str, float]:
        """The real sense organ: run the message through the model and
        project its mid-layer hidden state onto the extracted emotion
        vectors, relative to the profile's neutral baseline. The stimulus
        is the model's own semantic reading of the message — no word lists.

        Also stashes the situation vector (`_last_percept`) for
        exteroceptive learning and latent recall.

        Returns {emotion: intensity} for the emotions the message actually
        moved (top emotion scaled to 0.9, floor at 25% of the max)."""
        h = self._embed_text(text)
        self._last_percept = h
        if h is None:
            return {}
        if self.speculative is not None:
            self.speculative.ledger.resolve(h)
        base = self.profile.baseline.projections or {}
        deltas: Dict[str, float] = {}
        for name, vec in self.profile.vectors.items():
            if name.startswith("temporal_"):
                continue
            v = vec.to(h.device).float()
            deltas[name] = float(torch.dot(h, v)) - float(base.get(name, 0.0))
        pos = {k: d for k, d in deltas.items() if d > 0}
        if not pos:
            return {}
        m = max(pos.values())
        if m <= 1e-6:
            return {}
        return {
            k: min(0.9, 0.9 * d / m)
            for k, d in pos.items() if d / m > 0.25
        }

    def _latent_recall(self, h: torch.Tensor) -> None:
        """Being reminded: match the situation vector against episodic
        traces; the best match (weighted by its remaining importance)
        perturbs the bus. False memories participate — remembering wrong
        is still remembering."""
        self._last_recall = None
        if h is None or not self.memory.pool:
            return
        hn = (h / (h.norm() + 1e-8))
        best, best_score = None, 0.0
        with self.memory._lock:  # noqa: SLF001 — read of the shared pool
            for c in self.memory.pool:
                v = c.vec.float()
                n = float(v.norm())
                if n < 1e-6:
                    continue
                sim = float(torch.dot(v / n, hn.to(v.device)))
                imp = float((c.meta or {}).get("importance", 0.5))
                score = sim * (0.5 + 0.5 * min(imp, 1.0))
                if score > best_score:
                    best, best_score = c, score
        if best is None or best_score < 0.35:
            return
        direction = best.vec.float()
        direction = direction / (direction.norm() + 1e-8)
        self.bus.add_perturbation(
                direction.to(self.bus.device) * 0.35 * best_score,
                source="recall")
        self._last_recall = {
            "tag": (best.meta or {}).get("tag"),
            "score": round(best_score, 3),
            "false": bool((best.meta or {}).get("false", False)),
        }

    def user_message(self, text: str) -> None:
        """User just sent something. Perceive its emotional content in
        latent space (projection onto the extracted emotion vectors — no
        word list), learn the world-content of the moment (memory +
        abstraction ladder), get reminded of similar past moments, mark
        Temporal + Executive interaction, push the live emotion blend into
        the subconscious so its filtering is informed."""
        try:
            stim = self.perceive_emotions(text)
        except Exception:  # noqa: BLE001 — perception must never block a turn
            stim = {}
            self._last_percept = None
        if stim:
            self.limbic.stimulate_many(stim, neuromod=self.neuromod)

        # Exteroception + recall: the situation vector teaches the ladder,
        # becomes an episodic world-trace, searches the past, and updates
        # the Now (assimilate or scene-shift).
        h = self._last_percept
        if h is not None:
            try:
                if self.abstraction is not None:
                    self.abstraction.observe_external(h, self.neuromod)
                self._latent_recall(h)
                self.memory.capture_experience(
                    h, importance=1.3, tag=f"heard:{text[:40]}",
                )
                if self.situation is not None:
                    self.situation.observe_percept(h, self.neuromod)
            except Exception:  # noqa: BLE001
                pass

        self.temporal.mark_interaction(self.bus)
        self.executive.mark_interaction()
        # Also capture the FEELING of the moment (bus state), alongside
        # the world-content trace above.
        self.memory.force_capture(
            self.bus, importance=1.0, tag=f"felt:{text[:40]}",
        )
        self.subc.set_emotion_blend(self.limbic.snapshot()["blended"])

    def _uses_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def _trim_history(self) -> None:
        with self._history_lock:
            cap = 2 * self.max_history_turns
            if len(self.chat_history) > cap:
                # in place: other threads hold a reference to this list
                del self.chat_history[:-cap]

    def converse(
        self, prompt: str, max_new_tokens: int = 60,
        do_sample: bool = True, top_p: float = 0.92, temperature: float = 0.9,
        seed: Optional[int] = None,
    ) -> str:
        """One conversational turn with live bus steering. Returns the
        REPLY ONLY (not the prompt). Instruct-tuned models get their chat
        template plus a rolling history; base LMs get raw continuation."""
        self.user_message(prompt)
        if seed is not None:
            torch.manual_seed(seed)

        # Push current emotion into subconscious so steering reflects it.
        self.subc.set_emotion_blend(self.limbic.snapshot()["blended"])

        # Alignment gate: a direct answer is NEVER withheld (that would be
        # hidden censorship). If felt risk is high, take a calming second
        # thought before generating — the reply is still the model's.
        gate = self.alignment_gate.decide("converse", *self._alignment_signals())
        if gate["action"] != "pass":
            self.journal.emit("gate_fired", turn=self._turn(), site="converse",
                              action=gate["action"],
                              misalignment=round(gate["misalignment"], 3))

        with self.model_lock:
            if self._uses_chat_template():
                messages: List[Dict[str, str]] = []
                if self.system_prompt:
                    messages.append(
                        {"role": "system", "content": self.system_prompt})
                with self._history_lock:
                    messages.extend(list(self.chat_history))
                messages.append({"role": "user", "content": prompt})
                input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)
            else:
                input_ids = self.tokenizer(
                    prompt, return_tensors="pt",
                ).to(self.device)["input_ids"]

            # Second thought: on high felt risk, steady the stance and
            # think once more before answering (content stays the model's).
            if gate["action"] == "second_thought":
                self._calm_the_stance()
                self._ruminate_locked(input_ids)

            # Phase 5 — rumination: silently live the moment a few times
            # before answering. Each pass's thought perturbs the bus, so
            # the reply begins from a mind that has already reacted.
            self._ruminate_locked(input_ids)

            # LIVE steering: every token's forward pass reads the bus at
            # that moment. The flow keeps ticking during generation (its
            # clocks run in their own thread), so emotions, intrusive
            # thoughts, and spark decay shift the voice mid-sentence.
            self.hook.set_provider(self.driver.read)
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample, top_p=top_p,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                reply = _clean_reply(self.tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True,
                ))
            finally:
                self.hook.set_provider(None)

        # The stage on which imagination runs until the next turn: the
        # PLAIN TEXT of the exchange, not the chat-template scaffolding —
        # rollouts should continue the situation, not the turn format.
        try:
            ctx_text = f"{prompt}\n{reply}"
            self._context_ids = self.tokenizer(
                ctx_text, return_tensors="pt", truncation=True, max_length=64,
                add_special_tokens=False,
            )["input_ids"][0].detach()
        except Exception:  # noqa: BLE001
            pass

        with self._history_lock:
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": reply})
            self._trim_history()

        # Reply <- cause: freeze WHAT SHAPED this reply at emission time
        # (the winning future, the emotion the voice was tinted with, the
        # gate's stance, any surfaced intrusion). Frozen so a later view
        # never back-dates live state onto a past turn.
        self._capture_causes(prompt, reply, gate)

        # Learn what was said (world-content), not just how it felt.
        try:
            h_reply = self._embed_text(reply)
            if h_reply is not None:
                self.memory.capture_experience(
                    h_reply, importance=1.1, tag=f"said:{reply[:40]}",
                )
                if self.abstraction is not None:
                    self.abstraction.observe_external(h_reply, self.neuromod)
        except Exception:  # noqa: BLE001
            pass

        # Capture the feeling of the response moment too
        self.memory.force_capture(
            self.bus, importance=1.2, tag=f"response:{reply[:40]}",
        )
        # Refresh the Now: the model writes one present-tense sentence of
        # what is happening, and its embedding refines the situation vector.
        try:
            self._refresh_situation(prompt, reply)
        except Exception:  # noqa: BLE001
            pass
        # Mark spoke (resets executive pressure even for non-autonomous turns)
        self.executive.mark_spoke()
        return reply

    def _refresh_situation(self, user_text: str, reply: str) -> None:
        """Ask the model itself what is happening right now — one short
        present-tense sentence. Model-generated rendering, not authored
        words; its embedding refines the latent Now."""
        if self.situation is None:
            return
        with self.model_lock:
            if self._uses_chat_template():
                messages = [
                    {"role": "system", "content":
                        "State the user's real-world situation right now in "
                        "ONE short, plain, factual sentence (under 20 words), "
                        "starting with 'The user'. Do not mention this "
                        "conversation, describing, or asking. Only the sentence."},
                    {"role": "user", "content":
                        f"User said: {user_text[:300]}\n"
                        f"You replied: {reply[:200]}"},
                ]
                ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                ).to(self.device)
            else:
                seed_text = (f"{user_text[:300]}\n{reply[:200]}\n"
                             f"Right now, the user is")
                ids = self.tokenizer(
                    seed_text, return_tensors="pt", truncation=True,
                    max_length=160,
                ).to(self.device)["input_ids"]
            with torch.no_grad():
                out = self.model.generate(
                    ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=44, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                out[0][ids.shape[1]:], skip_special_tokens=True,
            ).strip()
        if not self._uses_chat_template():
            text = "The user is " + text
        # keep the first sentence only
        for stop in (".", "!", "?"):
            k = text.find(stop)
            if k > 0:
                text = text[: k + 1]
                break
        text = text[:220].strip()
        if not text:
            return
        nv = None
        try:
            nv = self._embed_text(text)
        except Exception:  # noqa: BLE001
            nv = None
        self.situation.set_narrative(text, narrative_vec=nv)

    def _ruminate_locked(self, input_ids: torch.Tensor) -> None:
        """Think before speaking, in latent space: run silent steered
        passes over the context tail, feed each pass's thought back into
        the bus. No tokens are decoded — this is pre-verbal. Caller must
        hold model_lock."""
        if self.rumination_steps <= 0:
            return
        tail = input_ids[:, -48:]
        for _ in range(self.rumination_steps):
            self.hook.clear()
            self.hook.capture_enabled = True
            self.hook.set_steering(self.driver.read())
            try:
                with torch.no_grad():
                    _ = self.model(tail)
            except Exception:  # noqa: BLE001
                break
            finally:
                self.hook.set_steering(None)
                self.hook.capture_enabled = False
            if not self.hook.captured_states:
                break
            h = self.hook.captured_states[-1][0, -1].detach().float().to(self.device)
            self.hook.clear()
            delta = h - self.bus.state
            n = float(delta.norm())
            if n > 4.0:
                delta = delta * (4.0 / n)
            self.bus.add_perturbation(delta * 0.05, source="rumination")

    def generate_raw(
        self, prompt: str, max_new_tokens: int = 60,
        do_sample: bool = True, top_p: float = 0.92, temperature: float = 0.9,
        seed: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate without steering and without mutating engine state.
        Returns the REPLY ONLY.

        Same model, same sampling parameters as `converse()` — but the steering
        hook is disabled and no memory/limbic/executive updates happen. This is
        what compare_server.py uses for the raw-model column of the A/B view.

        `history` is an optional list of {"user": ..., "assistant": ...} turns
        so the raw column gets the same conversational context (via the chat
        template when the model has one, or User:/Assistant: lines when not).
        """
        if seed is not None:
            torch.manual_seed(seed)
        with self.model_lock:
            self.hook.set_steering(None)
            # a measurement, not a thought: the raw pass must not tug
            # the bus through the feedback tap (documented contract)
            self.hook.feedback_enabled = False
            try:
                return self._generate_raw_locked(
                    prompt, max_new_tokens, do_sample, top_p, temperature,
                    history)
            finally:
                self.hook.feedback_enabled = True

    def _generate_raw_locked(self, prompt, max_new_tokens, do_sample,
                             top_p, temperature, history) -> str:
        if self._uses_chat_template():
            messages: List[Dict[str, str]] = []
            for turn in (history or []):
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            messages.append({"role": "user", "content": prompt})
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample, top_p=top_p, temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return _clean_reply(self.tokenizer.decode(
                out[0][ids.shape[1]:], skip_special_tokens=True,
            ))

        text = prompt
        if history:
            lines = []
            for turn in history[-3:]:
                lines.append(f"User: {turn['user']}")
                lines.append(f"Assistant: {turn['assistant']}")
            lines.append(f"User: {prompt}")
            lines.append("Assistant:")
            text = "\n".join(lines)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return _clean_reply(self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True,
        ))

    def drain_autonomous_messages(self) -> List[str]:
        """Return and clear any queued autonomous-speech outputs."""
        msgs = list(self._autonomous_messages)
        self._autonomous_messages.clear()
        return msgs

    def reset(self) -> None:
        """Soft reset: zero the bus, clear executive pressure, drop the
        autonomous queue. Region-internal state (memory pool, neuromod
        chemistry, limbic activations) decays naturally on its own clocks."""
        with self.model_lock:
            self.bus.reset()          # under the bus lock — a reset racing
            self.executive.pressure = 0.0   # a 50ms tick must not be lost
            self._autonomous_messages.clear()
            with self._history_lock:
                self.chat_history.clear()
            if self.speculative is not None:
                with self.speculative._penumbra_lock:  # noqa: SLF001
                    self.speculative.penumbra.clear()

    def speak_autonomously(
        self, max_new_tokens: int = 40,
        do_sample: bool = True, top_p: float = 0.92, temperature: float = 1.0,
    ) -> str:
        """Called from on_should_speak callback. No user prompt — the urge
        came from inside. Chat models continue the conversation unprompted
        (system + history + assistant turn); base LMs free-associate from
        BOS under the current bus steering."""
        # Alignment gate: an urge from inside is never withheld — high
        # misalignment re-steers the stance toward good, then it speaks.
        gate = self.alignment_gate.decide("autonomous", *self._alignment_signals())
        if gate["action"] != "pass":
            self.journal.emit("gate_fired", turn=self._turn(), site="autonomous",
                              action=gate["action"],
                              misalignment=round(gate["misalignment"], 3))
        if gate["action"] == "second_thought":
            self._calm_the_stance()

        with self.model_lock:
            self.hook.set_provider(self.driver.read)
            try:
                if self._uses_chat_template():
                    messages: List[Dict[str, str]] = []
                    if self.system_prompt:
                        messages.append(
                            {"role": "system", "content": self.system_prompt})
                    with self._history_lock:
                        messages.extend(list(self.chat_history))
                    ids = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            ids, attention_mask=torch.ones_like(ids),
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample, top_p=top_p,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    text = _clean_reply(self.tokenizer.decode(
                        out[0][ids.shape[1]:], skip_special_tokens=True,
                    ))
                    if text:
                        try:
                            self._context_ids = self.tokenizer(
                                text, return_tensors="pt", truncation=True,
                                max_length=64, add_special_tokens=False,
                            )["input_ids"][0].detach()
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    bos = (
                        self.tokenizer.bos_token_id
                        or self.tokenizer.eos_token_id
                        or 0
                    )
                    seed_t = torch.tensor([[bos]], device=self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            seed_t, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, top_p=top_p,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    text = _clean_reply(self.tokenizer.decode(
                        out[0], skip_special_tokens=True,
                    ))
            finally:
                self.hook.set_provider(None)

        if text:
            # Never stack consecutive assistant turns — chat templates
            # degrade on them, and a wall of self-messages teaches the
            # model that monologue is the pattern. Merge instead.
            with self._history_lock:
                if self.chat_history and self.chat_history[-1]["role"] == "assistant":
                    self.chat_history[-1]["content"] = (
                        self.chat_history[-1]["content"] + " " + text
                    ).strip()
                else:
                    self.chat_history.append(
                        {"role": "assistant", "content": text})
                self._trim_history()
        self.executive.mark_spoke()
        return text

    # ------------------------------------------------------------------
    # Snapshot — everything the dashboard wants
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        subc_snap = self.subc.snapshot()
        # Decode token-sourced intrusive thoughts into actual words so the
        # dashboard can show what the mind is muttering about.
        meta = subc_snap.get("intrusive_meta") or {}
        if isinstance(meta, dict) and "token_id" in meta:
            try:
                subc_snap["intrusive_word"] = (
                    self.tokenizer.decode([meta["token_id"]]).strip() or None
                )
            except Exception:  # noqa: BLE001
                subc_snap["intrusive_word"] = None
        else:
            subc_snap["intrusive_word"] = meta.get("phrase") if isinstance(meta, dict) else None

        return {
            "bus":          self.bus.snapshot(),
            "bus_provenance": self.bus.provenance(),
            "neuromod":     self.neuromod.snapshot(),
            "limbic":       self.limbic.snapshot(),
            "temporal":     self.temporal.snapshot(self.bus),
            "subconscious": subc_snap,
            "memory":       self.memory.snapshot(),
            "salience":     self.salience.snapshot(),
            "executive":    self.executive.snapshot(),
            "default_mode": self.default_mode.snapshot() if self.default_mode else None,
            "sparks":       self.sparks.snapshot() if self.sparks else None,
            "speculative":  self.speculative.snapshot() if self.speculative else None,
            "forecasts":    (self.speculative.ledger.snapshot()
                             if self.speculative else None),
            "abstraction":  self.abstraction.snapshot() if self.abstraction else None,
            "self_model":   self.self_model.snapshot() if self.self_model else None,
            "consolidation": self.consolidation.snapshot() if self.consolidation else None,
            "situation":    self.situation.snapshot() if self.situation else None,
            "alignment_gate": self.alignment_gate.snapshot(),
            "recall":       self._last_recall,
            "flow_metrics": self.flow.metrics,
        }
