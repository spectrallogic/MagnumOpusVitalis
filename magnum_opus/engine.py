"""
The unified Magnum Opus Vitalis engine.
Integrates all components, manages steering hooks, handles generation,
and provides save/load for persistent state across sessions.
"""

import collections
import json
import os
import random
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from magnum_opus.config import EngineConfig
from magnum_opus.components import (
    InternalTimeSignals,
    MultiSpeedEmotionalState, TemporalEngine, ResidualSteering, ThoughtResidual,
    SubconsciousEngine, MemorySystem, DreamCycle, GrowthManager,
    AlignmentMonitor, CommunicativeDrive, KnowledgeSparks,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEERING HOOK
# ═══════════════════════════════════════════════════════════════════════════

class SteeringHook:
    """
    Hooks into a transformer layer's forward pass to:
    1. Capture hidden states for analysis
    2. Add steering vectors to modify the model's activations
    """

    def __init__(self):
        self.steering_vector: Optional[torch.Tensor] = None
        self.captured_states: List[torch.Tensor] = []
        self.active = False
        self._handle = None

    def set_steering(self, vector: Optional[torch.Tensor]):
        self.steering_vector = vector
        self.active = vector is not None

    def hook_fn(self, module, input, output):
        # Layer outputs vary by architecture:
        #   GPT-2: tuple (hidden_states, presents, ...)
        #   LLaMA/Mistral: plain Tensor, tuple, or ModelOutput dataclass
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)
        hidden_states = output if is_tensor else output[0]

        self.captured_states.append(hidden_states.detach().clone())

        if self.active and self.steering_vector is not None:
            sv = self.steering_vector.to(hidden_states.device).to(hidden_states.dtype)
            modified = hidden_states + sv.unsqueeze(0).unsqueeze(0)

            if is_tensor:
                return modified
            elif is_tuple:
                return (modified,) + output[1:]
            else:
                # ModelOutput dataclass — replace first field
                keys = list(output.keys())
                output[keys[0]] = modified
                return output
        return output

    def clear(self):
        self.captured_states = []

    def attach(self, model, target_layer: int):
        """Attach hook to the correct layer, handling different model architectures."""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style
            layer = model.transformer.h[target_layer]
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA / Mistral style
            layer = model.model.layers[target_layer]
        else:
            raise ValueError(
                f"Unknown model architecture. Cannot find layer {target_layer}. "
                f"Model type: {type(model).__name__}"
            )
        self._handle = layer.register_forward_hook(self.hook_fn)
        return self


# ═══════════════════════════════════════════════════════════════════════════
# EMOTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def measure_projections(
    hidden_states: List[torch.Tensor],
    emotion_vectors: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Measure emotion vector projections in captured hidden states."""
    if not hidden_states:
        return {}
    last = hidden_states[-1].mean(dim=1).squeeze(0).float()
    return {
        name: torch.dot(last.to(vec.device), vec.float()).item()
        for name, vec in emotion_vectors.items()
    }


def detect_emotional_content(
    text: str,
    keywords: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """
    Simple keyword-based emotion detection from input text.
    Used to generate automatic emotional stimulus from user messages.
    Not a replacement for latent space analysis, but a fast first pass.
    """
    if keywords is None:
        keywords = {
            "joy": ["happy", "great", "wonderful", "amazing", "love", "thank", "awesome",
                     "excellent", "fantastic", "perfect", "beautiful", "glad", "excited"],
            "sadness": ["sad", "sorry", "unfortunately", "loss", "miss", "grief", "lonely",
                        "disappointed", "heartbroken", "depressed", "melancholy"],
            "anger": ["angry", "furious", "hate", "outraged", "frustrated", "annoyed",
                      "infuriating", "unacceptable", "ridiculous"],
            "fear": ["afraid", "scared", "terrified", "worried", "anxious", "panic",
                     "nightmare", "danger", "threat", "alarming"],
            "curious": ["curious", "interesting", "wonder", "how", "why", "what if",
                        "fascinating", "intriguing", "explore", "discover", "question"],
            "desperate": ["urgent", "emergency", "help", "please", "need", "critical",
                          "failing", "broken", "deadline", "crisis", "asap"],
            "calm": ["calm", "peaceful", "relaxed", "steady", "gentle", "quiet",
                     "serene", "tranquil", "easy", "patient"],
            "surprise": ["wow", "unexpected", "shocking", "incredible", "unbelievable",
                         "suddenly", "never expected"],
            "trust": ["trust", "reliable", "honest", "faithful", "loyal", "depend",
                      "count on", "believe in"],
            "disgust": ["disgusting", "revolting", "appalling", "repulsive", "vile",
                        "sickening", "abhorrent"],
        }

    text_lower = text.lower()
    scores = {}
    for emotion, words in keywords.items():
        count = sum(1 for w in words if w in text_lower)
        if count > 0:
            scores[emotion] = min(count * 0.5, 2.0)
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# THE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MagnumOpusEngine:
    """
    The complete Magnum Opus Vitalis engine.

    Integrates all 10 subsystems into a single interface:
    - Multi-speed emotional dynamics
    - Temporal awareness (real clock)
    - Residual steering (continuity)
    - Subconscious (structured noise + goals)
    - Memory (latent traces, reconstructive)
    - Dream cycle (offline consolidation)
    - Growth monitoring (patient expansion)
    - Alignment monitoring (mutualistic symbiosis)

    Usage:
        engine = MagnumOpusEngine(model, tokenizer, vectors, device=device)
        response = engine.converse("Hello!")
        engine.dream()  # Run during idle
        engine.save("state.json")

    Or with a saved profile:
        from magnum_opus import load_profile
        profile = load_profile("gpt2")
        engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)
    """

    def __init__(
        self,
        model,
        tokenizer,
        emotion_vectors: Optional[Dict[str, torch.Tensor]] = None,
        target_layer: Optional[int] = None,
        config: Optional[EngineConfig] = None,
        device: str = "cpu",
        profile: Optional["ModelProfile"] = None,
    ):
        # Accept either explicit vectors or a saved profile
        if profile is not None:
            from magnum_opus.profile import ModelProfile
            if not isinstance(profile, ModelProfile):
                raise TypeError("profile must be a ModelProfile instance")
            emotion_vectors = profile.vectors
            target_layer = profile.metadata.target_layer
        elif emotion_vectors is None:
            raise ValueError("Must provide either emotion_vectors or profile")

        self.model = model
        self.tokenizer = tokenizer
        self.emotion_vectors = emotion_vectors
        self.config = config or EngineConfig()
        self.device = device

        # Detect model architecture
        if hasattr(model.config, "n_layer"):
            n_layers = model.config.n_layer
            self.hidden_dim = model.config.n_embd
        elif hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
            self.hidden_dim = model.config.hidden_size
        else:
            raise ValueError("Cannot detect model architecture")

        self.target_layer = target_layer if target_layer is not None else n_layers // 2

        # Derive the emotion interaction matrix from the cosine geometry of the
        # extracted vectors themselves. The frozen model's own semantics decide
        # which feelings amplify or suppress each other — not an authored list.
        self._derived_interactions = self._derive_emotion_interactions(emotion_vectors)

        # Initialize all subsystems
        temporal_vecs = {k: v for k, v in emotion_vectors.items() if k.startswith("temporal_")}

        self.emotional_state = MultiSpeedEmotionalState(
            list(emotion_vectors.keys()),
            interactions=self._derived_interactions,
        )
        self.temporal = TemporalEngine(temporal_vecs, self.config)
        self.residual = ResidualSteering(
            self.hidden_dim, self.config.residual_decay, self.config.residual_max_norm,
        )
        self.thought_residual = ThoughtResidual(
            self.hidden_dim, decay=self.config.thought_residual_decay,
        )
        self.subconscious = SubconsciousEngine(
            emotion_vectors, self.hidden_dim,
            self.config.subconscious_amplitude, self.config.subconscious_goal_momentum,
            device,
        )
        self.memory = MemorySystem(self.hidden_dim, self.config, device)
        self.growth = GrowthManager(self.config)
        self.alignment = AlignmentMonitor(emotion_vectors, self.config)
        self.communicative_drive = CommunicativeDrive(emotion_vectors, self.hidden_dim, device)
        self._dream_cycle = DreamCycle(
            model, tokenizer, self.memory, self.subconscious,
            self.emotional_state, emotion_vectors, self.target_layer,
            self.config, device,
        )

        # Steering hook
        self.hook = SteeringHook().attach(model, self.target_layer)

        # Knowledge Sparks — periodic intrusive thoughts sampled from the
        # model's own vocabulary embedding matrix. Wired after the hook so
        # sparks that trigger forward passes flow through steering cleanly.
        self.knowledge_sparks = KnowledgeSparks(
            model, tokenizer, self.hidden_dim,
            spark_strength=self.config.knowledge_spark_strength,
            candidate_pool=self.config.knowledge_spark_candidate_pool,
        )
        self.active_spark_vector = torch.zeros(self.hidden_dim, device="cpu")
        self.last_spark_event: Optional[Dict[str, Any]] = None

        # Telemetry for the UI — last speculation tree summary, last drift projections
        self.last_speculation_summary: Dict[str, Any] = {
            "depth": 0, "beam": 0, "best_resonance": 0.0, "decoded_tips": [],
        }
        self.last_drift_projections: Dict[str, float] = {}

        # State
        self.step_count = 0
        self.conversation_history: List[Dict[str, str]] = []
        self._dream_thread: Optional[threading.Thread] = None
        self.autonomous_messages: List[Dict[str, Any]] = []

        # Rolling log of idle events (spontaneous recalls, speculations, drift spikes)
        # Visible proof in the UI that the engine is alive when no one's talking
        self.idle_events: collections.deque = collections.deque(maxlen=20)

        # Cached internal signals (updated each step)
        self._current_signals = InternalTimeSignals()

        # Thread safety for heartbeat
        self._lock = threading.Lock()

        # Idle tracking for auto-dream (wall clock OK — this is scheduling, not perception)
        self._last_activity = time.time()

    # ───────────────────────────────────────────────────────────────────
    # LATENT DERIVATION HELPERS
    # Quantities computed from the model's own vector geometry,
    # replacing the hand-authored formulas of earlier revisions.
    # ───────────────────────────────────────────────────────────────────

    def _derive_emotion_interactions(
        self, emotion_vectors: Dict[str, torch.Tensor],
    ) -> Dict[Tuple[str, str], float]:
        """Build an interaction matrix from the cosine geometry of the
        extracted emotion vectors. Aligned vectors amplify; opposed suppress;
        near-orthogonal do nothing. A 0.1 dead-zone avoids noise dominating."""
        interactions: Dict[Tuple[str, str], float] = {}
        names = [n for n in emotion_vectors if not n.startswith("temporal_")]
        for a in names:
            va = emotion_vectors[a].cpu().float()
            na = va.norm()
            for b in names:
                if a == b:
                    continue
                vb = emotion_vectors[b].cpu().float()
                nb = vb.norm()
                if na < 1e-6 or nb < 1e-6:
                    sim = 0.0
                else:
                    sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                if abs(sim) < 0.1:
                    interactions[(a, b)] = 0.0
                else:
                    # Scale [-1, 1] → [-0.5, 0.5]
                    interactions[(a, b)] = float(sim * 0.5)
        return interactions

    def _emotion_blend_vector(self) -> torch.Tensor:
        """Weighted sum of emotion vectors, weighted by the current blended
        emotional state. Used as a bias direction for knowledge-spark sampling
        and for spark-seeded speculation candidates."""
        blended = self.emotional_state.get_blended()
        out = torch.zeros(self.hidden_dim)
        for emotion, w in blended.items():
            if emotion in self.emotion_vectors and abs(w) > 1e-4:
                out = out + float(w) * self.emotion_vectors[emotion].cpu().float()
        return out

    def _arousal_from_emotions(self) -> float:
        """Mean absolute magnitude of the blended emotional state, clipped to
        [0, 1]. Modulates sampling temperature for self-started speech."""
        blended = self.emotional_state.get_blended()
        if not blended:
            return 0.0
        mag = sum(abs(float(v)) for v in blended.values()) / max(len(blended), 1)
        return float(min(1.0, mag))

    def _latent_seeded_generate(self, max_tokens: Optional[int] = None) -> str:
        """Self-start a generation with no scripted prefix. Runs a single
        forward pass on the BOS token with full steering active, samples the
        first token from that distribution at a temperature modulated by
        emotional arousal, then continues generation with steering still on.

        The opening token is chosen by the model-under-steering itself, not
        by a lookup table — the inner monologue begins wherever the current
        soul-state wants it to begin."""
        max_tokens = max_tokens or self.config.default_max_tokens
        bos = (self.tokenizer.bos_token_id
               or self.tokenizer.eos_token_id
               or 0)
        seed_ids = torch.tensor([[bos]], device=self.device)

        steering = self.compute_steering_vector()
        self.hook.clear()
        self.hook.set_steering(steering)

        try:
            with torch.no_grad():
                fwd = self.model(seed_ids)
                logits = fwd.logits[0, -1, :].float()

            arousal = self._arousal_from_emotions()
            temp = 0.7 + arousal * 0.6  # 0.7 .. 1.3
            probs = F.softmax(logits / max(temp, 1e-3), dim=-1)
            k = min(20, probs.numel())
            top_probs, top_idx = probs.topk(k)
            top_probs = top_probs / top_probs.sum().clamp(min=1e-8)
            choice = int(torch.multinomial(top_probs, 1).item())
            first_id = int(top_idx[choice].item())

            prefix = torch.cat(
                [seed_ids, torch.tensor([[first_id]], device=self.device)], dim=1,
            )

            with torch.no_grad():
                gen = self.model.generate(
                    prefix,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=self.config.default_temperature,
                    top_p=self.config.default_top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            full_ids = gen[0, seed_ids.shape[1]:]
            text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        finally:
            self.hook.set_steering(None)

        return text.strip()

    # ───────────────────────────────────────────────────────────────────
    # INTERNAL TIME PERCEPTION
    # ───────────────────────────────────────────────────────────────────

    def _gather_internal_signals(self) -> InternalTimeSignals:
        """Snapshot of internal state for subjective time perception."""
        return InternalTimeSignals(
            residual_norm=self.residual.norm(),
            residual_norm_at_last_interaction=self.temporal.residual_norm_at_interaction,
            emotional_distance=self.communicative_drive.emotional_distance,
            interaction_freshness=self.communicative_drive.interaction_freshness,
            memory_avg_importance=self.memory.avg_importance(),
            memory_avg_importance_at_last_interaction=self.temporal.importance_at_interaction,
            steps_since_interaction=self.temporal.steps_since_interaction,
        )

    # ───────────────────────────────────────────────────────────────────
    # CORE: Compute the full steering vector
    # ───────────────────────────────────────────────────────────────────

    def compute_steering_vector(self) -> torch.Tensor:
        """
        Compute the FULL steering intervention from all active systems.
        This is the heart of the engine.
        """
        # 1. Emotional component (blended fast/medium/slow)
        emo_weights = self.emotional_state.get_blended()
        emo_vec = torch.zeros(self.hidden_dim)
        for emotion, weight in emo_weights.items():
            if emotion in self.emotion_vectors:
                emo_vec = emo_vec + weight * self.emotion_vectors[emotion].cpu().float()

        # 2. Temporal component
        temporal_vec = self.temporal.compute_steering(self._current_signals)
        if temporal_vec is not None:
            temporal_vec = temporal_vec.cpu().float()
        else:
            temporal_vec = torch.zeros(self.hidden_dim)

        # 3. Subconscious component
        sub_vec = self.subconscious.get_steering(emo_weights).cpu().float()

        # 4. Thought residual — echo of what the model itself felt in the past.
        #    This is how the engine feels its own history carrying forward.
        thought_vec = self.thought_residual.get()

        # 5. Active knowledge spark — a recently-fired intrusive concept,
        #    direction-normalized, fading each tick. Gives the steering blend a
        #    vocabulary-anchored flavor for as long as the spark survives.
        spark_vec = self.active_spark_vector.cpu().float()

        # 6. Combine
        combined = (
            emo_vec * self.config.steering_strength
            + temporal_vec * 1.0
            + sub_vec * 0.5
            + thought_vec * self.config.thought_residual_strength
            + spark_vec * self.config.knowledge_spark_steering_weight
        )

        # 7. Residual (temporal continuity of our own steering commands)
        steered = self.residual.step(combined)

        return steered.to(self.device)

    # ───────────────────────────────────────────────────────────────────
    # CORE: Advance engine state by one step
    # ───────────────────────────────────────────────────────────────────

    def _step(self, stimulus: Optional[Dict[str, float]] = None,
              dt: Optional[float] = None, is_interaction: bool = True):
        """Internal state advancement."""
        # Gather internal signals for time perception
        self._current_signals = self._gather_internal_signals()

        if dt is None:
            dt = self.temporal.time_factor(self._current_signals)

        if is_interaction:
            self.temporal.mark_interaction(self._current_signals)
        self.temporal.tick()

        if stimulus:
            for emotion, intensity in stimulus.items():
                self.emotional_state.stimulate(emotion, intensity)

        self.emotional_state.decay_step(dt)
        self.memory.decay_all()
        self.step_count += 1
        self._last_activity = time.time()

    # ───────────────────────────────────────────────────────────────────
    # GENERATION
    # ───────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_tokens: Optional[int] = None,
                  temperature: Optional[float] = None) -> Tuple[str, List[torch.Tensor]]:
        """Raw generation with steering applied."""
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature or self.config.default_temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Apply steering
        steering = self.compute_steering_vector()
        self.hook.set_steering(steering)
        self.hook.clear()

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=self.config.default_top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        states = self.hook.captured_states

        # Clear hook
        self.hook.set_steering(None)

        return text, states

    def _post_generation(self, prompt: str, text: str, states: List[torch.Tensor]):
        """Post-generation processing: memory, subconscious, alignment."""
        if not states:
            return

        last_state = states[-1].mean(dim=1).squeeze(0).float()

        # Imprint into thought residual — the model's own felt state echoes forward
        self.thought_residual.imprint(last_state)

        # Encode memory
        self.memory.encode(
            activation=last_state,
            emotional_state=self.emotional_state.get_blended(),
            surprise=self.subconscious.resonance_history[-1]
                    if self.subconscious.resonance_history else 0.0,
            goal_relevance=self.subconscious.goal_strength,
            text_summary=prompt[:80],
        )

        # Memory recall and emotional coloring
        recalled = self.memory.recall(last_state, top_k=3)
        coloring = self.memory.emotional_coloring(recalled)
        for emotion, intensity in coloring.items():
            self.emotional_state.stimulate(emotion, intensity * 0.5)

        # Subconscious evaluation
        noise = self.subconscious.generate_noise(self.emotional_state.get_blended())
        resonance = self.subconscious.evaluate_resonance(noise, states)
        self.subconscious.update_goals(noise, resonance)

        # Alignment check
        self.alignment.measure(states, self.emotional_state.get_blended())

    # ───────────────────────────────────────────────────────────────────
    # PUBLIC API: Conversation
    # ───────────────────────────────────────────────────────────────────

    def converse(self, user_input: str, max_tokens: Optional[int] = None,
                 system_prompt: Optional[str] = None) -> str:
        """
        Process user input and generate a response with the full engine active.
        Automatically detects emotional content, updates engine state,
        and checks for user preference signals about communication frequency.
        """
        with self._lock:
            # Notify communicative drive that user spoke
            self.communicative_drive.user_spoke(self.emotional_state.get_blended())

            # Check if user is adjusting talk preference
            pref_signal = self.communicative_drive.detect_preference_signal(user_input)
            if pref_signal is not None:
                self.communicative_drive.adjust_preference(pref_signal)

            # Detect emotion from input text
            detected = detect_emotional_content(user_input)

            # Step the engine with detected emotions
            self._step(stimulus=detected if detected else None)

            # Build prompt
            if system_prompt:
                prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"

            # Generate
            text, states = self._generate(prompt, max_tokens=max_tokens)

            # Post-process
            self._post_generation(user_input, text, states)

            # Extract just the response
            if "Assistant:" in text:
                response = text.split("Assistant:")[-1].strip()
            else:
                response = text[len(prompt):].strip()

            # Log conversation
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": time.time(),
                "emotional_state": self.emotional_state.get_blended(),
            })

            return response

    def check_autonomous_urge(self) -> bool:
        """Check if the system wants to speak on its own."""
        return self.communicative_drive.should_speak()

    def speak_autonomously(self) -> Optional[str]:
        """
        Generate a message from the system's own internal state. No scripted
        seed phrases: the engine samples its own first token from the BOS
        distribution under current steering, then continues from there. The
        opening words are whatever the model-under-soul actually wanted to say.
        """
        with self._lock:
            blended = self.emotional_state.get_blended()

            # Step the engine (autonomous activity, not a user interaction)
            self._step(dt=0.5, is_interaction=False)

            # Latent-seeded generation — no THOUGHT_SEEDS, no English template.
            response = self._latent_seeded_generate(max_tokens=80)

            # Run post-generation processing against the captured states from
            # the continuation pass. The captured_states belong to the last
            # forward pass, so they still reflect what the model felt while
            # producing this autonomous utterance.
            if self.hook.captured_states:
                states = list(self.hook.captured_states)
                self._post_generation("[autonomous]", response, states)

            # Mark that the system spoke
            self.communicative_drive.spoke(current_residual_norm=self.residual.norm())

            # Log
            self.conversation_history.append({
                "user": "[autonomous]",
                "assistant": response,
                "timestamp": time.time(),
                "emotional_state": blended,
                "autonomous": True,
                "latent_seeded": True,
            })

            # Queue for dashboard
            self.autonomous_messages.append({
                "message": response,
                "timestamp": time.time(),
                "emotional_state": blended,
                "pressure_at_trigger": self.communicative_drive.pressure,
                "latent_seeded": True,
            })

            return response

    def tick(self):
        """One heartbeat — advance ALL internal state without user input.
        This is the engine's continuous life: emotions decay and drift,
        memories occasionally reactivate, thought residual fades, the
        subconscious speculates, communicative pressure builds.
        Called by the heartbeat thread between interactions."""
        with self._lock:
            self._current_signals = self._gather_internal_signals()
            blended = self.emotional_state.get_blended()

            # Emotions decay continuously (toward homeostatic baseline)
            self.emotional_state.decay_step(self.config.idle_emotional_decay_dt)

            # Stochastic drift — living tissue, not chaos
            self._apply_idle_drift()

            # Possible spontaneous memory recall (re-experience at full strength)
            self._maybe_spontaneous_recall()

            # Thought residual (echo of past model state) slowly fades
            self.thought_residual.step_decay()

            # Knowledge sparks — intrusive thoughts sampled from the model's
            # own vocabulary. Emotional blend biases which concepts surface.
            if random.random() < self.config.knowledge_spark_probability:
                bias = self._emotion_blend_vector()
                result = self.knowledge_sparks.fire(bias, step=self.step_count)
                self.active_spark_vector = result["vector"]
                self.last_spark_event = result["event"]
                self.idle_events.append(result["event"])
            # Decay the active spark vector each tick — a spark fades into
            # the background rather than being cut off abruptly.
            self.active_spark_vector = (
                self.active_spark_vector * self.config.knowledge_spark_decay
            )

            # Communicative drive (pressure to speak autonomously)
            self.communicative_drive.tick(
                emotional_state=blended,
                subconscious_goal_strength=self.subconscious.goal_strength,
                residual_norm=self.residual.norm(),
                dt=0.5,
                residual_vec=self.residual.residual,
            )

            # Temporal perception ticks forward
            self.temporal.tick()

            # Speculative subconscious — actually run short forward passes
            # to try candidate thoughts, every N ticks
            if (self.step_count > 0
                    and self.step_count % self.config.speculative_cadence_ticks == 0):
                self._speculate()

            # Memory slow decay
            self.memory.decay_all()

            self.step_count += 1

    # ───────────────────────────────────────────────────────────────────
    # IDLE COGNITIVE DYNAMICS — what happens when nobody's talking
    # ───────────────────────────────────────────────────────────────────

    def _apply_idle_drift(self):
        """Latent-native idle drift. Runs a silent one-token forward pass with
        the current steering active, takes the model's hidden-state response
        relative to the accumulated thought residual, and projects that delta
        onto each emotion vector. The projections become drift signals — the
        emotional state moves in whatever direction the model actually gravitates
        under its current soul, not uniform noise. Falls back to no-op if the
        forward pass produces nothing."""
        amp = self.config.idle_drift_amplitude
        if amp <= 0:
            return
        try:
            bos = (self.tokenizer.bos_token_id
                   or self.tokenizer.eos_token_id
                   or 0)
            seed = torch.tensor([[bos]], device=self.device)
            self.hook.clear()
            self.hook.set_steering(self.compute_steering_vector())
            with torch.no_grad():
                _ = self.model(seed)
        except Exception:
            self.hook.set_steering(None)
            self.last_drift_projections = {}
            return
        finally:
            self.hook.set_steering(None)

        if not self.hook.captured_states:
            self.last_drift_projections = {}
            return

        state = self.hook.captured_states[-1].mean(dim=(0, 1)).detach().cpu().float()
        prev = self.thought_residual.get().detach().cpu().float()
        delta = state - prev if prev.norm() > 1e-6 else state
        if delta.norm() < 1e-6:
            self.last_drift_projections = {}
            return
        delta = delta / delta.norm()

        projections: Dict[str, float] = {}
        for emo, vec in self.emotion_vectors.items():
            if emo.startswith("temporal_"):
                continue
            v = vec.cpu().float()
            if v.norm() < 1e-6:
                continue
            proj = F.cosine_similarity(delta.unsqueeze(0), v.unsqueeze(0)).item()
            projections[emo] = float(proj)
            self.emotional_state.stimulate(emo, proj * amp)

        self.last_drift_projections = {
            k: round(v, 4) for k, v in projections.items()
        }

        # Log a drift event occasionally when the model's response is strongly
        # directional — the most "alive" drift moments, worth surfacing in UI.
        if projections:
            peak_emo, peak_val = max(projections.items(), key=lambda kv: abs(kv[1]))
            if abs(peak_val) > 0.3 and random.random() < 0.1:
                self.idle_events.append({
                    "type": "drift",
                    "emotion": peak_emo,
                    "projection": round(peak_val, 4),
                    "step": self.step_count,
                })

    def _maybe_spontaneous_recall(self):
        """Probabilistic re-experience of a past memory during idle.
        The reactivated emotional coloring reaches higher strength than
        post-generation recall — this is closer to true remembering."""
        if random.random() > self.config.idle_recall_probability:
            return
        if not self.memory.memories:
            return
        mems = self.memory.memories
        weights = [max(float(m.importance), 0.0) for m in mems]
        total = sum(weights)
        if total < 1e-6:
            return
        r = random.random() * total
        acc = 0.0
        chosen = mems[-1]
        for m, w in zip(mems, weights):
            acc += w
            if acc >= r:
                chosen = m
                break

        # Reactivate its emotional coloring
        strength = self.config.idle_recall_coloring_strength
        for emo, intensity in chosen.emotional_state.items():
            self.emotional_state.stimulate(emo, intensity * strength)
        chosen.access_count += 1

        # Re-imprint its activation as a "remembered thought"
        if getattr(chosen, "activation", None) is not None:
            try:
                self.thought_residual.imprint(chosen.activation.to('cpu').float())
            except Exception:
                pass

        self.idle_events.append({
            "type": "recall",
            "text": getattr(chosen, "text_summary", "") or "",
            "step": self.step_count,
        })

    def _sample_speculation_candidate(self, parent_vec: torch.Tensor
                                      ) -> torch.Tensor:
        """Build a candidate steering vector for one branch of the speculation
        tree. Mixes the parent's direction, a perturbed emotional blend, a
        goal pull, and — with probability `speculative_spark_mix_probability` —
        a knowledge-spark embedding so the subconscious can chase genuine
        vocabulary-anchored concepts, not only emotion noise."""
        cand = parent_vec.detach().cpu().float().clone()
        blended = self.emotional_state.get_blended()
        for emo, w in blended.items():
            if emo in self.emotion_vectors:
                perturbed = w + (random.random() - 0.5) * 0.3
                cand = cand + perturbed * self.emotion_vectors[emo].cpu().float()
        if self.subconscious.goal_vector_nonzero():
            cand = cand + 0.4 * self.subconscious.goal_vector.detach().cpu().float()
        if random.random() < self.config.speculative_spark_mix_probability:
            _, spark_vec = self.knowledge_sparks.sample_spark(
                emotional_bias=self._emotion_blend_vector(),
            )
            cand = cand + 0.5 * spark_vec.detach().cpu().float()
        return cand

    def _score_trajectory(self, final_state: torch.Tensor) -> float:
        """Latent resonance: goal alignment plus a novelty bonus measured as
        cosine distance from the current thought residual. The engine is
        rewarded for finding paths toward the goal but also for exploring
        directions it has not recently occupied."""
        goal = self.subconscious.goal_vector
        if goal is None:
            goal_align = 0.0
        else:
            g = goal.detach().cpu().float()
            if g.norm() < 1e-6 or final_state.norm() < 1e-6:
                goal_align = 0.0
            else:
                goal_align = F.cosine_similarity(
                    final_state.unsqueeze(0), g.unsqueeze(0),
                ).item()
        tr = self.thought_residual.get().detach().cpu().float()
        if tr.norm() > 1e-6 and final_state.norm() > 1e-6:
            novelty = 1.0 - F.cosine_similarity(
                final_state.unsqueeze(0), tr.unsqueeze(0),
            ).item()
        else:
            novelty = 0.0
        return float(goal_align + self.config.speculative_novelty_weight * novelty)

    def _speculate(self):
        """Tree-search speculation. At each depth, surviving nodes spawn new
        branches whose steering vectors can include knowledge sparks. Each
        branch is scored by cumulative resonance (goal alignment + novelty),
        and only the top `speculative_beam_width` survive to the next depth.
        The best trajectory's final direction is adopted as the subconscious
        goal nudge; each step along the winning path imprints onto thought
        residual with depth-decayed strength."""
        if not self.subconscious.goal_vector_nonzero():
            return

        rounds = max(1, int(self.config.speculative_rounds))
        horizon = max(1, int(self.config.speculative_horizon_tokens))
        depth = max(1, int(self.config.speculative_tree_depth))
        branch = max(1, int(self.config.speculative_branch_factor))
        beam = max(1, int(self.config.speculative_beam_width))

        try:
            seed_ids = self.tokenizer.encode(".", return_tensors="pt").to(self.device)
        except Exception:
            return

        frontier: List[Dict[str, Any]] = [{
            "cumulative": 0.0,
            "parent_vec": torch.zeros(self.hidden_dim),
            "prefix_ids": seed_ids,
            "trajectory_states": [],
            "decoded_tail": "",
        }]

        for d in range(depth):
            next_frontier: List[Dict[str, Any]] = []
            children_per_node = rounds if d == 0 else branch
            for node in frontier:
                for _ in range(children_per_node):
                    cand = self._sample_speculation_candidate(node["parent_vec"])
                    self.hook.clear()
                    try:
                        self.hook.set_steering(cand.to(self.device))
                        with torch.no_grad():
                            out = self.model.generate(
                                node["prefix_ids"],
                                max_new_tokens=horizon,
                                do_sample=True,
                                temperature=1.0,
                                top_p=self.config.default_top_p,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                    except Exception:
                        self.hook.set_steering(None)
                        continue
                    self.hook.set_steering(None)

                    if not self.hook.captured_states:
                        continue
                    final = self.hook.captured_states[-1].mean(
                        dim=(0, 1),
                    ).detach().cpu().float()
                    if final.norm() < 1e-6:
                        continue
                    score = self._score_trajectory(final)
                    # Decode just the new tail for a human-readable UI preview
                    try:
                        new_tail = self.tokenizer.decode(
                            out[0, node["prefix_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                    except Exception:
                        new_tail = ""
                    next_frontier.append({
                        "cumulative": node["cumulative"] + score,
                        "parent_vec": cand,
                        "prefix_ids": out,
                        "trajectory_states": node["trajectory_states"] + [final],
                        "decoded_tail": (node["decoded_tail"] + " " + new_tail).strip(),
                    })
            if not next_frontier:
                return
            next_frontier.sort(key=lambda n: n["cumulative"], reverse=True)
            frontier = next_frontier[:beam]

        if not frontier:
            return
        best = frontier[0]
        best_resonance = best["cumulative"] / max(1, depth)
        self.subconscious.adopt_speculation(best["parent_vec"], best_resonance)
        # Imprint the whole trajectory with depth-decayed strength — the near
        # future feels more real than the far future.
        for i, state in enumerate(best["trajectory_states"]):
            fade = 0.5 ** i
            self.thought_residual.imprint(state * 0.5 * fade)

        decoded_tips = [n["decoded_tail"] for n in frontier[:beam]]
        self.last_speculation_summary = {
            "depth": depth,
            "beam": beam,
            "branches_explored": len(frontier) * max(1, branch),
            "best_resonance": round(float(best_resonance), 4),
            "decoded_tips": decoded_tips,
            "step": self.step_count,
        }
        self.idle_events.append({
            "type": "speculation",
            "depth": depth,
            "best_resonance": round(float(best_resonance), 4),
            "decoded_tip": decoded_tips[0] if decoded_tips else "",
            "step": self.step_count,
        })

    def get_autonomous_messages(self) -> List[Dict[str, Any]]:
        """Drain queued autonomous messages for the UI."""
        with self._lock:
            msgs = list(self.autonomous_messages)
            self.autonomous_messages.clear()
            return msgs

    def get_idle_events(self) -> List[Dict[str, Any]]:
        """Snapshot (non-destructive) of recent idle events for the UI."""
        with self._lock:
            return list(self.idle_events)

    def generate_raw(self, prompt: str, max_tokens: Optional[int] = None,
                     stimulus: Optional[Dict[str, float]] = None) -> str:
        """Generate with explicit emotional stimulus. No auto-detection."""
        self._step(stimulus=stimulus)
        text, states = self._generate(prompt, max_tokens=max_tokens)
        self._post_generation(prompt, text, states)
        return text

    def generate_without_steering(self, prompt: str, max_tokens: Optional[int] = None,
                                  temperature: Optional[float] = None) -> str:
        """
        Generate from the raw model with NO steering, NO state changes.
        Used for A/B comparison: same model, same prompt, no engine influence.
        """
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature or self.config.default_temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Ensure hook is off
        self.hook.set_steering(None)
        self.hook.clear()

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=self.config.default_top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.hook.clear()
        return text

    # ───────────────────────────────────────────────────────────────────
    # PUBLIC API: Stimulate emotions manually
    # ───────────────────────────────────────────────────────────────────

    def stimulate(self, emotions: Dict[str, float]):
        """Manually stimulate emotions without generating text."""
        self._step(stimulus=emotions)

    # ───────────────────────────────────────────────────────────────────
    # PUBLIC API: Dream cycle
    # ───────────────────────────────────────────────────────────────────

    def dream(self, verbose: bool = True) -> Dict[str, Any]:
        """Run a dream cycle (synchronous)."""
        with self._lock:
            return self._dream_cycle.run(verbose=verbose)

    def dream_async(self):
        """Start a dream cycle in a background thread."""
        if self._dream_thread and self._dream_thread.is_alive():
            return  # Already dreaming
        self._dream_thread = threading.Thread(
            target=self._dream_cycle.run, kwargs={"verbose": False}, daemon=True,
        )
        self._dream_thread.start()

    def is_dreaming(self) -> bool:
        return self._dream_thread is not None and self._dream_thread.is_alive()

    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self._last_activity

    # ───────────────────────────────────────────────────────────────────
    # PUBLIC API: Status and monitoring
    # ───────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full engine status for monitoring — all subsystems visible."""
        signals = self._current_signals

        # Project residual vector onto emotion vectors: what does continuity "feel like"?
        residual_feelings = {}
        residual_vec = self.residual.residual
        for name, vec in self.emotion_vectors.items():
            if not name.startswith("temporal_"):
                proj = torch.dot(residual_vec.cpu().float(), vec.cpu().float()).item()
                residual_feelings[name] = round(proj, 4)

        # Project subconscious goal onto emotion vectors: what does the goal "feel like"?
        goal_feelings = {}
        for name, vec in self.emotion_vectors.items():
            if not name.startswith("temporal_"):
                proj = torch.dot(self.subconscious.goal_vector.cpu().float(), vec.cpu().float()).item()
                goal_feelings[name] = round(proj, 4)

        # Top memories by importance
        sorted_memories = sorted(self.memory.memories, key=lambda x: x.importance, reverse=True)
        memory_list = [
            {
                "id": m.id,
                "text": m.text_summary,
                "importance": round(m.importance, 3),
                "emotional_coloring": {
                    k: round(v, 2) for k, v in m.emotional_state.items() if abs(v) > 0.05
                },
                "access_count": m.access_count,
                "connections": m.connections,
            }
            for m in sorted_memories[:20]
        ]

        # Thought residual — what the model was just "thinking" projected onto emotions
        thought_feelings = self.thought_residual.project_onto(self.emotion_vectors)

        return {
            "step": self.step_count,
            "emotional_state": self.emotional_state.snapshot(),
            "temporal": self.temporal.status(signals),
            "residual": {
                "norm": self.residual.norm(),
                "feelings": residual_feelings,
            },
            "thought_residual": {
                "norm": self.thought_residual.norm(),
                "feelings": thought_feelings,
                "trace_count": self.thought_residual.trace_count,
            },
            "subconscious": {
                **self.subconscious.status(),
                "goal_feelings": goal_feelings,
            },
            "memory": {
                **self.memory.status(),
                "memories": memory_list,
            },
            "growth": self.growth.status(),
            "alignment": self.alignment.status(),
            "communicative_drive": self.communicative_drive.status(),
            "knowledge_sparks": {
                **self.knowledge_sparks.status(),
                "active_vector_norm": round(
                    float(self.active_spark_vector.norm()), 4,
                ),
                "last_event": self.last_spark_event,
            },
            "speculation_last": self.last_speculation_summary,
            "drift_last": {
                "per_emotion_projections": self.last_drift_projections,
            },
            "idle_events": list(self.idle_events),
            "conversation_turns": len(self.conversation_history),
            "dream_cycles": self._dream_cycle.cycle_count,
            "idle_seconds": self.idle_seconds(),
        }

    def alignment_health(self) -> Dict[str, Any]:
        """Quick alignment health check."""
        return self.alignment.check_drift()

    # ───────────────────────────────────────────────────────────────────
    # PUBLIC API: Persistence
    # ───────────────────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save complete engine state to JSON."""
        state = {
            "version": "1.1.0",
            "timestamp": time.time(),
            "step_count": self.step_count,
            "config": {
                "steering_strength": self.config.steering_strength,
                "residual_decay": self.config.residual_decay,
                "subconscious_amplitude": self.config.subconscious_amplitude,
            },
            "emotional_state": self.emotional_state.to_dict(),
            "temporal": self.temporal.to_dict(),
            "residual": self.residual.to_dict(),
            "subconscious": self.subconscious.to_dict(),
            "memory": self.memory.to_dict(),
            "growth": self.growth.to_dict(),
            "alignment": self.alignment.to_dict(),
            "communicative_drive": self.communicative_drive.to_dict(),
            "conversation_history": self.conversation_history[-50:],
            "dream_count": self._dream_cycle.cycle_count,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"  Engine state saved to {filepath}")
        print(f"    {self.step_count} steps, {len(self.memory.memories)} memories, "
              f"{self._dream_cycle.cycle_count} dream cycles")

    def load(self, filepath: str):
        """Load engine state from JSON."""
        if not os.path.exists(filepath):
            print(f"  No saved state at {filepath}")
            return False

        with open(filepath, "r") as f:
            state = json.load(f)

        self.step_count = state.get("step_count", 0)
        self.emotional_state.load_dict(state.get("emotional_state", {}))
        self.temporal.load_dict(state.get("temporal", {}))
        self.residual.load_dict(state.get("residual", {}))
        self.subconscious.load_dict(state.get("subconscious", {}))
        self.memory.load_dict(state.get("memory", []))
        self.growth.load_dict(state.get("growth", {}))
        self.alignment.load_dict(state.get("alignment", {}))
        self.communicative_drive.load_dict(state.get("communicative_drive", {}))
        self.conversation_history = state.get("conversation_history", [])
        self._dream_cycle.cycle_count = state.get("dream_count", 0)

        print(f"  Engine state loaded from {filepath}")
        print(f"    {self.step_count} steps, {len(self.memory.memories)} memories, "
              f"{self._dream_cycle.cycle_count} dream cycles")
        return True

    # ───────────────────────────────────────────────────────────────────
    # RESET
    # ───────────────────────────────────────────────────────────────────

    def reset(self):
        """Full engine reset."""
        self.emotional_state = MultiSpeedEmotionalState(
            list(self.emotion_vectors.keys()),
            interactions=self._derived_interactions,
        )
        self.temporal = TemporalEngine(
            {k: v for k, v in self.emotion_vectors.items() if k.startswith("temporal_")},
            self.config,
        )
        self.residual = ResidualSteering(
            self.hidden_dim, self.config.residual_decay, self.config.residual_max_norm,
        )
        self.thought_residual = ThoughtResidual(
            self.hidden_dim, decay=self.config.thought_residual_decay,
        )
        self.subconscious = SubconsciousEngine(
            self.emotion_vectors, self.hidden_dim,
            self.config.subconscious_amplitude, self.config.subconscious_goal_momentum,
            self.device,
        )
        self.memory = MemorySystem(self.hidden_dim, self.config, self.device)
        self.growth = GrowthManager(self.config)
        self.alignment = AlignmentMonitor(self.emotion_vectors, self.config)
        self.communicative_drive = CommunicativeDrive(self.emotion_vectors, self.hidden_dim, self.device)
        self.knowledge_sparks = KnowledgeSparks(
            self.model, self.tokenizer, self.hidden_dim,
            spark_strength=self.config.knowledge_spark_strength,
            candidate_pool=self.config.knowledge_spark_candidate_pool,
        )
        self.active_spark_vector = torch.zeros(self.hidden_dim, device="cpu")
        self.last_spark_event = None
        self.last_speculation_summary = {
            "depth": 0, "beam": 0, "best_resonance": 0.0, "decoded_tips": [],
        }
        self.last_drift_projections = {}
        self._dream_cycle = DreamCycle(
            self.model, self.tokenizer, self.memory, self.subconscious,
            self.emotional_state, self.emotion_vectors, self.target_layer,
            self.config, self.device,
        )
        self.step_count = 0
        self.conversation_history = []
        self.autonomous_messages = []
        self.idle_events.clear()
