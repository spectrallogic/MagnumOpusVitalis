"""
The unified Magnum Opus Vitalis engine.
Integrates all components, manages steering hooks, handles generation,
and provides save/load for persistent state across sessions.
"""

import json
import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from magnum_opus.config import EngineConfig
from magnum_opus.components import (
    MultiSpeedEmotionalState, TemporalEngine, ResidualSteering,
    SubconsciousEngine, MemorySystem, DreamCycle, GrowthManager,
    AlignmentMonitor, CommunicativeDrive,
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
        hidden_states = output[0]
        self.captured_states.append(hidden_states.detach().clone())

        if self.active and self.steering_vector is not None:
            sv = self.steering_vector.to(hidden_states.device).to(hidden_states.dtype)
            modified = hidden_states + sv.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
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

        # Initialize all subsystems
        temporal_vecs = {k: v for k, v in emotion_vectors.items() if k.startswith("temporal_")}

        self.emotional_state = MultiSpeedEmotionalState(list(emotion_vectors.keys()))
        self.temporal = TemporalEngine(temporal_vecs, self.config.temporal_recency_halflife)
        self.residual = ResidualSteering(
            self.hidden_dim, self.config.residual_decay, self.config.residual_max_norm,
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

        # State
        self.step_count = 0
        self.conversation_history: List[Dict[str, str]] = []
        self._dream_thread: Optional[threading.Thread] = None
        self.autonomous_messages: List[Dict[str, Any]] = []  # Queue for self-initiated messages

        # Idle tracking for auto-dream
        self._last_activity = time.time()

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
        temporal_vec = self.temporal.compute_steering()
        if temporal_vec is not None:
            temporal_vec = temporal_vec.cpu().float()
        else:
            temporal_vec = torch.zeros(self.hidden_dim)

        # 3. Subconscious component
        sub_vec = self.subconscious.get_steering(emo_weights).cpu().float()

        # 4. Combine
        combined = (
            emo_vec * self.config.steering_strength
            + temporal_vec * 1.0
            + sub_vec * 0.5
        )

        # 5. Residual (temporal continuity)
        steered = self.residual.step(combined)

        return steered.to(self.device)

    # ───────────────────────────────────────────────────────────────────
    # CORE: Advance engine state by one step
    # ───────────────────────────────────────────────────────────────────

    def _step(self, stimulus: Optional[Dict[str, float]] = None, dt: Optional[float] = None):
        """Internal state advancement."""
        if dt is None:
            dt = self.temporal.time_factor()
        self.temporal.mark_interaction()

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
            self.emotional_state.stimulate(emotion, intensity * 0.3)

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
        # Notify communicative drive that user spoke (with current emotional state for drift tracking)
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
        Generate a message from the system's own internal state.
        Called when communicative pressure exceeds the threshold.

        The thought seed comes from the dominant emotion.
        The content is generated with full engine steering active,
        meaning the output reflects the system's current emotional
        landscape, subconscious goals, and accumulated experience.
        """
        blended = self.emotional_state.get_blended()

        # Get thought seed based on internal state
        seed = self.communicative_drive.get_thought_seed(
            blended, self.subconscious.goal_vector
        )

        # Step the engine (autonomous activity)
        self._step(dt=0.5)

        # Generate from the thought seed with full steering
        prompt = f"Assistant: {seed}"
        text, states = self._generate(prompt, max_tokens=80)
        self._post_generation(seed, text, states)

        # Extract response
        if seed in text:
            response = text.split(seed)[-1].strip()
            response = seed + " " + response
        else:
            response = seed + " " + text[len(prompt):].strip()

        # Mark that the system spoke (captures residual norm for latent cooldown)
        self.communicative_drive.spoke(current_residual_norm=self.residual.norm())

        # Log
        self.conversation_history.append({
            "user": "[autonomous]",
            "assistant": response,
            "timestamp": time.time(),
            "emotional_state": blended,
            "autonomous": True,
        })

        # Queue for dashboard
        self.autonomous_messages.append({
            "message": response,
            "timestamp": time.time(),
            "emotional_state": blended,
            "pressure_at_trigger": self.communicative_drive.pressure,
        })

        return response

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
        """Full engine status for monitoring."""
        return {
            "step": self.step_count,
            "emotional_state": self.emotional_state.snapshot(),
            "temporal": self.temporal.status(),
            "residual_norm": self.residual.norm(),
            "subconscious": self.subconscious.status(),
            "memory": self.memory.status(),
            "growth": self.growth.status(),
            "alignment": self.alignment.status(),
            "communicative_drive": self.communicative_drive.status(),
            "conversation_turns": len(self.conversation_history),
            "dream_cycles": self._dream_cycle.cycle_count,
            "idle_seconds": self.idle_seconds(),
        }

    def alignment_health(self) -> Dict[str, Any]:
        """Quick alignment health check."""
        return self.alignment.check_drift()

    def get_autonomous_messages(self) -> List[Dict[str, Any]]:
        """Drain the autonomous message queue."""
        msgs = list(self.autonomous_messages)
        self.autonomous_messages = []
        return msgs

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
        self.emotional_state = MultiSpeedEmotionalState(list(self.emotion_vectors.keys()))
        self.temporal = TemporalEngine(
            {k: v for k, v in self.emotion_vectors.items() if k.startswith("temporal_")},
            self.config.temporal_recency_halflife,
        )
        self.residual = ResidualSteering(
            self.hidden_dim, self.config.residual_decay, self.config.residual_max_norm,
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
        self._dream_cycle = DreamCycle(
            self.model, self.tokenizer, self.memory, self.subconscious,
            self.emotional_state, self.emotion_vectors, self.target_layer,
            self.config, self.device,
        )
        self.step_count = 0
        self.conversation_history = []
        self.autonomous_messages = []
