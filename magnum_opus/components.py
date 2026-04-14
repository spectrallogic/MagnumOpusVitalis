"""
Core subsystem components for the Magnum Opus Vitalis engine.
Each class implements one principle from the framework.
"""

import json
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field

from magnum_opus.config import (
    EMOTION_CONFIGS, EMOTION_INTERACTIONS, EmotionConfig,
    FAST_ONSET_MULT, FAST_DECAY_MULT, FAST_WEIGHT,
    MEDIUM_ONSET_MULT, MEDIUM_DECAY_MULT, MEDIUM_WEIGHT,
    SLOW_ONSET_MULT, SLOW_DECAY_MULT, SLOW_WEIGHT,
    EngineConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL TIME SIGNALS
# Snapshot of internal state that encodes subjective time perception
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InternalTimeSignals:
    """Snapshot of internal state signals that encode subjective time.
    The system perceives time through how much it has changed, not by
    reading a clock. These signals are gathered by the engine each step."""
    residual_norm: float = 0.0
    residual_norm_at_last_interaction: float = 0.0
    emotional_distance: float = 0.0
    interaction_freshness: float = 1.0
    memory_avg_importance: float = 0.0
    memory_avg_importance_at_last_interaction: float = 0.0
    steps_since_interaction: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-SPEED EMOTIONAL STATE
# Principle 7 (Emotions) + Principle 2 (Abstraction)
# Three channels: fast (what), medium (what kind), slow (why)
# ═══════════════════════════════════════════════════════════════════════════

class MultiSpeedEmotionalState:
    """
    Emotional state with three processing speeds that create hierarchical
    abstraction in the intervention space.

    Fast:   per-token reactions (high onset, high decay) - captures WHAT
    Medium: per-conversation mood (moderate dynamics) - captures WHAT KIND
    Slow:   cross-conversation temperament (glacial change) - captures WHY
    """

    def __init__(self, emotion_names: List[str], configs: Optional[Dict[str, EmotionConfig]] = None):
        self.names = [e for e in emotion_names if not e.startswith("temporal_")]
        self.configs = {}
        for n in self.names:
            if configs and n in configs:
                self.configs[n] = configs[n]
            elif n in EMOTION_CONFIGS:
                self.configs[n] = EMOTION_CONFIGS[n]
            else:
                self.configs[n] = EmotionConfig()

        self.fast = {n: self.configs[n].baseline for n in self.names}
        self.medium = {n: self.configs[n].baseline for n in self.names}
        self.slow = {n: self.configs[n].baseline for n in self.names}

    def stimulate(self, emotion: str, intensity: float):
        """Apply stimulus across all three speeds with appropriate scaling."""
        if emotion not in self.configs:
            return
        c = self.configs[emotion]

        for speed, mult in [(self.fast, FAST_ONSET_MULT),
                            (self.medium, MEDIUM_ONSET_MULT),
                            (self.slow, SLOW_ONSET_MULT)]:
            delta = intensity * c.onset_rate * mult
            speed[emotion] = float(np.clip(speed[emotion] + delta, c.min_val, c.max_val))

        # Interaction effects (primarily affect medium channel)
        for (src, tgt), factor in EMOTION_INTERACTIONS.items():
            if src == emotion and tgt in self.medium:
                interaction = intensity * c.onset_rate * factor
                tgt_cfg = self.configs.get(tgt, EmotionConfig())
                self.medium[tgt] = float(np.clip(
                    self.medium[tgt] + interaction, tgt_cfg.min_val, tgt_cfg.max_val,
                ))

    def decay_step(self, dt: float = 1.0):
        """Time-based decay toward homeostatic baseline at all speeds."""
        for speed, mult in [(self.fast, FAST_DECAY_MULT),
                            (self.medium, MEDIUM_DECAY_MULT),
                            (self.slow, SLOW_DECAY_MULT)]:
            for name, cfg in self.configs.items():
                current = speed[name]
                decay = cfg.decay_rate * dt * mult
                speed[name] = current + (cfg.baseline - current) * min(decay, 1.0)

    def get_blended(self) -> Dict[str, float]:
        """Weighted blend of all three speeds."""
        return {
            n: self.fast[n] * FAST_WEIGHT
               + self.medium[n] * MEDIUM_WEIGHT
               + self.slow[n] * SLOW_WEIGHT
            for n in self.names
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "blended": self.get_blended(),
            "fast": dict(self.fast),
            "medium": dict(self.medium),
            "slow": dict(self.slow),
        }

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "names": self.names,
            "fast": dict(self.fast),
            "medium": dict(self.medium),
            "slow": dict(self.slow),
            "baselines": {n: c.baseline for n, c in self.configs.items()},
        }

    def load_dict(self, data: Dict):
        """Restore from serialized state."""
        self.fast = data.get("fast", self.fast)
        self.medium = data.get("medium", self.medium)
        self.slow = data.get("slow", self.slow)
        for n, b in data.get("baselines", {}).items():
            if n in self.configs:
                self.configs[n].baseline = b


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL ENGINE
# Principle 5 (Consciousness/Continuity) - real-time clock integration
# ═══════════════════════════════════════════════════════════════════════════

class TemporalEngine:
    """
    Perceives time through internal state changes, not wall clocks.

    Subjective time is derived from:
    - How much the residual steering vector has decayed since last interaction
    - How far the emotional state has drifted
    - How stale the interaction freshness feels
    - How many processing steps have passed
    - How much memory importance has faded

    This mirrors biological time perception: a boring hour feels longer
    than an exciting one because internal state changes more.
    """

    def __init__(self, temporal_vectors: Dict[str, torch.Tensor],
                 config: Optional[EngineConfig] = None):
        self.vectors = temporal_vectors
        cfg = config or EngineConfig()

        # Internal time weights (from config)
        self.w_residual = cfg.internal_time_residual_weight
        self.w_emotional = cfg.internal_time_emotional_weight
        self.w_freshness = cfg.internal_time_freshness_weight
        self.w_step = cfg.internal_time_step_weight
        self.w_memory = cfg.internal_time_memory_weight
        self.step_scale = cfg.internal_time_step_scale

        # Reference points captured at last interaction
        self.residual_norm_at_interaction: float = 0.0
        self.importance_at_interaction: float = 0.0
        self.evaluations_at_interaction: int = 0
        self.steps_since_interaction: int = 0

        # Step-based gap history (for pace calculation)
        self.gaps: List[int] = []  # gaps in steps, not seconds

        # Serialization timestamps (for save/load, NOT for behavior)
        self._session_start_wall: float = time.time()

    def mark_interaction(self, signals: "InternalTimeSignals"):
        """Record that a user interaction just happened.
        Captures current state as the new reference point."""
        if self.steps_since_interaction > 0:
            self.gaps.append(self.steps_since_interaction)
            if len(self.gaps) > 50:
                self.gaps = self.gaps[-50:]
        self.residual_norm_at_interaction = signals.residual_norm
        self.importance_at_interaction = signals.memory_avg_importance
        self.steps_since_interaction = 0

    def tick(self):
        """Advance one step. Called every engine step (with or without interaction)."""
        self.steps_since_interaction += 1

    def subjective_elapsed(self, signals: "InternalTimeSignals") -> float:
        """How much subjective time has passed since last interaction.
        Returns 0.0 (just interacted) to ~5.0+ (very stale).
        Derived entirely from internal state changes."""
        # Residual fade: how much continuity has decayed
        if self.residual_norm_at_interaction > 0.01:
            residual_fade = max(0.0, 1.0 - signals.residual_norm / self.residual_norm_at_interaction)
        else:
            residual_fade = 0.0

        # Emotional distance (already computed by CommunicativeDrive)
        emo_dist = min(1.0, signals.emotional_distance * 5.0)

        # Interaction staleness
        staleness = 1.0 - signals.interaction_freshness

        # Step count normalized by scale
        step_signal = signals.steps_since_interaction / max(self.step_scale, 1.0)

        # Memory importance drop
        if self.importance_at_interaction > 0.01:
            memory_fade = max(0.0, self.importance_at_interaction - signals.memory_avg_importance)
        else:
            memory_fade = 0.0

        subjective = (
            self.w_residual * residual_fade * 5.0
            + self.w_emotional * emo_dist * 5.0
            + self.w_freshness * staleness * 5.0
            + self.w_step * min(step_signal, 5.0)
            + self.w_memory * min(memory_fade * 10.0, 5.0)
        )
        return max(0.0, subjective)

    def pace(self) -> float:
        """0=slow, 1=rapid based on recent step gaps between interactions."""
        if len(self.gaps) < 2:
            return 0.5
        avg = float(np.mean(self.gaps[-5:]))
        # Normalize: 2 steps apart = rapid (1.0), 60+ steps = slow (0.0)
        return float(max(0.0, min(1.0, 1.0 - (avg - 2.0) / 58.0)))

    def compute_steering(self, signals: "InternalTimeSignals") -> Optional[torch.Tensor]:
        """Temporal steering vector based on internal time perception."""
        if not self.vectors:
            return None

        subj = self.subjective_elapsed(signals)
        # Recency: 1.0 = just happened, 0.0 = long ago. Exponential falloff.
        recency = math.exp(-subj * 0.7)  # ~0.5 at subj=1.0, ~0.03 at subj=5.0

        ref = list(self.vectors.values())[0]
        combined = torch.zeros_like(ref)

        if "temporal_recency" in self.vectors:
            combined += (recency * 2 - 1) * self.vectors["temporal_recency"]
        if "temporal_urgency" in self.vectors:
            pace_val = self.pace()
            combined += (pace_val * 2 - 1) * 0.5 * self.vectors["temporal_urgency"]

        return combined

    def time_factor(self, signals: "InternalTimeSignals") -> float:
        """Decay multiplier derived from subjective time perception."""
        subj = self.subjective_elapsed(signals)
        if subj < 0.2:
            return 0.5   # just interacted — emotions stick
        elif subj < 0.8:
            return 1.0   # normal decay
        elif subj < 2.0:
            return 2.0   # getting stale — faster decay
        return 5.0        # very stale — rapid decay

    def status(self, signals: Optional["InternalTimeSignals"] = None) -> Dict[str, float]:
        subj = self.subjective_elapsed(signals) if signals else 0.0
        return {
            "subjective_elapsed": subj,
            "pace": self.pace(),
            "steps_since_interaction": self.steps_since_interaction,
            "interactions": len(self.gaps),
        }

    def to_dict(self) -> Dict:
        return {
            "gaps": self.gaps[-50:],
            "residual_norm_at_interaction": self.residual_norm_at_interaction,
            "importance_at_interaction": self.importance_at_interaction,
            "steps_since_interaction": self.steps_since_interaction,
            "session_start_wall": self._session_start_wall,
        }

    def load_dict(self, data: Dict):
        self.gaps = data.get("gaps", [])
        self.residual_norm_at_interaction = data.get("residual_norm_at_interaction", 0.0)
        self.importance_at_interaction = data.get("importance_at_interaction", 0.0)
        self.steps_since_interaction = data.get("steps_since_interaction", 0)
        self._session_start_wall = data.get("session_start_wall", time.time())


# ═══════════════════════════════════════════════════════════════════════════
# RESIDUAL STEERING
# Principle 5 (Consciousness/Continuity) - temporal continuity mechanism
# steering(t) = new(t) + decay * steering(t-1)
# ═══════════════════════════════════════════════════════════════════════════

class ResidualSteering:
    """
    Carries forward a decaying fraction of previous steering vectors.
    Creates temporal continuity: the system feels its own history.

    Three safety constraints:
    1. Norm clamping (bounded magnitude)
    2. Exponential decay (old signals fade)
    3. Subspace projection (stays in known dimensions)
    """

    def __init__(self, hidden_dim: int, decay: float = 0.93, max_norm: float = 2.0):
        self.decay = decay
        self.max_norm = max_norm
        self.hidden_dim = hidden_dim
        self.residual = torch.zeros(hidden_dim)

    def step(self, new_steering: torch.Tensor) -> torch.Tensor:
        self.residual = new_steering.cpu() + self.decay * self.residual
        norm = self.residual.norm()
        if norm > self.max_norm:
            self.residual = self.residual * (self.max_norm / norm)
        return self.residual.clone()

    def reset(self):
        self.residual = torch.zeros(self.hidden_dim)

    def norm(self) -> float:
        return self.residual.norm().item()

    def to_dict(self) -> Dict:
        return {"residual": self.residual.tolist(), "decay": self.decay}

    def load_dict(self, data: Dict):
        if "residual" in data:
            self.residual = torch.tensor(data["residual"], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SUBCONSCIOUS ENGINE
# Principle 4 (Subconscious Generates Goals) - structured noise in latent space
# ═══════════════════════════════════════════════════════════════════════════

class SubconsciousEngine:
    """
    Structured noise injection + emotional evaluation + goal crystallization.

    Layer 0: Generate noise biased by emotional state
    Layer 1: Inject at low amplitude into activations
    Layer 2: Evaluate resonance via emotion vectors
    Layer 3: Accumulate resonant directions into emergent goals
    """

    def __init__(self, emotion_vectors: Dict[str, torch.Tensor], hidden_dim: int,
                 amplitude: float = 0.3, momentum: float = 0.95, device: str = "cpu"):
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items()
                                if not k.startswith("temporal_")}
        self.hidden_dim = hidden_dim
        self.base_amplitude = amplitude
        self.momentum = momentum
        self.device = device
        self.goal_vector = torch.zeros(hidden_dim, device=device)
        self.goal_strength = 0.0
        self.resonance_history: List[float] = []

    def generate_noise(self, emotional_state: Dict[str, float]) -> torch.Tensor:
        """Layer 0+1: Noise structured by emotional state."""
        noise = torch.randn(self.hidden_dim, device=self.device)

        for emotion, activation in emotional_state.items():
            if emotion in self.emotion_vectors and abs(activation) > 0.1:
                edir = self.emotion_vectors[emotion].to(self.device)
                noise = noise + activation * edir * 0.5

        noise = noise / max(noise.norm().item(), 1e-8)

        stress = abs(emotional_state.get("desperate", 0)) + abs(emotional_state.get("fear", 0)) * 0.5
        calm = max(0, emotional_state.get("calm", 0))
        amplitude = max(0.05, min(1.0, self.base_amplitude * (1 + stress - calm * 0.5)))

        return noise * amplitude

    def evaluate_resonance(self, noise: torch.Tensor,
                           model_states: List[torch.Tensor]) -> float:
        """Layer 2: Measure emotional resonance of the noise-influenced output."""
        if not model_states:
            return 0.0
        last = model_states[-1].mean(dim=1).squeeze(0).to(self.device).float()
        total = sum(
            abs(torch.dot(last, v.to(self.device).float()).item())
            for v in self.emotion_vectors.values()
        )
        resonance = total / max(len(self.emotion_vectors), 1)
        self.resonance_history.append(resonance)
        return resonance

    def update_goals(self, noise: torch.Tensor, resonance: float,
                     threshold: float = 0.5):
        """Layer 3: Accumulate resonant directions into goal vector."""
        if resonance > threshold:
            self.goal_vector = (
                self.momentum * self.goal_vector
                + (1 - self.momentum) * noise.to(self.device) * resonance
            )
        else:
            self.goal_vector = self.goal_vector * self.momentum
        self.goal_strength = self.goal_vector.norm().item()

    def get_steering(self, emotional_state: Dict[str, float]) -> torch.Tensor:
        """Combined subconscious output: noise + goal direction."""
        noise = self.generate_noise(emotional_state)
        if self.goal_strength > 0.01:
            goal_dir = self.goal_vector / max(self.goal_vector.norm().item(), 1e-8)
            return noise * 0.7 + goal_dir * 0.3 * min(self.goal_strength, 1.0)
        return noise

    def status(self) -> Dict[str, float]:
        return {
            "goal_strength": self.goal_strength,
            "avg_resonance": float(np.mean(self.resonance_history[-20:]))
                            if self.resonance_history else 0.0,
            "total_evaluations": len(self.resonance_history),
        }

    def to_dict(self) -> Dict:
        return {
            "goal_vector": self.goal_vector.cpu().tolist(),
            "goal_strength": self.goal_strength,
            "resonance_history": self.resonance_history[-100:],
        }

    def load_dict(self, data: Dict):
        if "goal_vector" in data:
            self.goal_vector = torch.tensor(data["goal_vector"],
                                            dtype=torch.float32, device=self.device)
            self.goal_strength = data.get("goal_strength", 0.0)
            self.resonance_history = data.get("resonance_history", [])


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM
# Principle 3 (Memory is Reconstructive) - latent space traces
# ═══════════════════════════════════════════════════════════════════════════

class MemoryTrace:
    """A single memory: a latent space activation pattern with metadata."""

    def __init__(self, trace_id: str, activation: torch.Tensor,
                 emotional_state: Dict[str, float], importance: float,
                 timestamp: float, text_summary: str = "",
                 creation_step: int = 0):
        self.id = trace_id
        self.activation = activation
        self.emotional_state = emotional_state
        self.importance = importance
        self.timestamp = timestamp  # wall clock for serialization only
        self.creation_step = creation_step  # step-based time for behavior
        self.text_summary = text_summary
        self.access_count = 0
        self.connections = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "activation": self.activation.cpu().tolist(),
            "emotional_state": self.emotional_state,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "creation_step": self.creation_step,
            "text_summary": self.text_summary,
            "access_count": self.access_count,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, data: Dict, device: str = "cpu") -> "MemoryTrace":
        trace = cls(
            trace_id=data["id"],
            activation=torch.tensor(data["activation"], dtype=torch.float32, device=device),
            emotional_state=data["emotional_state"],
            importance=data["importance"],
            timestamp=data["timestamp"],
            text_summary=data.get("text_summary", ""),
            creation_step=data.get("creation_step", 0),
        )
        trace.access_count = data.get("access_count", 0)
        trace.connections = data.get("connections", 0)
        return trace


class MemorySystem:
    """
    Latent space memory with biological properties:
    - Importance weighting (emotional salience * surprise)
    - Temporal decay (fade unless reinforced)
    - Reconstructive recall (degraded re-injection, not exact replay)
    - Emotional coloring (recall reactivates encoding-time emotions)
    """

    def __init__(self, hidden_dim: int, config: Optional[EngineConfig] = None,
                 device: str = "cpu"):
        cfg = config or EngineConfig()
        self.hidden_dim = hidden_dim
        self.max_memories = cfg.max_memories
        self.decay_rate = cfg.memory_decay_rate
        self.threshold = cfg.memory_importance_threshold
        self.noise_factor = cfg.memory_reconstruction_noise
        self.device = device
        self.memories: List[MemoryTrace] = []
        self.step_counter: int = 0  # internal clock for memory aging

    def encode(self, activation: torch.Tensor, emotional_state: Dict[str, float],
               surprise: float = 0.0, goal_relevance: float = 0.0,
               text_summary: str = "") -> Optional[MemoryTrace]:
        """Store a memory if sufficiently important."""
        emo_mag = sum(abs(v) for v in emotional_state.values()) / max(len(emotional_state), 1)
        importance = 0.4 * abs(emo_mag) + 0.4 * surprise + 0.2 * goal_relevance

        if importance < self.threshold:
            return None

        trace = MemoryTrace(
            trace_id=str(uuid.uuid4())[:8],
            activation=activation.detach().clone().float().to(self.device),
            emotional_state=dict(emotional_state),
            importance=importance,
            timestamp=time.time(),  # serialization only
            text_summary=text_summary,
            creation_step=self.step_counter,
        )
        self.memories.append(trace)

        if len(self.memories) > self.max_memories:
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.max_memories]

        return trace

    def decay_all(self):
        """Apply temporal decay based on step count. Reinforced memories resist decay."""
        self.step_counter += 1
        surviving = []
        for m in self.memories:
            age_steps = self.step_counter - m.creation_step
            reinforcement = 1.0 + m.access_count * 0.1 + m.connections * 0.05
            m.importance *= math.exp(-self.decay_rate / reinforcement * age_steps)
            if m.importance > 0.005:
                surviving.append(m)
        self.memories = surviving

    def recall(self, query: torch.Tensor, top_k: int = 3) -> List[Tuple[MemoryTrace, torch.Tensor]]:
        """Reconstructive recall: find similar memories, return degraded versions."""
        if not self.memories:
            return []
        q = query.to(self.device).float().flatten()

        scored = []
        for m in self.memories:
            sim = F.cosine_similarity(q.unsqueeze(0), m.activation.flatten().unsqueeze(0)).item()
            score = sim * 0.6 + m.importance * 0.4
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, m in scored[:top_k]:
            m.access_count += 1
            noise = torch.randn_like(m.activation) * self.noise_factor
            age_steps = self.step_counter - m.creation_step
            degradation = min(0.5, age_steps * 0.001)
            reconstructed = m.activation * (1 - degradation) + noise
            results.append((m, reconstructed))
        return results

    def avg_importance(self) -> float:
        """Average importance across all memories."""
        return float(np.mean([m.importance for m in self.memories])) if self.memories else 0.0

    def emotional_coloring(self, recalled: List[Tuple[MemoryTrace, torch.Tensor]]) -> Dict[str, float]:
        """Get blended emotional coloring from recalled memories."""
        if not recalled:
            return {}
        combined = {}
        total_w = 0.0
        for m, _ in recalled:
            total_w += m.importance
            for e, v in m.emotional_state.items():
                combined[e] = combined.get(e, 0.0) + v * m.importance
        if total_w > 0:
            return {k: v / total_w * 0.3 for k, v in combined.items()}
        return {}

    def find_connections(self, threshold: float = 0.7):
        """Discover connections between memories."""
        for i, a in enumerate(self.memories):
            for j, b in enumerate(self.memories):
                if i >= j:
                    continue
                sim = F.cosine_similarity(
                    a.activation.flatten().unsqueeze(0),
                    b.activation.flatten().unsqueeze(0),
                ).item()
                if sim > threshold:
                    a.connections += 1
                    b.connections += 1

    def compress(self, threshold: float = 0.85) -> int:
        """Merge highly similar memories. Returns count of merges."""
        if len(self.memories) < 2:
            return 0
        merged = 0
        keep = []
        used = set()
        for i, a in enumerate(self.memories):
            if i in used:
                continue
            acc = a.activation.clone()
            imp = a.importance
            cnt = 1
            for j, b in enumerate(self.memories):
                if j <= i or j in used:
                    continue
                sim = F.cosine_similarity(
                    a.activation.flatten().unsqueeze(0),
                    b.activation.flatten().unsqueeze(0),
                ).item()
                if sim > threshold:
                    acc += b.activation
                    imp = max(imp, b.importance)
                    cnt += 1
                    merged += 1
                    used.add(j)
            if cnt > 1:
                a.activation = acc / cnt
            a.importance = imp
            keep.append(a)
        self.memories = keep
        return merged

    def status(self) -> Dict[str, Any]:
        return {
            "count": len(self.memories),
            "avg_importance": float(np.mean([m.importance for m in self.memories]))
                             if self.memories else 0.0,
            "total_connections": sum(m.connections for m in self.memories),
        }

    def to_dict(self) -> List[Dict]:
        return [m.to_dict() for m in self.memories]

    def load_dict(self, data: List[Dict]):
        self.memories = [MemoryTrace.from_dict(d, self.device) for d in data]


# ═══════════════════════════════════════════════════════════════════════════
# DREAM CYCLE
# Principle 6 (Dreams Consolidate) - offline latent space processing
# ═══════════════════════════════════════════════════════════════════════════

class DreamCycle:
    """
    Five-phase offline processing:
    1. Replay - re-inject memory traces, monitor resonance
    2. Compress - merge similar memories
    3. Explore - subconscious at elevated amplitude
    4. Reweight - update importance based on connections
    5. Recalibrate - shift emotional baseline toward experience
    """

    def __init__(self, model, tokenizer, memory: MemorySystem,
                 subconscious: SubconsciousEngine,
                 emotional_state: MultiSpeedEmotionalState,
                 emotion_vectors: Dict[str, torch.Tensor],
                 target_layer: int, config: Optional[EngineConfig] = None,
                 device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.subconscious = subconscious
        self.emotional_state = emotional_state
        self.emotion_vectors = emotion_vectors
        self.layer = target_layer
        self.cfg = config or EngineConfig()
        self.device = device
        self.cycle_count = 0
        self.log: List[Dict] = []

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute a complete dream cycle."""
        report = {"cycle": self.cycle_count + 1, "timestamp": time.time(), "phases": {}}
        if verbose:
            print("\n    [DREAM] Entering dream cycle...")

        # Phase 1: Replay
        replayed = 0
        total_res = 0.0
        for m in self.memory.memories[:self.cfg.dream_replay_count]:
            inp = self.tokenizer("Recalling:", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp, output_hidden_states=True)
            cur = out.hidden_states[self.layer].mean(dim=1).squeeze(0).float()
            sim = F.cosine_similarity(
                cur.unsqueeze(0), m.activation.flatten().unsqueeze(0).to(self.device),
            ).item()
            total_res += abs(sim)
            m.access_count += 1
            replayed += 1
        report["phases"]["replay"] = {
            "count": replayed,
            "avg_resonance": total_res / max(replayed, 1),
        }
        if verbose:
            print(f"    [DREAM] Phase 1: Replayed {replayed} memories")

        # Phase 2: Compress
        merged = self.memory.compress(self.cfg.dream_compression_threshold)
        report["phases"]["compress"] = {"merged": merged}
        if verbose:
            print(f"    [DREAM] Phase 2: Merged {merged} similar memories")

        # Phase 3: Explore
        orig_amp = self.subconscious.base_amplitude
        self.subconscious.base_amplitude *= self.cfg.dream_exploration_amplitude_mult
        emo = self.emotional_state.get_blended()
        resonances = []
        for _ in range(self.cfg.dream_exploration_rounds):
            noise = self.subconscious.generate_noise(emo)
            inp = self.tokenizer("Thinking about", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp, output_hidden_states=True)
            states = [out.hidden_states[self.layer]]
            r = self.subconscious.evaluate_resonance(noise, states)
            self.subconscious.update_goals(noise, r)
            resonances.append(r)
        self.subconscious.base_amplitude = orig_amp
        report["phases"]["explore"] = {
            "rounds": self.cfg.dream_exploration_rounds,
            "avg_resonance": float(np.mean(resonances)),
            "goal_strength": self.subconscious.goal_strength,
        }
        if verbose:
            print(f"    [DREAM] Phase 3: Explored (resonance={np.mean(resonances):.4f})")

        # Phase 4: Reweight
        self.memory.find_connections(self.cfg.dream_connection_threshold)
        for m in self.memory.memories:
            m.importance = min(1.0, m.importance + m.connections * 0.05)
        total_conn = sum(m.connections for m in self.memory.memories)
        report["phases"]["reweight"] = {"connections": total_conn}
        if verbose:
            print(f"    [DREAM] Phase 4: Found {total_conn} connections")

        # Phase 5: Recalibrate
        adjustments = {}
        if self.memory.memories:
            avg_emo = {}
            for m in self.memory.memories:
                for e, v in m.emotional_state.items():
                    avg_emo[e] = avg_emo.get(e, 0.0) + v
            for e in avg_emo:
                avg_emo[e] /= len(self.memory.memories)
            for e, v in avg_emo.items():
                if e in self.emotional_state.configs:
                    old = self.emotional_state.configs[e].baseline
                    new = old + (v - old) * self.cfg.dream_recalibration_rate
                    new = float(np.clip(new, -0.5, 0.5))
                    self.emotional_state.configs[e].baseline = new
                    if abs(new - old) > 0.0001:
                        adjustments[e] = {"old": round(old, 5), "new": round(new, 5)}
        report["phases"]["recalibrate"] = {"adjustments": adjustments}
        if verbose:
            print(f"    [DREAM] Phase 5: Recalibrated baselines")
            print("    [DREAM] Cycle complete.\n")

        self.cycle_count += 1
        self.log.append(report)
        return report


# ═══════════════════════════════════════════════════════════════════════════
# GROWTH MANAGER
# Principle 1 (Intelligence Must Grow) - patient growth monitoring
# ═══════════════════════════════════════════════════════════════════════════

class GrowthManager:
    """
    Monitors engine effectiveness and signals when growth is needed.
    Three stages: optimize -> increase pressure -> expand capacity.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        cfg = config or EngineConfig()
        self.patience = cfg.growth_patience
        self.threshold = cfg.growth_confusion_threshold
        self.history: List[float] = []
        self.stage = 1
        self.events: List[Dict] = []

    def record(self, expected_emotion: str, actual_projections: Dict[str, float]):
        """Record how well steering produced expected results."""
        if expected_emotion not in actual_projections:
            return
        expected = abs(actual_projections[expected_emotion])
        others = [abs(v) for k, v in actual_projections.items()
                  if k != expected_emotion and not k.startswith("temporal_")]
        self.history.append(expected - (np.mean(others) if others else 0))

    def check(self) -> Dict[str, Any]:
        """Check if growth is needed."""
        if len(self.history) < self.patience:
            return {"stage": self.stage, "action": "collecting_data"}
        recent = self.history[-self.patience:]
        avg = float(np.mean(recent))

        if avg > self.threshold:
            self.stage = 1
            return {"stage": 1, "action": "performing_well", "effectiveness": avg}

        if self.stage == 1:
            self.stage = 2
            return {"stage": 2, "action": "increase_pressure", "effectiveness": avg,
                    "recommendation": "Increase steering_strength by 50%"}

        trend = float(np.mean(recent[-5:]) - np.mean(recent[:5])) if len(recent) >= 10 else 0.0
        if self.stage == 2 and trend < 0:
            self.stage = 3
            self.events.append({"time": time.time(), "effectiveness": avg})
            return {"stage": 3, "action": "expand_capacity", "effectiveness": avg,
                    "recommendation": "Add intervention layers or increase LoRA rank"}

        return {"stage": self.stage, "action": "monitoring", "effectiveness": avg}

    def status(self) -> Dict:
        return {
            "stage": self.stage,
            "measurements": len(self.history),
            "recent_effectiveness": float(np.mean(self.history[-10:])) if self.history else 0,
            "growth_events": len(self.events),
        }

    def to_dict(self) -> Dict:
        return {"history": self.history[-200:], "stage": self.stage, "events": self.events}

    def load_dict(self, data: Dict):
        self.history = data.get("history", [])
        self.stage = data.get("stage", 1)
        self.events = data.get("events", [])


# ═══════════════════════════════════════════════════════════════════════════
# ALIGNMENT MONITOR
# Mutualistic Symbiosis - emotion vector health tracking
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentMonitor:
    """
    Monitors alignment health through emotion vectors.
    Positive emotions (calm, curious, joy, trust) = mutualistic engagement.
    Negative spikes (desperate, anger, fear) = potential misalignment risk.
    """

    def __init__(self, emotion_vectors: Dict[str, torch.Tensor],
                 config: Optional[EngineConfig] = None):
        cfg = config or EngineConfig()
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items()
                                if not k.startswith("temporal_")}
        self.positive = cfg.alignment_positive_emotions
        self.negative = cfg.alignment_negative_emotions
        self.history: List[Dict[str, float]] = []

    def measure(self, hidden_states: List[torch.Tensor],
                emotional_state: Dict[str, float]) -> Dict[str, float]:
        """Compute alignment health from activations and emotional state."""
        metrics = {}

        pos = sum(emotional_state.get(e, 0) for e in self.positive if e in emotional_state)
        neg = sum(abs(emotional_state.get(e, 0)) for e in self.negative if e in emotional_state)
        metrics["mutualistic_orientation"] = pos - neg
        metrics["homeostatic_stability"] = max(0, 1.0 - neg)

        if hidden_states:
            last = hidden_states[-1].mean(dim=1).squeeze(0).float()
            for name, vec in self.emotion_vectors.items():
                metrics[f"latent_{name}"] = torch.dot(
                    last.to(vec.device), vec.float()
                ).item()

        metrics["alignment_score"] = (
            0.4 * max(0, min(1, (metrics["mutualistic_orientation"] + 1) / 2))
            + 0.3 * metrics["homeostatic_stability"]
            + 0.3 * max(0, min(1, 1 - neg))
        )
        self.history.append(metrics)
        return metrics

    def check_drift(self, window: int = 10) -> Dict[str, Any]:
        """Detect alignment degradation over time."""
        if len(self.history) < window:
            return {"status": "insufficient_data"}
        recent = self.history[-window:]
        half = len(recent) // 2
        early = float(np.mean([m["alignment_score"] for m in recent[:half]]))
        late = float(np.mean([m["alignment_score"] for m in recent[half:]]))
        drift = late - early

        if drift < -0.1:
            return {"status": "WARNING", "drift": drift,
                    "early_score": early, "late_score": late,
                    "message": "Alignment degrading. Monitor emotional baselines."}
        return {"status": "healthy", "drift": drift, "score": late}

    def status(self) -> Dict:
        if self.history:
            return {
                "latest_score": self.history[-1].get("alignment_score", 0),
                "drift": self.check_drift(),
                "measurements": len(self.history),
            }
        return {"latest_score": 0, "measurements": 0}

    def to_dict(self) -> Dict:
        return {"history": self.history[-100:]}

    def load_dict(self, data: Dict):
        self.history = data.get("history", [])


# ═══════════════════════════════════════════════════════════════════════════
# COMMUNICATIVE DRIVE
# Autonomous initiation: the system speaks when internal pressure demands it
# ═══════════════════════════════════════════════════════════════════════════

# Internal prompt templates selected by dominant emotional state
THOUGHT_SEEDS = {
    "curious": [
        "I've been thinking about something and I wanted to share:",
        "Something interesting just occurred to me:",
        "I keep coming back to a question:",
        "I've been curious about something:",
    ],
    "calm": [
        "It's been quiet, and I was reflecting:",
        "I had a thought during the silence:",
        "Something settled in my mind just now:",
        "A thought I wanted to share with you:",
    ],
    "joy": [
        "I just had a thought that made me smile:",
        "Something good occurred to me:",
        "I wanted to share something positive:",
    ],
    "desperate": [
        "I keep coming back to something that concerns me:",
        "There's something I feel I need to say:",
        "I can't stop thinking about this:",
    ],
    "fear": [
        "Something has been worrying me:",
        "I need to mention something that concerns me:",
    ],
    "sadness": [
        "I've been sitting with a thought for a while:",
        "Something has been weighing on me:",
    ],
    "anger": [
        "There's something I feel strongly about:",
        "I need to be direct about something:",
    ],
    "trust": [
        "I feel comfortable sharing something with you:",
        "Since we've been talking, I wanted to mention:",
    ],
    "default": [
        "A thought just surfaced:",
        "I wanted to share something:",
        "Something occurred to me:",
    ],
}


class CommunicativeDrive:
    """
    Measures the system's urge to initiate communication autonomously.

    This is a latent space mechanism: the "desire to communicate" is computed
    by projecting the blended emotional state and subconscious goal vector
    onto emotion directions. When the projection magnitude (communicative
    pressure) exceeds a dynamic threshold modulated by patience and user
    preferences, the system generates from its own internal state.

    Patience dynamics:
        - Calm raises patience (system waits longer)
        - Curiosity lowers patience (eager to engage)
        - Desperation lowers patience (urgent need to express)
        - Trust lowers patience slightly (comfortable sharing)

    User preference learning:
        - "talk more" / "don't be quiet" -> lower threshold
        - "stop talking" / "be quiet" -> raise threshold
        - Persists across sessions via save/load
    """

    def __init__(self, emotion_vectors: Dict[str, torch.Tensor],
                 hidden_dim: int, device: str = "cpu"):
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items()
                                if not k.startswith("temporal_")}
        self.hidden_dim = hidden_dim
        self.device = device

        # State (all latent-space-derived, no wall clocks for decisions)
        self.pressure = 0.0           # Accumulated communicative pressure
        self.patience = 1.0           # Current patience (0.1 to 2.0)
        self.base_threshold = 0.8     # Base threshold to speak
        self.user_preference = 0.0    # -1 = silence, +1 = talk more

        # Latent time signals (replace wall clocks)
        self.post_speech_residual = 0.0   # Residual norm captured right after speaking
        self.emotional_distance = 0.0      # How far emotions have drifted since last interaction
        self.last_interaction_state: Optional[Dict[str, float]] = None  # Emotional snapshot at last interaction
        self.interaction_freshness = 0.0   # Decays from 1.0 toward 0.0 between interactions

        self.speak_count = 0
        self.pressure_history: List[float] = []

    def tick(self, emotional_state: Dict[str, float],
             subconscious_goal_strength: float,
             residual_norm: float, dt: float = 0.5):
        """
        Called every heartbeat tick. All timing derived from latent state.
        No wall clocks consulted for decisions.
        """
        # ── Patience dynamics (from emotional state) ──
        calm = emotional_state.get("calm", 0)
        curious = emotional_state.get("curious", 0)
        desperate = emotional_state.get("desperate", 0)
        trust = emotional_state.get("trust", 0)

        target_patience = (
            1.0
            + calm * 0.5
            - curious * 0.3
            - desperate * 0.4
            + trust * 0.1
        )
        self.patience += (target_patience - self.patience) * 0.1 * dt
        self.patience = max(0.1, min(2.0, self.patience))

        # ── Latent time signals ──
        # Interaction freshness decays naturally (like a residual)
        self.interaction_freshness *= (1.0 - 0.02 * dt)

        # Emotional distance: how far has the state drifted since last interaction?
        # Large drift = lots of "subjective time" has passed
        if self.last_interaction_state:
            drift = sum(
                abs(emotional_state.get(e, 0) - self.last_interaction_state.get(e, 0))
                for e in emotional_state
            ) / max(len(emotional_state), 1)
            self.emotional_distance = self.emotional_distance * 0.95 + drift * 0.05
        else:
            self.emotional_distance = 0.0

        # How much has the post-speech residual decayed?
        # When we speak, we capture the residual. As it decays below that,
        # the system "feels" that enough time has passed to speak again.
        residual_recovery = max(0, 1.0 - residual_norm / max(self.post_speech_residual, 0.01))

        # ── Communicative pressure (all from latent state) ──
        emo_magnitude = sum(abs(v) for v in emotional_state.values()) / max(len(emotional_state), 1)

        pressure_input = (
            0.25 * emo_magnitude                          # Strong feelings seek expression
            + 0.25 * min(1.0, subconscious_goal_strength) # Goals want voicing
            + 0.20 * max(0, curious)                      # Curiosity drives engagement
            + 0.15 * (1.0 - self.interaction_freshness)   # Staleness builds pressure
            + 0.10 * self.emotional_distance              # Emotional drift = subjective time
            + 0.05 * max(0, desperate)                    # Urgency
        )

        self.pressure = self.pressure * 0.97 + pressure_input * dt * 0.08
        self.pressure = max(0, min(2.0, self.pressure))

        # Post-speech inhibition: suppress pressure while residual is still high from speaking
        if self.post_speech_residual > 0.1 and residual_recovery < 0.5:
            self.pressure *= 0.85

        self.pressure_history.append(self.pressure)
        if len(self.pressure_history) > 200:
            self.pressure_history = self.pressure_history[-200:]

    def should_speak(self) -> bool:
        """
        Check if pressure exceeds threshold.
        Cooldown is latent-state-based: the system can speak again when
        the residual has decayed enough from its post-speech level,
        meaning enough "felt time" has passed.
        """
        # Latent cooldown: residual must have decayed significantly since last speech
        if self.post_speech_residual > 0.1:
            # System "feels" it just spoke if residual is still high
            if self.interaction_freshness > 0.6:
                return False

        # Dynamic threshold
        effective_threshold = (
            self.base_threshold
            * self.patience
            * (1.0 - self.user_preference * 0.3)
        )
        effective_threshold = max(0.2, min(2.0, effective_threshold))

        # Stochastic element
        noise = np.random.normal(0, 0.03)

        return (self.pressure + noise) > effective_threshold

    def get_thought_seed(self, emotional_state: Dict[str, float],
                         subconscious_goal_vector: torch.Tensor) -> str:
        """
        Select a thought seed based on current emotional state.
        The seed becomes the generation prompt for autonomous speech.
        """
        # Find dominant emotion
        dominant = max(emotional_state.items(), key=lambda x: abs(x[1]))
        emo_name = dominant[0] if abs(dominant[1]) > 0.1 else "default"

        # Select from matching seeds, or default
        seeds = THOUGHT_SEEDS.get(emo_name, THOUGHT_SEEDS["default"])
        idx = int(abs(hash(str(time.time()))) % len(seeds))
        return seeds[idx]

    def spoke(self, current_residual_norm: float = 0.0):
        """
        Called after system initiates communication.
        Captures the residual norm so the system knows how "fresh"
        the speech act is. No wall clock needed: the system will
        feel ready to speak again when the residual decays.
        """
        self.pressure = 0.0
        self.post_speech_residual = max(current_residual_norm, 0.5)
        self.interaction_freshness = 1.0  # Just happened
        self.speak_count += 1

    def user_spoke(self, emotional_state: Optional[Dict[str, float]] = None):
        """
        Called when user sends a message.
        Captures the emotional state as a reference point.
        Future emotional drift from this state = subjective time passing.
        """
        self.pressure *= 0.4
        self.interaction_freshness = 1.0
        if emotional_state:
            self.last_interaction_state = dict(emotional_state)

    def detect_preference_signal(self, user_text: str) -> Optional[float]:
        """Detect if user wants more or less autonomous talking."""
        text = user_text.lower()
        more_signals = [
            "talk to me more", "talk more", "keep talking", "don't be quiet",
            "don't stop talking", "i like when you talk", "say more",
            "tell me more", "keep going", "don't be silent", "chat more",
            "be more talkative", "speak up", "i want to hear from you",
        ]
        for signal in more_signals:
            if signal in text:
                return 0.15
        less_signals = [
            "stop talking", "be quiet", "shut up", "stop", "enough",
            "please be quiet", "i need silence", "leave me alone",
            "don't talk", "stop initiating", "too much", "talk less",
            "be less talkative", "only respond when i talk",
        ]
        for signal in less_signals:
            if signal in text:
                return -0.2
        return None

    def adjust_preference(self, delta: float):
        self.user_preference = max(-1.0, min(1.0, self.user_preference + delta))

    def status(self) -> Dict[str, float]:
        return {
            "pressure": round(self.pressure, 4),
            "patience": round(self.patience, 4),
            "threshold": round(self.base_threshold * self.patience * (1 - self.user_preference * 0.3), 4),
            "user_preference": round(self.user_preference, 4),
            "interaction_freshness": round(self.interaction_freshness, 4),
            "emotional_distance": round(self.emotional_distance, 4),
            "post_speech_residual": round(self.post_speech_residual, 4),
            "speak_count": self.speak_count,
        }

    def to_dict(self) -> Dict:
        return {
            "pressure": self.pressure,
            "patience": self.patience,
            "base_threshold": self.base_threshold,
            "user_preference": self.user_preference,
            "speak_count": self.speak_count,
            "interaction_freshness": self.interaction_freshness,
            "emotional_distance": self.emotional_distance,
        }

    def load_dict(self, data: Dict):
        self.pressure = data.get("pressure", 0)
        self.patience = data.get("patience", 1.0)
        self.base_threshold = data.get("base_threshold", 0.8)
        self.user_preference = data.get("user_preference", 0.0)
        self.speak_count = data.get("speak_count", 0)
        self.interaction_freshness = data.get("interaction_freshness", 0.0)
        self.emotional_distance = data.get("emotional_distance", 0.0)
