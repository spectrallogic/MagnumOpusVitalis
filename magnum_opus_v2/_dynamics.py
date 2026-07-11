"""
Emotional and temporal dynamics shared by v2 regions.

This module holds the per-emotion biological parameters (onset, decay,
homeostatic baseline), the three-speed (fast/medium/slow) emotion-state
class, and the subjective-time engine. The Limbic region wraps the emotion
state; the Temporal region wraps the time engine.

Originally these lived under v1's `magnum_opus/components.py` and `config.py`.
They are inlined here so v2 has no dependency on the legacy package.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Per-emotion biological parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmotionConfig:
    """Biological parameters for a single emotion dimension."""
    onset_rate: float = 0.3       # How fast it rises (0-1)
    decay_rate: float = 0.05      # How fast it fades per step
    baseline: float = 0.0         # Resting level (homeostatic set point)
    min_val: float = -1.0
    max_val: float = 1.0


EMOTION_CONFIGS: Dict[str, EmotionConfig] = {
    "calm":      EmotionConfig(onset_rate=0.15, decay_rate=0.02, baseline=0.3),
    "curious":   EmotionConfig(onset_rate=0.4,  decay_rate=0.08, baseline=0.1),
    "desperate": EmotionConfig(onset_rate=0.5,  decay_rate=0.03, baseline=0.0),
    "joy":       EmotionConfig(onset_rate=0.35, decay_rate=0.06, baseline=0.05),
    "anger":     EmotionConfig(onset_rate=0.6,  decay_rate=0.04, baseline=0.0),
    "fear":      EmotionConfig(onset_rate=0.7,  decay_rate=0.05, baseline=0.0),
    "surprise":  EmotionConfig(onset_rate=0.9,  decay_rate=0.15, baseline=0.0),
    "trust":     EmotionConfig(onset_rate=0.1,  decay_rate=0.01, baseline=0.2),
    "sadness":   EmotionConfig(onset_rate=0.2,  decay_rate=0.02, baseline=0.0),
    "disgust":   EmotionConfig(onset_rate=0.5,  decay_rate=0.06, baseline=0.0),
}


# Hand-authored interaction matrix — the FALLBACK. When a profile carries
# Mirror dynamics, this is replaced by a matrix FITTED at profile-creation
# time from the model's implied emotional trajectories (correlation of one
# emotion's level with another's next-beat change — see mirror.py). No
# cosine-geometry derivation exists; a comment here claimed one for a long
# time, and the Era-4 audit corrected it.
EMOTION_INTERACTIONS: Dict[Tuple[str, str], float] = {
    ("desperate", "calm"):    -0.5,
    ("joy", "desperate"):     -0.3,
    ("calm", "desperate"):    -0.4,
    ("desperate", "joy"):     -0.2,
    ("curious", "calm"):       0.1,
    ("anger", "calm"):        -0.6,
    ("anger", "fear"):         0.2,
    ("fear", "desperate"):     0.3,
    ("fear", "anger"):         0.15,
    ("joy", "trust"):          0.2,
    ("sadness", "joy"):       -0.4,
    ("joy", "sadness"):       -0.3,
    ("trust", "calm"):         0.15,
    ("disgust", "trust"):     -0.3,
    ("surprise", "curious"):   0.3,
    ("calm", "anger"):        -0.3,
    ("calm", "fear"):         -0.2,
}


# Three speeds of emotional processing — fast reactions, medium mood,
# slow temperament.
FAST_ONSET_MULT = 3.0
FAST_DECAY_MULT = 5.0
MEDIUM_ONSET_MULT = 1.0
MEDIUM_DECAY_MULT = 1.0
SLOW_ONSET_MULT = 0.1
SLOW_DECAY_MULT = 0.1

FAST_WEIGHT = 0.5
MEDIUM_WEIGHT = 0.35
SLOW_WEIGHT = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Subjective-time signals (snapshot of internal state)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InternalTimeSignals:
    """Internal-state inputs to the subjective-time computation.

    The system perceives time through how much it has changed — residual
    norm fade, emotional distance from baseline, freshness of the last
    interaction — not by reading a clock.
    """
    residual_norm: float = 0.0
    residual_norm_at_last_interaction: float = 0.0
    emotional_distance: float = 0.0
    interaction_freshness: float = 1.0
    memory_avg_importance: float = 0.0
    memory_avg_importance_at_last_interaction: float = 0.0
    steps_since_interaction: int = 0


@dataclass
class TemporalConfig:
    """Weights for blending the five subjective-time signals."""
    internal_time_residual_weight: float = 0.3
    internal_time_emotional_weight: float = 0.25
    internal_time_freshness_weight: float = 0.2
    internal_time_step_weight: float = 0.15
    internal_time_memory_weight: float = 0.1
    internal_time_step_scale: float = 20.0  # steps for ~1.0 subjective time


# ─────────────────────────────────────────────────────────────────────────────
# Multi-speed emotion state
# ─────────────────────────────────────────────────────────────────────────────

class MultiSpeedEmotionalState:
    """Three-channel emotion state with biological onset, decay, homeostasis,
    and cross-emotion interactions.

    Fast:   per-token reactions (high onset, high decay) — captures WHAT
    Medium: per-conversation mood (moderate dynamics) — captures WHAT KIND
    Slow:   cross-conversation temperament (glacial change) — captures WHY
    """

    def __init__(
        self,
        emotion_names: List[str],
        configs: Optional[Dict[str, EmotionConfig]] = None,
        interactions: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.names = [e for e in emotion_names if not e.startswith("temporal_")]
        self.configs: Dict[str, EmotionConfig] = {}
        for n in self.names:
            if configs and n in configs:
                self.configs[n] = configs[n]
            elif n in EMOTION_CONFIGS:
                self.configs[n] = EMOTION_CONFIGS[n]
            else:
                self.configs[n] = EmotionConfig()

        self.interactions = dict(interactions) if interactions is not None else dict(EMOTION_INTERACTIONS)

        self.fast = {n: self.configs[n].baseline for n in self.names}
        self.medium = {n: self.configs[n].baseline for n in self.names}
        self.slow = {n: self.configs[n].baseline for n in self.names}

    def stimulate(self, emotion: str, intensity: float) -> None:
        """Apply stimulus across all three speeds with appropriate scaling."""
        if emotion not in self.configs:
            return
        c = self.configs[emotion]

        for speed, mult in [(self.fast, FAST_ONSET_MULT),
                            (self.medium, MEDIUM_ONSET_MULT),
                            (self.slow, SLOW_ONSET_MULT)]:
            delta = intensity * c.onset_rate * mult
            speed[emotion] = float(np.clip(speed[emotion] + delta, c.min_val, c.max_val))

        # Interaction effects (primarily on the medium channel).
        for (src, tgt), factor in self.interactions.items():
            if src == emotion and tgt in self.medium:
                interaction = intensity * c.onset_rate * factor
                tgt_cfg = self.configs.get(tgt, EmotionConfig())
                self.medium[tgt] = float(np.clip(
                    self.medium[tgt] + interaction, tgt_cfg.min_val, tgt_cfg.max_val,
                ))

    def decay_step(self, dt: float = 1.0) -> None:
        """Time-based decay toward homeostatic baseline at all three speeds."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Temporal engine (subjective time)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEngine:
    """Subjective time derived from internal state changes, not wall clocks.

    Subjective time accumulates from residual norm fade, emotional drift,
    interaction staleness, step count, and memory-importance decay. The
    output is a single scalar that callers (and the steering vector) consume
    as "how stale does this moment feel?".
    """

    def __init__(
        self,
        temporal_vectors: Dict[str, torch.Tensor],
        config: Optional[TemporalConfig] = None,
    ):
        self.vectors = temporal_vectors
        cfg = config or TemporalConfig()

        self.w_residual = cfg.internal_time_residual_weight
        self.w_emotional = cfg.internal_time_emotional_weight
        self.w_freshness = cfg.internal_time_freshness_weight
        self.w_step = cfg.internal_time_step_weight
        self.w_memory = cfg.internal_time_memory_weight
        self.step_scale = cfg.internal_time_step_scale

        self.residual_norm_at_interaction: float = 0.0
        self.importance_at_interaction: float = 0.0
        self.steps_since_interaction: int = 0

        self.gaps: List[int] = []
        self._session_start_wall: float = time.time()

    def mark_interaction(self, signals: InternalTimeSignals) -> None:
        if self.steps_since_interaction > 0:
            self.gaps.append(self.steps_since_interaction)
            if len(self.gaps) > 50:
                self.gaps = self.gaps[-50:]
        self.residual_norm_at_interaction = signals.residual_norm
        self.importance_at_interaction = signals.memory_avg_importance
        self.steps_since_interaction = 0

    def tick(self) -> None:
        self.steps_since_interaction += 1

    def subjective_elapsed(self, signals: InternalTimeSignals) -> float:
        """0.0 (just interacted) to ~5.0+ (very stale)."""
        if self.residual_norm_at_interaction > 0.01:
            residual_fade = max(0.0, 1.0 - signals.residual_norm / self.residual_norm_at_interaction)
        else:
            residual_fade = 0.0

        emo_dist = min(1.0, signals.emotional_distance * 5.0)
        staleness = 1.0 - signals.interaction_freshness
        step_signal = signals.steps_since_interaction / max(self.step_scale, 1.0)

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
        return float(max(0.0, min(1.0, 1.0 - (avg - 2.0) / 58.0)))

    def compute_steering(self, signals: InternalTimeSignals) -> Optional[torch.Tensor]:
        """Temporal steering vector blended from recency + urgency directions."""
        if not self.vectors:
            return None

        subj = self.subjective_elapsed(signals)
        recency = math.exp(-subj * 0.7)

        ref = next(iter(self.vectors.values()))
        combined = torch.zeros_like(ref)

        if "temporal_recency" in self.vectors:
            combined = combined + (recency * 2 - 1) * self.vectors["temporal_recency"]
        if "temporal_urgency" in self.vectors:
            pace_val = self.pace()
            combined = combined + (pace_val * 2 - 1) * 0.5 * self.vectors["temporal_urgency"]

        return combined
