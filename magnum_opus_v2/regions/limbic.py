"""
Limbic region — wraps v1's MultiSpeedEmotionalState as a Region.

Three channels (fast/medium/slow) with biological onset/decay rates from
v1's config. Each flow tick:
  1. Decay emotions toward homeostatic baselines (scaled to wall-clock).
  2. Blend the three speeds into a single weighted emotion vector.
  3. Sum the named profile vectors weighted by blended values.
  4. Return that summed vector as a perturbation to the bus.

External callers (e.g. user message handler) use `stimulate(emotion,
intensity)` to inject events. Decay timing is preserved from v1: v1 ran
decay_step(dt=1.0) inside a 2s heartbeat, so decay per second was rate * 0.5.
We replicate that by scaling the per-tick dt with TICK_TO_V1_STEP_SCALE.
"""

import threading
from typing import Dict, List, Optional

import torch

from magnum_opus.components import MultiSpeedEmotionalState

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


# v1 ran one decay_step(dt=1.0) per ~2s heartbeat. To preserve the same
# per-second decay behavior, treat 1 second of real time as 0.5 v1 steps.
TICK_TO_V1_STEP_SCALE = 0.5


class Limbic(Region):
    """Three-channel emotion vector that perturbs the bus every flow tick."""

    name = "limbic"
    clock = "flow"

    def __init__(
        self,
        emotion_vectors: Dict[str, torch.Tensor],  # from profile.vectors
        device: str = "cpu",
        # Lower than v1's 4.0 because v1 applied steering once per
        # generation, while v2 runs Limbic.step every 50ms (~20×/s) and
        # the attractor + integration accumulate over many ticks.
        steering_strength: float = 1.0,
        emotion_names: Optional[List[str]] = None,
        configs=None,
        interactions=None,
    ):
        # Strip temporal_* names — those are handled by the Temporal region.
        names = emotion_names or [
            n for n in emotion_vectors.keys() if not n.startswith("temporal_")
        ]
        self._state = MultiSpeedEmotionalState(
            emotion_names=names, configs=configs, interactions=interactions,
        )

        # Cache emotion vectors on the right device, in float32, for the blend.
        self._emo_vecs: Dict[str, torch.Tensor] = {
            n: emotion_vectors[n].to(device).float()
            for n in names
            if n in emotion_vectors
        }
        self.device = device
        self.steering_strength = float(steering_strength)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Personality baseline vector — what the bus settles toward at rest.
    # Computed from EmotionConfig.baseline values blended into the emotion
    # vectors. Engine assembly should call bus.set_baseline(this) so the
    # substrate naturally homeostates at the personality, not at zero.
    # ------------------------------------------------------------------
    def baseline_vector(self) -> torch.Tensor:
        with self._lock:
            if not self._emo_vecs:
                return torch.zeros(0, device=self.device)
            ref = next(iter(self._emo_vecs.values()))
            accum = torch.zeros_like(ref)
            for n, vec in self._emo_vecs.items():
                cfg = self._state.configs.get(n)
                if cfg is None:
                    continue
                # Blend at the same fast/medium/slow weights the steady-state
                # would produce if all channels sat at their baselines.
                weight = cfg.baseline  # all three channels share the same baseline at rest
                accum = accum + weight * vec
        return (accum * self.steering_strength).to(self.device)

    # ------------------------------------------------------------------
    # External interface — called from request handlers
    # ------------------------------------------------------------------
    def stimulate(self, emotion: str, intensity: float,
                  neuromod: object = None) -> None:
        # Cortisol amplifies threat-related onset; serotonin damps it.
        scale = 1.0
        if neuromod is not None:
            if hasattr(neuromod, "cortisol_gain") and emotion in (
                "fear", "anger", "desperate", "sadness", "disgust",
            ):
                scale *= neuromod.cortisol_gain(scale=0.6)
            if hasattr(neuromod, "serotonin_damp"):
                scale *= neuromod.serotonin_damp(scale=0.4)
        with self._lock:
            self._state.stimulate(emotion, intensity * scale)

    def stimulate_many(self, stim: Dict[str, float],
                       neuromod: object = None) -> None:
        for emo, mag in stim.items():
            self.stimulate(emo, float(mag), neuromod=neuromod)

    def snapshot(self) -> dict:
        with self._lock:
            return self._state.snapshot()

    # ------------------------------------------------------------------
    # Region step (called by flow clock every ~50ms)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        with self._lock:
            self._state.decay_step(dt=dt * TICK_TO_V1_STEP_SCALE)
            blended = self._state.get_blended()

        if not self._emo_vecs:
            return None
        accum = torch.zeros_like(next(iter(self._emo_vecs.values())))
        for emo, weight in blended.items():
            vec = self._emo_vecs.get(emo)
            if vec is None or weight == 0.0:
                continue
            accum = accum + float(weight) * vec

        # Neuromodulator: norepinephrine globally amplifies emotional gain.
        gain = 1.0
        if neuromod is not None and hasattr(neuromod, "norepinephrine_gain"):
            gain = neuromod.norepinephrine_gain(scale=0.4)
        return (accum * self.steering_strength * gain).to(bus.device)
