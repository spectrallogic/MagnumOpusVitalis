"""
Neuromodulator layer — chemistry that does work.

Point 7 of the user's vision:
    "Things like stress and other chemical processes need to be useful in
     some way, otherwise its pointless and just feels like we are
     simulating for no reason."

Four slow scalars in [0, ~2]:

  cortisol          — stress / threat reactivity. Driven up by strong
                      negative emotions and high bus divergence held over
                      time. Drives down naturally during calm periods.
  dopamine          — reward / novelty. Driven up by knowledge sparks that
                      align with current direction (concept "lands"), and
                      by surprise promotions in subconscious. Drives down
                      naturally.
  serotonin         — sustained well-being. Driven up by long calm + low
                      pressure periods. Drives down by sustained negative
                      emotion or pressure.
  norepinephrine    — arousal / gain. Driven up by high bus velocity
                      (lots of motion in latent space). Drives down when
                      bus is quiet.

Critically, these are NOT extra vectors added to steering. They are
PARAMETER MODULATORS that change *how* other regions behave:

  cortisol         → ↑ Limbic.fast_onset_rate, ↓ DefaultMode.idle_drift_amp,
                     ↑ Salience weight on memories tagged with negative
                     emotions.
  dopamine         → ↓ Executive.speech_threshold,
                     ↑ KnowledgeSparks.fire_probability,
                     ↑ SubconsciousStack.l2_surprise_probability.
  serotonin        → shifts Limbic baselines slightly toward calm,
                     damps bus velocity (more stability).
  norepinephrine   → ↑ bus.temperature globally,
                     ↑ all region perturbation magnitudes uniformly.

`NeuromodState` is the value object passed to every Region.step(). Regions
read scalars and apply multipliers themselves — that keeps the modulation
explicit at each region's site, not magic distributed mutation.

`NeuromodulatorRegion` runs on the slow clock (~30s). It samples the bus +
limbic + executive snapshots once and updates the four scalars accordingly.
Regions can also emit events directly via `neuromod.bump(...)` for
fine-grained reactivity (e.g. KnowledgeSparks bumps dopamine when a spark
aligns).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


@dataclass
class NeuromodState:
    cortisol: float = 0.0
    dopamine: float = 0.0
    serotonin: float = 0.0
    norepinephrine: float = 0.0

    # Baselines each scalar drifts toward
    cortisol_baseline: float = 0.0
    dopamine_baseline: float = 0.2
    serotonin_baseline: float = 0.5
    norepinephrine_baseline: float = 0.1

    # Drift speed per slow tick (~30s). 0.1 means halflife ~5 slow ticks (~2.5min).
    drift_speed: float = 0.15

    # Hard ceilings
    max_value: float = 2.0
    min_value: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def bump(self, name: str, delta: float) -> None:
        """Atomic increment. Used by event injection paths."""
        with self._lock:
            cur = getattr(self, name)
            new = max(self.min_value, min(self.max_value, cur + float(delta)))
            setattr(self, name, new)

    def drift_one_tick(self) -> None:
        """Drift each scalar toward its baseline by drift_speed."""
        with self._lock:
            for name, baseline_attr in [
                ("cortisol", "cortisol_baseline"),
                ("dopamine", "dopamine_baseline"),
                ("serotonin", "serotonin_baseline"),
                ("norepinephrine", "norepinephrine_baseline"),
            ]:
                cur = getattr(self, name)
                base = getattr(self, baseline_attr)
                new = cur + (base - cur) * self.drift_speed
                setattr(self, name, max(self.min_value, min(self.max_value, new)))

    # ------------------------------------------------------------------
    # Read helpers — short names for region code readability
    # ------------------------------------------------------------------
    def cortisol_gain(self, scale: float = 1.0) -> float:
        return 1.0 + self.cortisol * scale

    def cortisol_suppress(self, scale: float = 1.0) -> float:
        return max(0.0, 1.0 - self.cortisol * scale)

    def dopamine_drop(self, scale: float = 1.0) -> float:
        return max(0.0, 1.0 - self.dopamine * scale)

    def dopamine_boost(self, scale: float = 1.0) -> float:
        return 1.0 + self.dopamine * scale

    def norepinephrine_gain(self, scale: float = 1.0) -> float:
        return 1.0 + self.norepinephrine * scale

    def serotonin_damp(self, scale: float = 1.0) -> float:
        # Higher serotonin → more damping → 0.5 means "half as agitated"
        return max(0.05, 1.0 - self.serotonin * scale * 0.5)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cortisol": self.cortisol,
                "dopamine": self.dopamine,
                "serotonin": self.serotonin,
                "norepinephrine": self.norepinephrine,
            }


# ---------------------------------------------------------------------------
# Slow-clock region that updates the four scalars from observed state.
# ---------------------------------------------------------------------------

class NeuromodulatorRegion(Region):
    name = "neuromodulator"
    clock = "slow"

    def __init__(
        self,
        neuromod: NeuromodState,
        limbic_provider=None,    # callable returning Limbic snapshot dict, or None
        executive_provider=None, # callable returning Executive snapshot dict, or None
        # event sensitivities — how much each observation shifts each scalar
        cortisol_per_negative_emotion: float = 0.6,
        cortisol_per_high_divergence: float = 0.4,
        serotonin_per_calm_minute: float = 0.3,
        norepinephrine_per_high_velocity: float = 0.5,
        dopamine_drift_when_idle: float = 0.0,
    ):
        self.neuromod = neuromod
        self.limbic_provider = limbic_provider
        self.executive_provider = executive_provider

        self.cort_neg = float(cortisol_per_negative_emotion)
        self.cort_div = float(cortisol_per_high_divergence)
        self.sero_calm = float(serotonin_per_calm_minute)
        self.norep_vel = float(norepinephrine_per_high_velocity)
        self.dopa_idle = float(dopamine_drift_when_idle)

        # Calm-streak tracker (in slow ticks)
        self._calm_streak: int = 0

    # ------------------------------------------------------------------
    # Region step — runs ~once per 30s
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[object]:
        # 1. Drift toward baselines first (slow homeostasis)
        self.neuromod.drift_one_tick()

        # 2. Observe and inject from current state
        div = float(bus.divergence_from_baseline())
        vel = float(bus.velocity.norm())

        # cortisol from negative-emotion magnitude (anger/fear/sadness/disgust/desperate)
        if self.limbic_provider:
            try:
                limb = self.limbic_provider() or {}
                blend = limb.get("blended", {}) if isinstance(limb, dict) else {}
                neg = sum(
                    abs(blend.get(e, 0.0))
                    for e in ("anger", "fear", "sadness", "disgust", "desperate")
                )
                if neg > 0.3:
                    self.neuromod.bump("cortisol", self.cort_neg * (neg - 0.3))
            except Exception:  # noqa: BLE001
                pass

        # cortisol from high sustained divergence
        if div > 0.25:
            self.neuromod.bump("cortisol", self.cort_div * (div - 0.25))

        # norepinephrine from high velocity
        if vel > 0.3:
            self.neuromod.bump("norepinephrine", self.norep_vel * (vel - 0.3))

        # serotonin: requires sustained calm (low div + low vel)
        if div < 0.1 and vel < 0.15:
            self._calm_streak += 1
            if self._calm_streak >= 2:  # ~1 minute of calm
                self.neuromod.bump("serotonin", self.sero_calm)
        else:
            self._calm_streak = 0

        # dopamine: gentle drift (other regions bump it on resonance events)
        if self.dopa_idle != 0.0:
            self.neuromod.bump("dopamine", self.dopa_idle)

        return None
