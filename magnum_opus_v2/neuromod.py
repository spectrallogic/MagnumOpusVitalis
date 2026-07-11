"""
Modulation layer — global gain channels that do work.

Design rule: modulation must do real work. A channel with no wired
consumer is simulation for its own sake, and does not ship.

Four slow scalars in [0, ~2], named by FUNCTION (the hormone names this
layer once borrowed claimed a biology the code never implemented; the
mechanisms below are exactly what existed under those names — this was a
rename, not a redesign):

  stress   — threat reactivity. Raised by strong negative emotions, high
             bus divergence held over time, and imagined risk.
  reward   — incentive. Raised by abstraction-ladder novelty spikes and
             high-benefit imagined futures.
  calm     — stability. Raised only by sustained calm (low divergence +
             low velocity) held across slow ticks.
  arousal  — alertness / gain. Raised by high bus velocity, self-model
             surprise, and scene shifts.

There are NO active down-drivers: every channel rises on measured events
and otherwise DRIFTS toward its disclosed baseline (drift_one_tick). Any
prose claiming active suppression would overclaim — the homeostasis here
is drift, nothing more.

Critically, these are NOT extra vectors added to steering. They are
PARAMETER MODULATORS that change *how* other regions behave. Every effect
listed here is implemented — nothing on this list is decorative:

  stress   → ↑ Limbic threat-emotion onset (fear/anger/desperate/
               sadness/disgust stimulation gain),
             ↑ SpeculativeFutures risk weight (imagined danger looms
               larger under stress).
  reward   → ↓ Executive speech threshold (reward loosens the tongue),
             ↑ KnowledgeSparks fire probability,
             ↑ SubconsciousStack L2 surprise probability,
             ↑ SpeculativeFutures benefit weight.
  calm     → ↑ Limbic decay toward homeostatic baseline (emotional
               stability),
             ↓ Limbic stimulation gain,
             damps bus velocity (calmer substrate).
  arousal  → ↑ bus noise temperature,
             ↑ Limbic output gain,
             ↑ SubconsciousStack L3 intrusive loudness.

`NeuromodState` is the value object passed to every Region.step(). Regions
read scalars and apply multipliers themselves — that keeps the modulation
explicit at each region's site, not magic distributed mutation. ("Neuro-
modulation" here names that engineering pattern — global scalars gating
region gains — not a claim about brain chemistry.)

`NeuromodulatorRegion` runs on the slow clock (~30s). It samples the bus +
limbic snapshots once and updates the four scalars accordingly. Regions
can also emit events directly via `neuromod.bump(...)` for fine-grained
reactivity (e.g. AbstractionLadder bumps reward on a novelty spike;
SelfModel and SituationModel bump arousal on surprise).

Deliberate non-goal, on the record: the `Limbic` region keeps its name.
It is an anatomy metaphor for a region that genuinely holds and decays
blended emotion state — it claims no absent mechanism, and its name
crosses the primordium substrate boundary (MoodField subclasses it).
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


@dataclass
class NeuromodState:
    stress: float = 0.0
    reward: float = 0.0
    calm: float = 0.0
    arousal: float = 0.0

    # Baselines each scalar drifts toward
    stress_baseline: float = 0.0
    reward_baseline: float = 0.2
    calm_baseline: float = 0.5
    arousal_baseline: float = 0.1

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
                ("stress", "stress_baseline"),
                ("reward", "reward_baseline"),
                ("calm", "calm_baseline"),
                ("arousal", "arousal_baseline"),
            ]:
                cur = getattr(self, name)
                base = getattr(self, baseline_attr)
                new = cur + (base - cur) * self.drift_speed
                setattr(self, name, max(self.min_value, min(self.max_value, new)))

    # ------------------------------------------------------------------
    # Read helpers — short names for region code readability
    # ------------------------------------------------------------------
    def stress_gain(self, scale: float = 1.0) -> float:
        return 1.0 + self.stress * scale

    def reward_drop(self, scale: float = 1.0) -> float:
        return max(0.0, 1.0 - self.reward * scale)

    def reward_boost(self, scale: float = 1.0) -> float:
        return 1.0 + self.reward * scale

    def arousal_gain(self, scale: float = 1.0) -> float:
        return 1.0 + self.arousal * scale

    def calm_damp(self, scale: float = 1.0) -> float:
        # Higher calm → more damping → 0.5 means "half as agitated"
        return max(0.05, 1.0 - self.calm * scale * 0.5)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stress": self.stress,
                "reward": self.reward,
                "calm": self.calm,
                "arousal": self.arousal,
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
        # event sensitivities — how much each observation shifts each scalar
        stress_per_negative_emotion: float = 0.6,
        stress_per_high_divergence: float = 0.4,
        calm_per_calm_minute: float = 0.3,
        arousal_per_high_velocity: float = 0.5,
    ):
        self.neuromod = neuromod
        self.limbic_provider = limbic_provider

        self.stress_neg = float(stress_per_negative_emotion)
        self.stress_div = float(stress_per_high_divergence)
        self.calm_gain = float(calm_per_calm_minute)
        self.arousal_vel = float(arousal_per_high_velocity)

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

        # stress from negative-emotion magnitude (anger/fear/sadness/disgust/desperate)
        if self.limbic_provider:
            try:
                limb = self.limbic_provider() or {}
                blend = limb.get("blended", {}) if isinstance(limb, dict) else {}
                neg = sum(
                    abs(blend.get(e, 0.0))
                    for e in ("anger", "fear", "sadness", "disgust", "desperate")
                )
                if neg > 0.3:
                    self.neuromod.bump("stress", self.stress_neg * (neg - 0.3))
            except Exception:  # noqa: BLE001
                pass

        # stress from high sustained divergence
        if div > 0.25:
            self.neuromod.bump("stress", self.stress_div * (div - 0.25))

        # arousal from high velocity
        if vel > 0.3:
            self.neuromod.bump("arousal", self.arousal_vel * (vel - 0.3))

        # calm: requires sustained stillness (low div + low vel)
        if div < 0.1 and vel < 0.15:
            self._calm_streak += 1
            if self._calm_streak >= 2:  # ~1 minute of calm
                self.neuromod.bump("calm", self.calm_gain)
        else:
            self._calm_streak = 0

        # reward has no slow-clock driver: it moves only on events
        # (abstraction novelty, high-benefit futures) plus baseline drift
        return None
