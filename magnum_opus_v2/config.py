"""
v2 configuration. All parameters in one place.

Three nested config objects so substrate-only tests don't need to construct
region or model configs they aren't using yet.
"""

from dataclasses import dataclass, field


@dataclass
class BusConfig:
    """LatentBus dynamics."""
    # How strongly state is pulled toward attractors per second.
    # Too high → state collapses to baseline (no motion). Too low → drifts away.
    attractor_strength: float = 0.6

    # Velocity damping per flow tick (multiplicative).
    # Lower → more momentum (ball-of-liquid feel). Higher → quicker to settle.
    velocity_damping: float = 0.85

    # Multiplier on the raw noise term (then × bus.temperature).
    noise_scale: float = 0.05

    # Hard ceiling on |state| so it can't blow up under bad regions.
    max_state_norm: float = 8.0

    # Hard ceiling on number of attractors (oldest non-baseline drops).
    max_attractors: int = 6

    # Default starting temperature (modulated later by neuromod/norepinephrine).
    initial_temperature: float = 1.0


@dataclass
class ClockConfig:
    """Multi-rate clock periods, in seconds."""
    # ~50ms — pure-math substrate update (attractor dynamics, layered
    # subconscious propagation, neuromod drift). No model passes. Cheap.
    flow_dt_seconds: float = 0.05

    # ~200ms — thought residual decay, communicative pressure, subjective time.
    perception_dt_seconds: float = 0.20

    # ~1.5s — silent forward passes (idle drift), knowledge spark firing.
    expensive_dt_seconds: float = 1.50

    # ~30s — neuromod baseline shifts, false memory consolidation.
    slow_dt_seconds: float = 30.0


@dataclass
class V2Config:
    """Top-level v2 config. Compose nested configs."""
    hidden_dim: int = 768  # default for gpt2; overridden when a model is attached
    device: str = "cpu"
    bus: BusConfig = field(default_factory=BusConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    # Whether to print per-clock errors. Useful during dev.
    verbose_errors: bool = True
