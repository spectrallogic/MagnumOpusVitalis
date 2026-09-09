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

    # Default starting temperature (modulated later by neuromod/arousal).
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
class SpeculativeConfig:
    """Hypothetical rollouts with cooperative compute limits.

    Depth counts recursively conditioned events; rollout_tokens counts
    tokens per event. One in-flight forward can exceed the time budget.
    """
    n_futures: int = 4
    rollout_tokens: int = 14          # imagined depth (was 6): futures are phrases
    rollout_budget_ms: float = 250.0  # wall-clock cap on one candidate's rollout
    chained_continuation_tokens: int = 8  # WORLD mode reads its own trajectory further
    max_depth: int = 2
    branching_factor: int = 2
    beam_width: int = 2
    max_nodes: int = 12
    max_total_tokens: int = 128
    round_budget_ms: float = 500.0
    discount: float = 0.8


@dataclass
class V2Config:
    """Top-level v2 config. Compose nested configs."""
    hidden_dim: int = 768  # default for gpt2; overridden when a model is attached
    device: str = "cpu"
    bus: BusConfig = field(default_factory=BusConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    spec: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    # Whether to print per-clock errors. Useful during dev.
    verbose_errors: bool = True
