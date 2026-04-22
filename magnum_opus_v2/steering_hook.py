"""
Steering hook for v2 — re-exports v1's SteeringHook (zero changes) and adds
a BusSteeringDriver that converts bus.state into the steering vector the
hook injects.

The hook itself is exactly the v1 implementation: register_forward_hook on
the target transformer layer, capture hidden states, optionally add a
steering vector. Multi-arch (GPT-2, LLaMA/Mistral). See
magnum_opus/engine.py:32-92 for the implementation.

The BusSteeringDriver is the bridge: every time the model is about to
generate, the driver reads bus.state, optionally smooths it, scales by
steering_strength, and writes it to the hook. The model never sees the
bus directly.
"""

import threading
from typing import Optional

import torch

# Reuse v1's hook unchanged. If/when v2 needs a different hook shape, copy
# the class here and diverge — but right now they're the same.
from magnum_opus.engine import SteeringHook  # noqa: F401  re-exported

from magnum_opus_v2.bus import LatentBus


class BusSteeringDriver:
    """
    Reads bus.state and produces the steering vector the hook injects.

    The simplest reading: steering_vector = bus.state * steering_strength.
    Since the bus lives in the model's hidden_dim space and is shaped by
    regions that already emit perturbations along emotion-vector
    directions, the bus state IS already a meaningful steering vector.

    Optionally smooths over time (exponential moving average) so the hook
    sees a less jittery signal than the raw 50ms-tick bus.
    """

    def __init__(
        self,
        bus: LatentBus,
        steering_strength: float = 1.0,
        smoothing: float = 0.0,  # 0 = use raw bus.state, 0.9 = heavy EMA
    ):
        self.bus = bus
        self.steering_strength = float(steering_strength)
        self.smoothing = float(smoothing)
        self._smoothed: Optional[torch.Tensor] = None
        self._lock = threading.Lock()

    def read(self) -> torch.Tensor:
        """
        Read current bus.state, apply smoothing + strength, return the
        steering vector to hand to the SteeringHook.
        """
        with self._lock:
            current = self.bus.state.detach().clone()
            if self.smoothing <= 0.0 or self._smoothed is None:
                self._smoothed = current
            else:
                self._smoothed = (
                    self.smoothing * self._smoothed
                    + (1.0 - self.smoothing) * current
                )
            return self._smoothed * self.steering_strength

    def reset_smoothing(self) -> None:
        with self._lock:
            self._smoothed = None
