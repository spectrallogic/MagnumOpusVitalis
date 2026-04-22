"""
Region protocol — the uniform shape every brain region implements.

A region reads the bus, runs its internal dynamics + noise, and returns a
perturbation tensor (or None) that gets added to the bus state. Regions on
the "flow" clock have their perturbations summed and applied through the
substrate's velocity update. Regions on slower clocks are called for side
effects (and may also write directly to the bus via add_attractor, etc.).

This is the abstraction that unblocks point 6 of the vision:
    "We apply the logics from above in different manners with other areas
     of the brain."

Add a new region by subclassing Region and giving it a clock label.
"""

from typing import Literal, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from magnum_opus_v2.bus import LatentBus

ClockName = Literal["flow", "perception", "expensive", "slow"]


class Region:
    """Base class for all brain regions."""

    name: str = "region"
    clock: ClockName = "flow"

    def step(
        self,
        bus: "LatentBus",
        neuromod: object,  # NeuromodState — typed loosely until that lands
        dt: float,
    ) -> Optional[torch.Tensor]:
        """
        One tick of this region. Read bus.state, do internal work, return a
        perturbation tensor of shape (hidden_dim,) on the bus's device — or
        None to write nothing this tick.

        Side effects (writing memories, mutating neuromod, calling
        bus.add_attractor) are allowed. The returned tensor is what gets
        summed into the substrate's velocity update.
        """
        return None


class NoOpRegion(Region):
    """A region that does nothing. Useful for substrate-only verification."""

    def __init__(self, name: str = "noop", clock: ClockName = "flow"):
        self.name = name
        self.clock = clock


class PerturbationRegion(Region):
    """
    Test region: emits a constant perturbation (small random vector) every
    flow tick. Used by demo_v2_substrate.py to confirm the bus actually
    responds to inputs and that attractor dynamics pull it back.
    """

    def __init__(
        self,
        hidden_dim: int,
        device: str = "cpu",
        magnitude: float = 0.5,
        seed: int = 0,
        name: str = "test_perturb",
        clock: ClockName = "flow",
    ):
        self.name = name
        self.clock = clock
        self._device = device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        # Fixed direction so we can verify the bus moves toward it then back.
        v = torch.randn(hidden_dim, generator=gen)
        v = v / (v.norm() + 1e-8)
        self.direction = (v * magnitude).to(device)

    def step(self, bus, neuromod, dt):
        return self.direction
