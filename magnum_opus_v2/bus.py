"""
LatentBus — the shared substrate.

A single continuously-evolving latent state vector with attractor dynamics.
Every region reads from it and writes a perturbation back. This is the
"playground" the user described: regions don't get told to do things in
order, they all share this one substrate.

Update rule (per flow tick, dt ~50ms):

    perturbation   = sum(region.step(bus, neuromod, dt) for flow-clock regions)
    attractor_pull = mean over attractors of  weight_i * (a_i - state)
    noise          = randn * temperature * noise_scale
    velocity       = damping * velocity + dt * (perturbation + attractor_pull*K + noise)
    state          = state + dt * velocity
    state          = clip_norm(state, max_norm)

This Ornstein-Uhlenbeck-like flow gives "ball of liquid": continuous motion
that always wants to settle but gets perturbed by everything else.
"""

import threading
import time
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from magnum_opus_v2.config import BusConfig


class LatentBus:
    def __init__(
        self,
        hidden_dim: int,
        device: str = "cpu",
        config: Optional[BusConfig] = None,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.config = config or BusConfig()

        self.state = torch.zeros(hidden_dim, device=device)
        self.velocity = torch.zeros(hidden_dim, device=device)

        # (vector, weight). Index 0 is the calm baseline (kept across pruning).
        self.attractors: List[Tuple[torch.Tensor, float]] = [
            (torch.zeros(hidden_dim, device=device), 1.0),
        ]

        self.temperature = float(self.config.initial_temperature)

        # Bookkeeping
        self.tick_count = 0
        self.wall_time_started = time.monotonic()
        self.last_flow_time = self.wall_time_started

        # threading.Lock — flow runs in its own thread (asyncio inside),
        # generation runs in the Flask handler thread. Both touch the bus.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Substrate update — called ONLY by the flow clock (50ms).
    # Slower clocks call run_side_effects() instead so they don't
    # double-integrate the substrate.
    # ------------------------------------------------------------------
    def step(
        self,
        regions: Iterable,  # iterable of Region (flow-clock only)
        neuromod: object,
        dt: float,
        verbose_errors: bool = False,
    ) -> dict:
        """One flow tick. Returns a small snapshot dict for logging."""
        with self._lock:
            perturbation = self._collect_perturbations(
                regions, neuromod, dt, verbose_errors
            )
            attractor_pull = self._attractor_pull()

            # Neuromod modulation of substrate dynamics:
            # - norepinephrine raises effective temperature (more arousal noise)
            # - serotonin damps velocity (more stability)
            temp_gain = 1.0
            damp_extra = 1.0
            if neuromod is not None:
                if hasattr(neuromod, "norepinephrine_gain"):
                    temp_gain = neuromod.norepinephrine_gain(scale=0.6)
                if hasattr(neuromod, "serotonin_damp"):
                    damp_extra = neuromod.serotonin_damp(scale=0.4)

            # Normalize raw noise by sqrt(hidden_dim) so per-tick noise has
            # expected norm ~= temperature * noise_scale regardless of model
            # size. Without this, a 4096-dim model would have ~2.3x the
            # noise of a 768-dim one for the same config.
            noise = (
                torch.randn_like(self.state)
                * (self.temperature * temp_gain * self.config.noise_scale)
                / (self.hidden_dim ** 0.5)
            )

            self.velocity = (
                self.config.velocity_damping * damp_extra * self.velocity
                + dt * (perturbation + attractor_pull + noise)
            )
            self.state = self.state + dt * self.velocity

            # Clip norm so a runaway region can't blow the substrate up.
            n = self.state.norm()
            if n > self.config.max_state_norm:
                self.state = self.state * (self.config.max_state_norm / (n + 1e-8))

            self.tick_count += 1
            self.last_flow_time = time.monotonic()

        return self.snapshot()

    def run_side_effects(
        self,
        regions: Iterable,
        neuromod: object,
        dt: float,
        verbose_errors: bool = False,
    ) -> None:
        """
        Called by perception / expensive / slow clocks. Calls each region's
        step() for its side effects (memory writes, neuromod updates, calls
        to add_perturbation/add_attractor/set_baseline). Any returned vector
        is treated as a perturbation injected into bus.velocity directly,
        so slower-clock regions can still nudge the substrate without
        triggering a fresh integration step.
        """
        with self._lock:
            for region in regions:
                try:
                    p = region.step(self, neuromod, dt)
                except Exception as e:  # noqa: BLE001
                    if verbose_errors:
                        print(f"  [Region {region.name} error] {e}")
                    continue
                if p is None:
                    continue
                if not isinstance(p, torch.Tensor):
                    p = torch.as_tensor(p, device=self.device, dtype=self.state.dtype)
                if p.device != self.device:
                    p = p.to(self.device)
                if p.shape != self.state.shape:
                    if verbose_errors:
                        print(f"  [Region {region.name}] shape mismatch")
                    continue
                # Inject directly into velocity — flow clock will integrate it.
                self.velocity = self.velocity + p

    def add_perturbation(self, vec: torch.Tensor) -> None:
        """
        Inject a one-shot perturbation into velocity. Used by code paths
        outside the clock loop (e.g. user message arriving in Flask).
        """
        v = vec.detach().clone().to(self.device)
        if v.shape != self.state.shape:
            return
        with self._lock:
            self.velocity = self.velocity + v

    def _collect_perturbations(
        self,
        regions: Iterable,
        neuromod: object,
        dt: float,
        verbose_errors: bool,
    ) -> torch.Tensor:
        """Sum perturbations from a list of regions. Caller holds the lock."""
        out = torch.zeros(self.hidden_dim, device=self.device)
        for region in regions:
            try:
                p = region.step(self, neuromod, dt)
            except Exception as e:  # noqa: BLE001 — never crash substrate
                if verbose_errors:
                    print(f"  [Region {region.name} error] {e}")
                continue
            if p is None:
                continue
            if not isinstance(p, torch.Tensor):
                p = torch.as_tensor(p, device=self.device, dtype=self.state.dtype)
            if p.device != self.device:
                p = p.to(self.device)
            if p.shape != self.state.shape:
                if verbose_errors:
                    print(f"  [Region {region.name}] shape mismatch")
                continue
            out = out + p
        return out

    def _attractor_pull(self) -> torch.Tensor:
        if not self.attractors:
            return torch.zeros_like(self.state)
        pull = torch.zeros_like(self.state)
        for vec, weight in self.attractors:
            pull = pull + weight * (vec - self.state)
        pull = pull / len(self.attractors)
        return pull * self.config.attractor_strength

    # ------------------------------------------------------------------
    # Attractor management
    # ------------------------------------------------------------------
    def add_attractor(self, vec: torch.Tensor, weight: float = 1.0) -> None:
        """Append a new attractor. Drops the oldest non-baseline if over cap."""
        v = vec.detach().clone().to(self.device)
        if v.shape != self.state.shape:
            return
        with self._lock:
            self.attractors.append((v, float(weight)))
            while len(self.attractors) > self.config.max_attractors:
                # Index 0 is baseline — never drop it. Drop the next-oldest.
                self.attractors.pop(1)

    def set_baseline(self, vec: torch.Tensor, weight: float = 1.0) -> None:
        """Replace the calm baseline (attractor index 0)."""
        v = vec.detach().clone().to(self.device)
        if v.shape != self.state.shape:
            return
        with self._lock:
            if not self.attractors:
                self.attractors = [(v, float(weight))]
            else:
                self.attractors[0] = (v, float(weight))

    # ------------------------------------------------------------------
    # Read helpers (used by Executive, steering hook driver, dashboards)
    # ------------------------------------------------------------------
    def divergence_from_baseline(self) -> float:
        """1 - cosine_similarity(state, baseline). 0 = aligned, 2 = opposite."""
        if not self.attractors:
            return 0.0
        baseline = self.attractors[0][0]
        if baseline.norm() < 1e-8 or self.state.norm() < 1e-8:
            return float(self.state.norm())
        cs = F.cosine_similarity(
            self.state.unsqueeze(0).float(),
            baseline.unsqueeze(0).float(),
        ).item()
        return 1.0 - cs

    def snapshot(self) -> dict:
        return {
            "state_norm": float(self.state.norm()),
            "velocity_norm": float(self.velocity.norm()),
            "temperature": float(self.temperature),
            "tick_count": int(self.tick_count),
            "wall_seconds": time.monotonic() - self.wall_time_started,
            "n_attractors": len(self.attractors),
            "divergence": self.divergence_from_baseline(),
        }
