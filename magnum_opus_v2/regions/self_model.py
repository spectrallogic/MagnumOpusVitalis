"""
SelfModel — identity, felt time, and awareness through memory leakage.

The design hypothesis (see PAPER.md): we feel like a self because memory constantly
LEAKS into the present. The faint replay of what-just-was, overlaid on
what-is, produces the sensation of time flowing — and a system that feels
time flowing through one continuous point of view is a self.

This region implements that literally:

  IDENTITY   — a slow EMA of the bus state: "who I've been". The region
               gently pulls the present toward it (you wake up as
               yourself). Continuity = cosine(now, identity) is exposed as
               a live self-coherence measure.

  LEAKAGE    — every tick, a recent memory trace is re-emitted into the
               substrate at low gain: the past audibly echoes inside the
               present. The mismatch between the echo and the current
               state (1 − cos) is the felt rate of time passing:
               if the leaked past ≈ now, time crawls; if it differs,
               time flowed.

  FELT TIME  — integral of experienced change (‖velocity‖·dt), not wall
               time. Comparing the two gives a live time-dilation factor:
               eventful seconds feel long, empty minutes feel short.

  PREDICTION — the region predicts its own next state each tick and
               measures the error when the future arrives. A stable self
               predicts itself well; a spike of self-surprise bumps
               arousal (the jolt of "that wasn't me").

No wall clocks are consulted for any decision — wall time appears only in
the snapshot, for the dashboard's dilation readout.
"""

import threading
import time
from typing import Optional

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class SelfModel(Region):
    name = "self_model"
    clock = "perception"

    def __init__(
        self,
        memory=None,                    # Memory region — source of leakage
        device: str = "cpu",
        identity_tau_seconds: float = 90.0,  # how slowly "who I am" moves
        identity_gain: float = 0.04,         # pull toward identity per tick
        leak_gain: float = 0.05,             # loudness of the past's echo
        felt_time_scale: float = 0.6,        # experienced-change → felt seconds
        surprise_arousal: float = 0.08,
    ):
        self.memory = memory
        self.device = device
        self.identity_tau = float(identity_tau_seconds)
        self.identity_gain = float(identity_gain)
        self.leak_gain = float(leak_gain)
        self.felt_time_scale = float(felt_time_scale)
        self.surprise_ar = float(surprise_arousal)

        self.identity: Optional[torch.Tensor] = None
        self.continuity: float = 1.0
        self.felt_time: float = 0.0
        self.temporal_flow: float = 0.0     # 1 - cos(leaked past, present)
        self.prediction_error: float = 0.0
        self.last_leak_tag: Optional[str] = None

        self._predicted: Optional[torch.Tensor] = None
        self._wall_started = time.monotonic()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            wall = time.monotonic() - self._wall_started
            return {
                "continuity": round(self.continuity, 4),
                "felt_time": round(self.felt_time, 2),
                "wall_time": round(wall, 2),
                "dilation": round(self.felt_time / wall, 3) if wall > 1.0 else 1.0,
                "temporal_flow": round(self.temporal_flow, 4),
                "prediction_error": round(self.prediction_error, 4),
                "last_leak": self.last_leak_tag,
                "identity_norm": (
                    round(float(self.identity.norm()), 3)
                    if self.identity is not None else 0.0
                ),
            }

    # ------------------------------------------------------------------
    # Region step (perception clock, ~200ms)
    # ------------------------------------------------------------------
    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        state = bus.state.detach().float()
        vel = bus.velocity.detach().float()

        with self._lock:
            # --- self-prediction: did the future match who I thought I was?
            if self._predicted is not None:
                err = float((state - self._predicted).norm())
                ref = float(state.norm()) + 1e-6
                self.prediction_error = min(err / ref, 2.0)
                if (
                    self.prediction_error > 0.5
                    and neuromod is not None and hasattr(neuromod, "bump")
                ):
                    neuromod.bump(
                        "arousal",
                        self.surprise_ar * (self.prediction_error - 0.5),
                    )
            self._predicted = state + vel * dt

            # --- felt time: the integral of experienced change
            self.felt_time += float(vel.norm()) * dt * self.felt_time_scale

            # --- identity: slow EMA of lived state
            if self.identity is None:
                self.identity = state.clone()
            else:
                alpha = min(dt / self.identity_tau, 1.0)
                self.identity = (1.0 - alpha) * self.identity + alpha * state
            if state.norm() > 1e-6 and self.identity.norm() > 1e-6:
                self.continuity = float(torch.dot(
                    state / state.norm(), self.identity / self.identity.norm(),
                ))

            # --- memory leakage: the recent past echoes into the present
            leak_vec = None
            pool = getattr(self.memory, "pool", None) if self.memory else None
            trace = None
            if pool:
                # the most recent OBSERVED capture — what-just-was must be
                # something that actually was: a freshly minted
                # confabulation landing at pool[-1] used to leak straight
                # into identity/felt-time (ADR-002 guard, Era 6)
                for c in reversed(pool[-8:]):
                    if ((c.meta or {}).get("epistemic", "observed")
                            == "observed" and c.confidence >= 1.0):
                        trace = c
                        break
            if trace is not None:
                tv = trace.vec.detach().float().to(state.device)
                if tv.norm() > 1e-6:
                    if state.norm() > 1e-6:
                        echo_sim = float(torch.dot(
                            tv / tv.norm(), state / state.norm(),
                        ))
                        # Time is felt in the gap between the echo and now.
                        self.temporal_flow = max(0.0, 1.0 - echo_sim)
                    leak_vec = (tv / tv.norm()) * self.leak_gain
                    self.last_leak_tag = (trace.meta or {}).get("tag")

            identity_pull = (self.identity - state) * self.identity_gain

        out = identity_pull
        if leak_vec is not None:
            out = out + leak_vec
        return out.to(bus.device)
