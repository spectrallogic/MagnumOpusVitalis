"""
DefaultMode region — idle drift via silent forward passes.

Port of v1's `_apply_idle_drift` (engine.py:731-795) wrapped as a Region.
Runs on the expensive clock (~1.5s).

Mechanism:
  1. With the steering hook active and bus.state injected via the driver,
     run a silent one-token forward pass through the model (no decoding).
  2. Capture the layer's hidden state.
  3. Compute delta = state - bus.state  (the model's response under steering).
  4. Project delta onto each emotion vector — these projections are
     "drift signals": directions the model itself gravitates toward under
     its current soul.
  5. Stimulate Limbic with those projections (amplitude × idle_drift_amp).

The result: even when the user is silent, the model's own latent geometry
nudges the emotional state. Combined with SubconsciousStack and continuous
flow, this is what makes idle feel alive — the system genuinely "thinks"
without input.

This region uses the model and the hook. The engine should pass a
`model_lock` so DefaultMode does not race against user-driven generation.
"""

import random
import threading
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region
from magnum_opus_v2.steering_hook import SteeringHook, BusSteeringDriver
from magnum_opus_v2.regions.limbic import Limbic


class DefaultMode(Region):
    name = "default_mode"
    clock = "expensive"

    def __init__(
        self,
        model,
        tokenizer,
        hook: SteeringHook,
        driver: BusSteeringDriver,
        limbic: Limbic,
        emotion_vectors: Dict[str, torch.Tensor],
        device: str = "cpu",
        idle_drift_amplitude: float = 0.04,
        log_event_threshold: float = 0.3,
        log_event_probability: float = 0.1,
        model_lock: Optional[threading.Lock] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook = hook
        self.driver = driver
        self.limbic = limbic
        self.device = device
        self.idle_drift_amplitude = float(idle_drift_amplitude)
        self.log_threshold = float(log_event_threshold)
        self.log_probability = float(log_event_probability)
        self.model_lock = model_lock or threading.Lock()

        # Cache emotion vectors on CPU float for projection (cheap dot products)
        self._emo_vecs_cpu: Dict[str, torch.Tensor] = {
            n: v.detach().cpu().float()
            for n, v in emotion_vectors.items()
            if not n.startswith("temporal_")
        }

        # Diagnostics
        self.last_projections: Dict[str, float] = {}
        self.events: list = []  # rolling list of strong-drift events

    def snapshot(self) -> dict:
        return {
            "last_projections": dict(self.last_projections),
            "n_events": len(self.events),
            "last_event": self.events[-1] if self.events else None,
        }

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        amp = self.idle_drift_amplitude
        if amp <= 0:
            return None
        bos = (
            self.tokenizer.bos_token_id
            or self.tokenizer.eos_token_id
            or 0
        )
        seed = torch.tensor([[bos]], device=self.device)

        with self.model_lock:
            self.hook.clear()
            # Push the live bus state into the hook so the silent pass is
            # steered by *current* state, not a stale snapshot.
            self.hook.set_steering(self.driver.read())
            try:
                with torch.no_grad():
                    _ = self.model(seed)
            except Exception:  # noqa: BLE001 — never crash the substrate
                self.hook.set_steering(None)
                self.last_projections = {}
                return None
            finally:
                # Always release the hook so generation calls aren't surprised.
                self.hook.set_steering(None)

            if not self.hook.captured_states:
                self.last_projections = {}
                return None

            state = self.hook.captured_states[-1].mean(dim=(0, 1)).detach().cpu().float()

        prev = bus.state.detach().cpu().float()
        delta = state - prev if prev.norm() > 1e-6 else state
        if delta.norm() < 1e-6:
            self.last_projections = {}
            return None
        delta = delta / delta.norm()

        projections: Dict[str, float] = {}
        for emo, v in self._emo_vecs_cpu.items():
            if v.norm() < 1e-6:
                continue
            proj = float(F.cosine_similarity(
                delta.unsqueeze(0), v.unsqueeze(0)
            ).item())
            projections[emo] = proj
            self.limbic.stimulate(emo, proj * amp)

        self.last_projections = {k: round(v, 4) for k, v in projections.items()}

        if projections:
            peak_emo, peak_val = max(projections.items(), key=lambda kv: abs(kv[1]))
            if abs(peak_val) > self.log_threshold and random.random() < self.log_probability:
                self.events.append({
                    "type": "drift",
                    "emotion": peak_emo,
                    "projection": round(peak_val, 4),
                })
                # Cap rolling log
                if len(self.events) > 100:
                    self.events = self.events[-100:]
        # No perturbation returned — we modify Limbic, which writes back next flow tick.
        return None
