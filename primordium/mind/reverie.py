"""
Reverie — rollouts of moments that haven't happened.

The organism-native replacement for the engine's DefaultMode and
SpeculativeFutures (those need an LLM; this needs only its own core).
Awake and idle it daydreams; asleep it dreams from replayed seeds. The
imagined trajectory gently perturbs the bus — wandering minds drift —
and decoded dream frames go to the dashboard. Rollouts are never trained
on: it does not teach itself its own hallucinations.
"""

import threading
import time
from typing import Optional

import torch


class Reverie:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_dream_at = 0.0
        self.last_delta: Optional[torch.Tensor] = None
        self.dreams_total = 0
        self.last_frames: list = []      # PNG bytes for the UI strip
        self._lock = threading.Lock()

    def due(self, asleep: bool, novelty: float, presence: float) -> bool:
        now = time.monotonic()
        if now - self.last_dream_at < self.cfg.dream_every_s:
            return False
        if asleep:
            return True
        return novelty < 0.15 and presence < 0.2      # idle daydream

    def record(self, pooled_delta: torch.Tensor, frames_png: list) -> None:
        with self._lock:
            self.last_delta = pooled_delta.detach().cpu()
            self.last_frames = frames_png[-4:]
            self.last_dream_at = time.monotonic()
            self.dreams_total += 1

    def take_delta(self) -> Optional[torch.Tensor]:
        with self._lock:
            d, self.last_delta = self.last_delta, None
            return d

    def frames(self) -> list:
        with self._lock:
            return list(self.last_frames)

    def snapshot(self) -> dict:
        with self._lock:
            return {"dreams_total": self.dreams_total,
                    "last_dream_age_s": (
                        round(time.monotonic() - self.last_dream_at, 1)
                        if self.last_dream_at else None)}
