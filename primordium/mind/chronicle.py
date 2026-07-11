"""
Chronicle — raw lived moments, replayed by what they meant.

Stores RAW sensors (source-resolution JPEG + mel + motor + intero), not
latents: replay re-encodes with the CURRENT encoders at the CURRENT
developmental acuity, so after a growth spurt the organism literally
re-sees its old life with new eyes. Prioritized by surprise and reward —
what moved you is what returns.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Episode:
    tick_id: int
    jpeg: bytes                 # source-res RGB, JPEG-encoded
    mel: np.ndarray             # (n_mels, frames) float16
    motor: np.ndarray           # (PROPRIO_DIM,) float16 (full efference)
    intero: np.ndarray          # (INTERO_DIM,) float16
    priority: float
    txt_ids: Optional[np.ndarray] = None   # (text_tokens,) int16
    canvas_png: bytes = b""                # its Easel at that moment


class Chronicle:
    def __init__(self, capacity: int = 20000, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buf: List[Episode] = []
        self._next = 0
        self._lock = threading.Lock()
        self.replayed_total = 0

    def add(self, ep: Episode) -> None:
        with self._lock:
            if len(self.buf) < self.capacity:
                self.buf.append(ep)
            else:
                self.buf[self._next] = ep
                self._next = (self._next + 1) % self.capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self.buf)

    def sample_windows(self, batch: int, window: int,
                       rng: Optional[np.random.Generator] = None
                       ) -> List[List[Episode]]:
        """Prioritized windows of consecutive ticks (verified contiguous)."""
        rng = rng or np.random.default_rng()
        with self._lock:
            n = len(self.buf)
            if n < window + 2:
                return []
            by_tick = sorted(self.buf, key=lambda e: e.tick_id)
            pri = np.array([e.priority for e in by_tick], dtype=np.float64)
            pri = np.maximum(pri, 1e-3) ** self.alpha
            windows: List[List[Episode]] = []
            tries = 0
            while len(windows) < batch and tries < batch * 8:
                tries += 1
                end = int(rng.choice(n, p=pri / pri.sum()))
                if end < window:
                    continue
                win = by_tick[end - window + 1: end + 1]
                ids = [e.tick_id for e in win]
                if ids[-1] - ids[0] == window - 1:      # contiguous life
                    windows.append(win)
            self.replayed_total += len(windows)
            return windows

    @staticmethod
    def decode_frame(ep: Episode) -> Optional[np.ndarray]:
        arr = np.frombuffer(ep.jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def tail(self, n: int) -> List[Episode]:
        with self._lock:
            return sorted(self.buf, key=lambda e: e.tick_id)[-n:]

    def snapshot(self) -> dict:
        with self._lock:
            return {"size": len(self.buf), "replayed_total": self.replayed_total}
