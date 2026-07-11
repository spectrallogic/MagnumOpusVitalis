"""
LightPort — the eye's mailbox.

The browser posts JPEG frames (~6 fps, 96x96 RGB). The organism only ever
wants the newest one; older frames are simply the past and are discarded.
The worker downsamples to the current developmental acuity — a newborn
genuinely cannot see the detail that is present in the signal.
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class LightPort:
    def __init__(self, source_res: int = 96):
        self.source_res = int(source_res)
        self._jpeg: Optional[bytes] = None
        self._decoded: Optional[np.ndarray] = None   # (96, 96, 3) uint8 RGB
        self._ts: float = 0.0
        self._frames_total = 0
        self._lock = threading.Lock()

    def offer(self, jpeg_bytes: bytes, ts: Optional[float] = None) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes
            self._decoded = None          # decode lazily, once, on demand
            self._ts = ts if ts is not None else time.monotonic()
            self._frames_total += 1

    def offer_array(self, rgb: np.ndarray) -> None:
        """Direct injection path for SyntheticWorld / tests."""
        with self._lock:
            self._jpeg = None
            self._decoded = rgb.astype(np.uint8)
            self._ts = time.monotonic()
            self._frames_total += 1

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Newest frame as (96,96,3) uint8 RGB, plus its timestamp."""
        with self._lock:
            if self._decoded is None and self._jpeg is not None:
                arr = np.frombuffer(self._jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    if bgr.shape[:2] != (self.source_res, self.source_res):
                        bgr = cv2.resize(bgr, (self.source_res, self.source_res),
                                         interpolation=cv2.INTER_AREA)
                    self._decoded = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return self._decoded, self._ts

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def stage_view(self, res: int, channels: int,
                   gain: float = 1.0,
                   crop: Optional[tuple] = None) -> Optional[np.ndarray]:
        """The world at the organism's CURRENT acuity: (res, res, channels)
        uint8. `gain` < 1 = eyelids (sleep dims the world, never cuts it).
        `crop` = (cx, cy, zoom) from the Gaze: a movable window onto the
        source frame; zoom 1.0 is the whole scene, untouched."""
        frame, _ = self.latest()
        if frame is None:
            return None
        if crop is not None and crop[2] < 0.999:
            cx, cy, zoom = crop
            H = frame.shape[0]
            side = max(8, int(round(H * float(np.clip(zoom, 0.1, 1.0)))))
            half = side // 2
            px = int(round((cx + 1) / 2 * H))
            py = int(round((cy + 1) / 2 * H))
            px = min(max(px, half), H - (side - half))
            py = min(max(py, half), H - (side - half))
            frame = frame[py - half: py - half + side,
                          px - half: px - half + side]
        small = cv2.resize(frame, (res, res), interpolation=cv2.INTER_AREA)
        if channels == 1:
            small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)[..., None]
        if gain < 0.999:
            small = (small.astype(np.float32) * gain).astype(np.uint8)
        return small

    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_total": self._frames_total,
                "age_s": (time.monotonic() - self._ts) if self._ts else None,
            }
