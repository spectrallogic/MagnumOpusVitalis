"""
SyntheticWorld — a womb for tests.

Feeds the same LightPort/SoundPort interfaces the browser would: a
bouncing dot whose horizontal position controls the pitch of a sine tone
(so vision genuinely predicts audio — cross-modal structure to learn),
with occasional scene flips for surprise. Lets every test run without a
camera, a microphone, or a human.
"""

import numpy as np


class SyntheticWorld:
    def __init__(self, retina, cochlea, res: int = 96, sr: int = 16000,
                 seed: int = 7):
        self.retina = retina
        self.cochlea = cochlea
        self.res = res
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.x, self.y = 0.3, 0.5
        self.vx, self.vy = 0.021, 0.013
        self.bg = 30
        self.phase = 0.0
        self.t = 0

    def step(self, dt_s: float = 0.15) -> None:
        self.t += 1
        # occasional scene flip = genuine novelty
        if self.rng.random() < 0.002:
            self.bg = int(self.rng.integers(10, 90))
            self.vx, self.vy = -self.vx, self.vy

        self.x += self.vx
        self.y += self.vy
        if not 0.05 < self.x < 0.95:
            self.vx = -self.vx
        if not 0.05 < self.y < 0.95:
            self.vy = -self.vy

        frame = np.full((self.res, self.res, 3), self.bg, dtype=np.uint8)
        cx, cy = int(self.x * self.res), int(self.y * self.res)
        rr = 7
        yy, xx = np.ogrid[:self.res, :self.res]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
        frame[mask] = (220, 200, 90)
        self.retina.offer_array(frame)

        # tone frequency rides the dot's x — sight predicts sound
        f = 200.0 + 500.0 * self.x
        n = int(self.sr * dt_s)
        tt = np.arange(n) / self.sr
        pcm = 0.2 * np.sin(2 * np.pi * f * tt + self.phase)
        self.phase = float((self.phase + 2 * np.pi * f * dt_s) % (2 * np.pi))
        self.cochlea.push((pcm * 32767).astype(np.int16))
