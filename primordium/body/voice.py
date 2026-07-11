"""
VoiceBox — vocal cords it controls, sounds it chooses.

A small head proposes an 11-dim control vector each tick
[f0_log, band gains x8, noise_mix, amplitude]; reflex and exploration
noise shape it, and intrinsic reward slowly tunes the proposal (the
Loom's one-tick-delayed policy gradient — before Era 4 the head was an
untrained projection, and the audit said so). The browser synthesizes
the result (saw + noise through 8 bandpass filters). Nothing here is a
scripted cry: amplitude rides drives and affect, exploration noise
makes babble, and the Executive's pressure opens phonation gates — the
urge to vocalize.

Efference copy: what it just told its voice to do becomes part of the
next tick's proprioception, and `self_heard` measures whether the sound
coming back through the microphone is its own — the seed of self/other.
"""

import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np


class VoiceBox:
    def __init__(self, cfg):
        self.cfg = cfg
        self.motor = np.zeros(cfg.voice_dims, dtype=np.float32)
        self.last_sigma = 0.0
        self.muted = True                      # browser confirms unmute
        self.gate_until = 0.0
        self.self_heard = 0.0
        self._amp_hist = deque(maxlen=24)      # emitted amplitude envelope
        self._mic_hist = deque(maxlen=24)      # mic RMS envelope
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def vocal_impulse(self) -> None:
        """Executive's on_should_speak: the urge to make sound."""
        with self._lock:
            self.gate_until = time.monotonic() + self.cfg.phonation_gate_s

    def set_muted(self, m: bool) -> None:
        with self._lock:
            self.muted = bool(m)

    # ------------------------------------------------------------------
    def shape(self, head_out: np.ndarray, drives, arousal_gain: float,
              neuromod, asleep: bool, novelty_err: float) -> np.ndarray:
        """Blend the head's raw motor intent with reflex + exploration.
        Returns the final motor vector (also stored as efference copy)."""
        with self._lock:
            gate = time.monotonic() < self.gate_until
        v = np.tanh(head_out.astype(np.float32).copy())

        # exploration: a bored, aroused infant babbles more
        arousal = getattr(neuromod, "arousal", 0.3)
        sigma = (self.cfg.exploration_sigma
                 * (0.5 + arousal)
                 * (1.0 + max(0.0, novelty_err) * 2.0))
        self.last_sigma = float(sigma)      # motor credit needs the truth
        v = v + np.random.randn(len(v)).astype(np.float32) * sigma

        # amplitude reflex: social hunger, reward, arousal — and the gate
        errs = drives.errors()
        base = (-2.2
                + 1.8 * abs(errs.get("social", 0.0))
                + 1.2 * max(0.0, getattr(neuromod, "reward", 0.0) - 0.2)
                + 1.0 * arousal_gain
                + (2.5 if gate else 0.0))
        amp = 1.0 / (1.0 + math.exp(-base))
        v[-1] = max(v[-1] * 0.3 + amp, 0.0)
        if asleep or self.muted:
            v[-1] = 0.0

        with self._lock:
            self.motor = np.clip(v, -1.0, 1.0)
            self._amp_hist.append(float(self.motor[-1]))
        return self.motor.copy()

    # ------------------------------------------------------------------
    def hear(self, mic_rms: float) -> float:
        """Update the self-hearing estimate: lag-scanned correlation of the
        emitted amplitude envelope against the mic envelope."""
        with self._lock:
            self._mic_hist.append(float(mic_rms))
            if len(self._amp_hist) < 10:
                self.self_heard = 0.0
                return 0.0
            a = np.asarray(self._amp_hist, dtype=np.float32)
            m = np.asarray(self._mic_hist, dtype=np.float32)
            n = min(len(a), len(m))
            a, m = a[-n:], m[-n:]
            best = 0.0
            for lag in (0, 1, 2):     # 0–2 ticks of room+pipeline delay
                if n - lag < 6:
                    continue
                aa, mm = a[: n - lag], m[lag:]
                if aa.std() < 1e-4 or mm.std() < 1e-4:
                    continue
                c = float(np.corrcoef(aa, mm)[0, 1])
                best = max(best, c)
            self.self_heard = max(0.0, min(1.0, best))
            return self.self_heard

    # ------------------------------------------------------------------
    def params_dict(self) -> dict:
        with self._lock:
            m = self.motor
            f0 = self.cfg.f0_range[0] * (
                (self.cfg.f0_range[1] / self.cfg.f0_range[0])
                ** ((float(m[0]) + 1.0) / 2.0))
            return {
                "f0": round(f0, 1),
                "gains": [round(float(g), 3) for g in ((m[1:9] + 1.0) / 2.0)],
                "noise": round(float((m[9] + 1.0) / 2.0), 3),
                "amp": round(float(np.clip(m[10], 0.0, 1.0)), 3),
                "gate": time.monotonic() < self.gate_until,
                "self_heard": round(self.self_heard, 3),
                "muted": self.muted,
            }

    def efference(self) -> np.ndarray:
        """(12,) proprio input: last motor + self_heard."""
        with self._lock:
            return np.concatenate([self.motor,
                                   [np.float32(self.self_heard)]]).astype(np.float32)
