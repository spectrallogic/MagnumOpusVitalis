"""
SoundPort — the ear's short echo, and its mel frontend.

The browser posts 200 ms Int16 PCM chunks at 16 kHz (echo cancellation OFF
— the organism must be able to hear its own voice). The ring holds the
last ~5 s; each tick the worker takes the newest tick's worth and renders
a log-mel patch on the GPU.
"""

import threading
from typing import Optional

import numpy as np
import torch


class SoundPort:
    def __init__(self, sample_rate: int = 16000, seconds: float = 5.0):
        self.sr = int(sample_rate)
        self.n = int(self.sr * seconds)
        self.buf = np.zeros(self.n, dtype=np.int16)
        self.write = 0
        self.total = 0
        self.underruns = 0
        self._lock = threading.Lock()

    def push(self, pcm_int16: np.ndarray) -> None:
        x = np.asarray(pcm_int16, dtype=np.int16).ravel()
        with self._lock:
            m = len(x)
            if m >= self.n:
                self.buf[:] = x[-self.n:]
                self.write = 0
            else:
                end = self.write + m
                if end <= self.n:
                    self.buf[self.write:end] = x
                else:
                    k = self.n - self.write
                    self.buf[self.write:] = x[:k]
                    self.buf[: end - self.n] = x[k:]
                self.write = end % self.n
            self.total += m

    def take_latest(self, n_samples: int) -> np.ndarray:
        """Newest n_samples as float32 in [-1, 1]; zero-padded on underrun."""
        with self._lock:
            n = min(n_samples, self.n)
            start = (self.write - n) % self.n
            if start + n <= self.n:
                out = self.buf[start:start + n].copy()
            else:
                k = self.n - start
                out = np.concatenate([self.buf[start:], self.buf[: n - k]])
            if self.total < n_samples:
                self.underruns += 1
                pad = np.zeros(n_samples - len(out), dtype=np.int16)
                out = np.concatenate([pad, out])
        return out.astype(np.float32) / 32768.0

    def rms(self, n_samples: int = 3200) -> float:
        x = self.take_latest(n_samples)
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def stats(self) -> dict:
        with self._lock:
            return {"samples_total": self.total, "underruns": self.underruns}


class MelFrontend:
    """torchaudio log-mel on the worker's device."""

    def __init__(self, cfg, device: str = "cuda"):
        import torchaudio
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop, n_mels=cfg.n_mels, power=2.0,
        ).to(device)
        self.n_frames = cfg.audio_tokens * cfg.mel_frames_per_token
        self.n_samples = cfg.hop * self.n_frames + cfg.n_fft  # a little margin

    def render(self, pcm_f32: "np.ndarray") -> torch.Tensor:
        """(n_mels, n_frames) log-mel, roughly unit-scale."""
        x = torch.from_numpy(pcm_f32).to(self.device)
        with torch.no_grad():
            m = self.mel(x)                      # (n_mels, T)
            m = torch.log1p(m)
        if m.shape[1] < self.n_frames:
            m = torch.nn.functional.pad(m, (self.n_frames - m.shape[1], 0))
        return (m[:, -self.n_frames:] / 4.0)
