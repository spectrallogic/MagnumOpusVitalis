"""
ReverieDecoder — a window into what it expects.

Turns predicted vision latents back into pixels for the dashboard only:
gradients stop at the latents, so pixel pressure never shapes the
representation. Trained casually (MSE vs the frame that actually arrives)
at ~1 Hz; it lets us watch its reverie and its dreams.
"""

import torch
import torch.nn as nn


class ReverieDecoder(nn.Module):
    def __init__(self, cfg, stage: int):
        super().__init__()
        self.cfg = cfg
        self.build(stage)

    def build(self, stage: int) -> None:
        s = self.cfg.stages[stage]
        self.stage = stage
        self.grid = s.res // s.patch
        d = self.cfg.d_model
        ch = s.channels
        # latent grid (d, g, g) -> pixels (ch, res, res)
        ups = []
        cur = 128
        ups.append(nn.Conv2d(d, cur, 1))
        size = self.grid
        while size < s.res:
            ups += [nn.GELU(), nn.ConvTranspose2d(cur, max(cur // 2, 16), 4, 2, 1)]
            cur = max(cur // 2, 16)
            size *= 2
        ups += [nn.GELU(), nn.Conv2d(cur, ch, 3, 1, 1), nn.Tanh()]
        self.net = nn.Sequential(*ups)

    def forward(self, vision_latents: torch.Tensor) -> torch.Tensor:
        """(n_vis, d) detached -> (ch, res, res) in [-1, 1]."""
        g = self.grid
        x = vision_latents.detach().T.reshape(1, -1, g, g)
        return self.net(x).squeeze(0)


class AudioDecoder(nn.Module):
    """Predicted audio latents -> a mel patch. The browser sonifies it:
    low-fi on purpose — this is what its expectation of sound looks like,
    not a recording. Stop-grad into the trunk, like the vision decoder."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        out = cfg.n_mels * cfg.mel_frames_per_token
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 256), nn.GELU(), nn.Linear(256, out))

    def forward(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """(audio_tokens, d) detached -> (n_mels, frames)."""
        k = self.cfg.mel_frames_per_token
        y = self.net(audio_latents.detach())            # (T, n_mels*k)
        y = y.reshape(-1, self.cfg.n_mels, k)           # (T, n_mels, k)
        return torch.cat([y[i] for i in range(y.shape[0])], dim=1)
