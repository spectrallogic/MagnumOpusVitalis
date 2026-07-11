"""
JepaObjective — learn by expecting the next moment.

The hidden state at sensory position i of tick t predicts the EMA-target
latent of sensory position i of tick t+1 (group-shifted, latent-space:
no pixel regression in the representation path), across all four sensory
modalities: audio, vision, text, canvas. VICReg-lite variance/covariance
terms keep the latents from collapsing — a real failure mode at this
scale, watched live on the dashboard.

Anti-reward-hacking rule: the CANVAS term trains but is EXCLUDED from
the scalar fed to observe() (surprise / learning-progress / competence).
Its own paintings are trivially predictable; a mind must not be able to
satisfy its competence drive by staring at its own artwork.
"""

import copy
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class JepaObjective:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.device = None
        self._rebuild_targets(tokenizer)
        self.target_ln = nn.LayerNorm(cfg.d_model)

        # surprise bookkeeping (wall-time EMAs)
        self.ema_fast = None
        self.ema_slow = None
        self._last_t = time.monotonic()
        self.last_latent_std = 0.0

    def to(self, device):
        self.device = device
        for m in (self.t_vision, self.t_audio, self.t_text, self.t_canvas,
                  self.target_ln):
            m.to(device)
        return self

    def _rebuild_targets(self, tokenizer) -> None:
        """EMA target encoders = frozen copies of the online embedders."""
        self.t_vision = copy.deepcopy(tokenizer.vision_embed)
        self.t_audio = copy.deepcopy(tokenizer.audio_embed)
        self.t_text = copy.deepcopy(tokenizer.text_embed)
        self.t_canvas = copy.deepcopy(tokenizer.canvas_embed)
        for m in (self.t_vision, self.t_audio, self.t_text, self.t_canvas):
            for p in m.parameters():
                p.requires_grad_(False)

    def on_stage_grown(self, tokenizer) -> None:
        self._rebuild_targets(tokenizer)
        if self.device is not None:
            for m in (self.t_vision, self.t_audio, self.t_text, self.t_canvas):
                m.to(self.device)

    @torch.no_grad()
    def ema_update(self, tokenizer) -> None:
        m = self.cfg.ema_target_momentum
        pairs = [
            (self.t_vision, tokenizer.vision_embed),
            (self.t_audio, tokenizer.audio_embed),
            (self.t_text, tokenizer.text_embed),
            (self.t_canvas, tokenizer.canvas_embed),
        ]
        for tgt, online in pairs:
            for tp, op in zip(tgt.parameters(), online.parameters()):
                tp.mul_(m).add_(op.detach(), alpha=1 - m)

    @torch.no_grad()
    def targets(self, vis_raw: torch.Tensor, aud_raw: torch.Tensor,
                txt_ids: torch.Tensor, cnv_raw: torch.Tensor) -> torch.Tensor:
        """EMA-encoded latents of a tick's raw sensors:
        (n_a + n_v + n_t + n_c, d), in sensory order aud|vis|txt|cnv."""
        za = self.target_ln(self.t_audio(aud_raw))
        zv = self.target_ln(self.t_vision(vis_raw))
        zt = self.target_ln(self.t_text(txt_ids))
        zc = self.target_ln(self.t_canvas(cnv_raw))
        return torch.cat([za, zv, zt, zc], dim=0)

    def loss(
        self,
        pred: torch.Tensor,        # (n_sens, d) predictor outputs (tick t)
        target: torch.Tensor,      # (n_sens, d) EMA targets of tick t+1
        online_latents: torch.Tensor,  # (n_sens, d) current online hiddens
        n_vis: int,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        n_a = cfg.audio_tokens
        n_t = cfg.text_tokens
        a0, v0 = 0, n_a
        t0 = v0 + n_vis
        c0 = t0 + n_t

        p = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        per_tok = F.smooth_l1_loss(p, t, reduction="none").mean(-1)
        aud_loss = per_tok[a0:v0].mean()
        vis_loss = per_tok[v0:t0].mean()
        txt_loss = per_tok[t0:c0].mean()
        cnv_loss = per_tok[c0:].mean()

        # the WORLD term drives surprise/competence; canvas is excluded
        world_loss = (vis_loss
                      + cfg.audio_loss_weight * aud_loss
                      + cfg.txt_loss_weight * txt_loss)
        pred_loss = world_loss + cfg.cnv_loss_weight * cnv_loss

        # anti-collapse (VICReg-lite) on the online latents
        z = online_latents
        std = z.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        cov_loss = (off ** 2).sum() / z.shape[1]

        total = (pred_loss
                 + cfg.var_loss_weight * var_loss
                 + cfg.cov_loss_weight * cov_loss)
        self.last_latent_std = float(std.mean())
        return total, {
            "pred": float(world_loss),          # canvas-free, for observe()
            "aud": float(aud_loss), "vis": float(vis_loss),
            "txt": float(txt_loss), "cnv": float(cnv_loss),
            "var": float(var_loss), "cov": float(cov_loss),
            "latent_std": self.last_latent_std,
        }

    def observe(self, loss_value: float) -> Tuple[float, float]:
        """Update wall-time EMAs; return (surprise, learning_progress)."""
        now = time.monotonic()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now
        af = 1 - pow(2.718281828, -dt / self.cfg.surprise_fast_tau_s)
        asl = 1 - pow(2.718281828, -dt / self.cfg.surprise_slow_tau_s)
        self.ema_fast = (loss_value if self.ema_fast is None
                         else (1 - af) * self.ema_fast + af * loss_value)
        self.ema_slow = (loss_value if self.ema_slow is None
                         else (1 - asl) * self.ema_slow + asl * loss_value)
        surprise = loss_value / (self.ema_slow + 1e-8)
        lp = self.ema_fast - self.ema_slow      # negative = improving
        return float(surprise), float(lp)
