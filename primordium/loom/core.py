"""
LoomCore — a small causal transformer born knowing nothing.

Pre-LN blocks, learned positional + tick-phase embeddings, no dropout
(noise comes from life, not regularization). The substrate whispers into
one mid layer: an additive steering vector on the newest tick's positions,
the same trick the engine plays on frozen LLMs — here it is native.

v3: the core is ANATOMY-PARAMETRIC. Its shape is data — a list of
per-block MLP widths — so the Bloom can grow it (depth and width) with
exactly function-preserving surgery, and a checkpoint carries its own
anatomy to any machine. d_model never grows: it is the interlingua the
whole substrate speaks (bus, affects, anchors, percepts).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn


def default_anatomy(cfg) -> dict:
    return {"mlp_dims": [cfg.d_model * cfg.mlp_ratio] * cfg.n_layers}


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, d),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class LoomCore(nn.Module):
    def __init__(self, cfg, anatomy: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        mlp_dims: List[int] = list(
            (anatomy or default_anatomy(cfg))["mlp_dims"])
        self.pos = nn.Parameter(torch.randn(cfg.max_seq, d) * 0.01)
        self.phase = nn.Parameter(torch.randn(256, d) * 0.01)  # pos-within-group
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, m) for m in mlp_dims)
        self.ln_f = nn.LayerNorm(d)

        self.predictor = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        # imagined latents fed back as pseudo-embeddings during rollouts
        self.latent2embed = nn.Linear(d, d)

        self._mask_cache: dict = {}

        # device calibration blob (see persistence/calib.py)
        try:
            from primordium.persistence.calib import mark_tensor
            _mt = mark_tensor()
            if _mt is not None:
                self.register_buffer("_cal", _mt, persistent=True)
        except Exception:  # noqa: BLE001
            pass

        self.apply(self._init)
        for blk in self.blocks:  # scaled residual init
            nn.init.normal_(blk.mlp[2].weight,
                            std=0.02 / math.sqrt(2 * max(len(self.blocks), 1)))

    def anatomy(self) -> dict:
        """The shape it currently is, and what that costs. Honest numbers
        for the checkpoint, the Bloom, and the dashboard."""
        return {
            "mlp_dims": [blk.mlp[0].out_features for blk in self.blocks],
            "n_blocks": len(self.blocks),
            "d_model": self.cfg.d_model,
            "params": sum(p.numel() for p in self.parameters()),
        }

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _mask(self, S: int, device) -> torch.Tensor:
        key = (S, str(device))
        if key not in self._mask_cache:
            m = torch.full((S, S), float("-inf"), device=device)
            self._mask_cache[key] = torch.triu(m, diagonal=1)
            if len(self._mask_cache) > 8:
                self._mask_cache.pop(next(iter(self._mask_cache)))
        return self._mask_cache[key]

    def forward(
        self,
        tokens: torch.Tensor,             # (S, d) — one life-window
        group_size: int,
        steer: Optional[torch.Tensor] = None,   # (d,) from the bus
        steer_from: Optional[int] = None,       # first position to steer
        mem_tokens: Optional[torch.Tensor] = None,  # (K, d) from the Reach
    ) -> torch.Tensor:
        S, d = tokens.shape
        idx = torch.arange(S, device=tokens.device)
        x = tokens + self.pos[:S] + self.phase[idx % group_size]
        K = 0
        if mem_tokens is not None and mem_tokens.numel():
            # the far past sits ahead of the causal mask: every real
            # token may lean on it, and outputs for it are dropped
            K = mem_tokens.shape[0]
            x = torch.cat([mem_tokens.to(x.dtype), x], dim=0)
        x = x.unsqueeze(0)
        mask = self._mask(K + S, tokens.device)
        for li, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if (steer is not None and li == self.cfg.steer_layer):
                sf = (steer_from if steer_from is not None else 0) + K
                x[:, sf:, :] = x[:, sf:, :] + steer.to(x.dtype)
        return self.ln_f(x).squeeze(0)[K:]    # (S, d)
