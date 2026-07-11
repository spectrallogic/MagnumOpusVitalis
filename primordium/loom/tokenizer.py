"""
SensoryTokenizer — the moment, rendered as tokens.

One tick becomes one fixed-order token group:

    [STATE x4] [INTERO x1] [PROPRIO x1] [AUD x3] [VIS x n_vis] [TXT x8] [CNV x4]

STATE tokens are last tick's detached recurrence; INTERO is the felt body
(drives, chemistry, sleep, stage, dormant organs); PROPRIO is the
efference copy of voice + keys + paint; AUD/VIS are the world; TXT is the
conversation stream (others' words and its own, a few bytes per tick);
CNV is its own canvas, seen as a body part. Vision acuity follows the
developmental stage — growth performs kernel surgery so earlier learning
survives. TXT/CNV have fixed acuity from birth and are never surgeried.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SensoryTokenizer(nn.Module):
    # 5 drive levels + 5 errors + 3 tide channels [arousal, reward, calm]
    # + sleep + 3 stage one-hot
    # + 2 gates x [present, open] — dormant limbs are honestly felt
    INTERO_DIM = 21
    # voice 11 + self_heard 1 + keys 3 [last_char/255, typing_gate, chars/2]
    # + paint efference 8 [gate,x,y,radius,r,g,b,alpha]
    # + gaze efference 3 [x, y, zoom] — it knows where it is looking
    PROPRIO_DIM = 26

    def __init__(self, cfg, stage: int = 0):
        super().__init__()
        self.cfg = cfg
        self.stage = int(stage)
        d = cfg.d_model

        spec = cfg.stages[self.stage]
        self.vision_embed = nn.Linear(spec.patch_dim, d)
        self.audio_embed = nn.Linear(cfg.n_mels * cfg.mel_frames_per_token, d)
        self.proprio_embed = nn.Sequential(
            nn.Linear(self.PROPRIO_DIM, 128), nn.GELU(), nn.Linear(128, d))
        self.intero_embed = nn.Sequential(
            nn.Linear(self.INTERO_DIM, 128), nn.GELU(), nn.Linear(128, d))
        self.state_proj = nn.Linear(d, d)
        self.text_embed = nn.Embedding(cfg.text_vocab, d)
        cnv_dim = cfg.canvas_patch * cfg.canvas_patch * 3
        self.canvas_embed = nn.Linear(cnv_dim, d)

        # modality embeddings: STATE/INTERO/PROPRIO/AUD/VIS/TXT/CNV
        self.modality = nn.Parameter(torch.randn(7, d) * 0.02)

    # ------------------------------------------------------------------
    # layout helpers
    # ------------------------------------------------------------------
    @property
    def spec(self):
        return self.cfg.stages[self.stage]

    @property
    def n_vis(self) -> int:
        return self.spec.n_tokens

    @property
    def n_cnv(self) -> int:
        return (self.cfg.canvas_res // self.cfg.canvas_patch) ** 2

    @property
    def group_size(self) -> int:
        return (self.cfg.state_tokens + 2 + self.cfg.audio_tokens
                + self.n_vis + self.cfg.text_tokens + self.n_cnv)

    @property
    def sensory_slice(self) -> slice:
        """Positions of AUD+VIS+TXT+CNV within a group. AUD stays first,
        so `[:audio_tokens]` slicing of predictions remains valid."""
        start = self.cfg.state_tokens + 2
        return slice(start, start + self.cfg.audio_tokens + self.n_vis
                     + self.cfg.text_tokens + self.n_cnv)

    # ------------------------------------------------------------------
    # raw feature extraction
    # ------------------------------------------------------------------
    def patchify(self, img_u8: np.ndarray) -> torch.Tensor:
        """(res,res,C) uint8 -> (n_vis, patch_dim) float in ~[-1,1]."""
        s = self.spec
        x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
        x = (x - 0.5) * 2.0
        x = x.reshape(s.res // s.patch, s.patch, s.res // s.patch, s.patch, s.channels)
        x = x.permute(0, 2, 1, 3, 4).reshape(s.n_tokens, s.patch_dim)
        return x

    def melify(self, mel: torch.Tensor) -> torch.Tensor:
        """(n_mels, frames) -> (audio_tokens, n_mels*frames_per_token)."""
        k = self.cfg.mel_frames_per_token
        toks = []
        for i in range(self.cfg.audio_tokens):
            toks.append(mel[:, i * k:(i + 1) * k].reshape(-1))
        return torch.stack(toks)

    def cnvify(self, fb_u8: np.ndarray) -> torch.Tensor:
        """(res,res,3) uint8 canvas -> (n_cnv, patch_dim) float ~[-1,1]."""
        p = self.cfg.canvas_patch
        r = self.cfg.canvas_res
        x = torch.from_numpy(fb_u8.astype(np.float32) / 255.0)
        x = (x - 0.5) * 2.0
        x = x.reshape(r // p, p, r // p, p, 3)
        x = x.permute(0, 2, 1, 3, 4).reshape(self.n_cnv, p * p * 3)
        return x

    # ------------------------------------------------------------------
    # tick encoding
    # ------------------------------------------------------------------
    def encode_tick(
        self,
        vis_raw: torch.Tensor,      # (n_vis, patch_dim) on device
        aud_raw: torch.Tensor,      # (audio_tokens, 160) on device
        intero: torch.Tensor,       # (INTERO_DIM,)
        proprio: torch.Tensor,      # (PROPRIO_DIM,)
        state_toks: torch.Tensor,   # (state_tokens, d) detached recurrence
        txt_ids: torch.Tensor,      # (text_tokens,) long
        cnv_raw: torch.Tensor,      # (n_cnv, canvas patch_dim)
    ) -> torch.Tensor:
        """-> (group_size, d) token group for one tick."""
        st = self.state_proj(state_toks) + self.modality[0]
        it = self.intero_embed(intero).unsqueeze(0) + self.modality[1]
        pr = self.proprio_embed(proprio).unsqueeze(0) + self.modality[2]
        au = self.audio_embed(aud_raw) + self.modality[3]
        vi = self.vision_embed(vis_raw) + self.modality[4]
        tx = self.text_embed(txt_ids) + self.modality[5]
        cn = self.canvas_embed(cnv_raw) + self.modality[6]
        return torch.cat([st, it, pr, au, vi, tx, cn], dim=0)

    # ------------------------------------------------------------------
    # developmental surgery: earlier learning survives the new acuity
    # ------------------------------------------------------------------
    @torch.no_grad()
    def grow_stage(self, new_stage: int) -> None:
        old, new = self.spec, self.cfg.stages[new_stage]
        d = self.cfg.d_model
        w = self.vision_embed.weight.data          # (d, old_patch_dim)
        b = self.vision_embed.bias.data
        # (d, p, p, c) -> resample spatially -> replicate channels
        w4 = w.reshape(d, old.patch, old.patch, old.channels).permute(0, 3, 1, 2)
        w4 = torch.nn.functional.interpolate(
            w4, size=(new.patch, new.patch), mode="bilinear", align_corners=False)
        if new.channels != old.channels:
            w4 = w4.repeat(1, new.channels // old.channels, 1, 1) / (
                new.channels / old.channels)
        w_new = w4.permute(0, 2, 3, 1).reshape(d, new.patch_dim)
        layer = nn.Linear(new.patch_dim, d)
        layer.weight.data.copy_(w_new)
        layer.bias.data.copy_(b)
        self.vision_embed = layer.to(w.device)
        self.stage = int(new_stage)
