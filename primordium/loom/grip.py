"""
Grip — the hand made load-bearing.

Era 6's action-conditioning probe measured an honest negative: zeroing
the paint efference changed canvas prediction by ~0.05%. The efference
was an input the world model IGNORED, because the previous canvas sits
right there in context and copying it is nearly free. Chasing that
number exposed the structural truth: the stroke that paints canvas_t is
only recorded in proprio_t — causally INVISIBLE to the prediction
formed a tick earlier — so with the canvas visible, efference is
redundant BY CONSTRUCTION. The open-eyes probe measures redundancy, not
use. A mind whose predictions don't run through its own actions can
never plan with them.

The grip's pressures, each measured — and each shaped by two negative
results that are kept on the record:

  HAND MODEL — an explicit action-conditioned forward model: a small
  head learns, SUPERVISED on every lived stroke, what that stroke does
  to the canvas latents (action -> per-token latent delta; zero-action
  pairs teach it that no stroke means no change). This is direct
  regression with a strong gradient. It exists because the implicit
  path failed twice, measured: (1) hoping the trunk's attention would
  discover a lone proprio token among hundreds — ratio pinned at
  1.0000 after 4000 hot batched steps; (2) injecting raw efference
  embeddings into the masked slots for the trunk to decode end-to-end
  — the trunk learned to IGNORE the noisy slots instead (an easy local
  optimum) and the ratio stayed 1.0000. Biology reached the same
  design: motor commands are copied into sensory maps as corollary
  discharge, not left for attention to discover.

  BLINDFOLD — every few ticks, a second forward pass re-lives the same
  window with the RECENT canvases replaced by mask + the hand model's
  RUNNING RECONSTRUCTION (last visible canvas latents plus the
  accumulated predicted stroke deltas). Only the canvas prediction is
  scored, delta-weighted toward the tokens the strokes changed so the
  near-static background cannot drown the signal. The trunk learns to
  lean on the reconstruction; the reconstruction exists only through
  the efference.

  INVERSE DYNAMICS — a small head infers the executed efference a_t
  from (the lived hidden state before the action, the context-free EMA
  encoding of the moment after). Honest caveat, on the record: the
  "before" hidden can see a_{t-1}, and actions autocorrelate — a
  partial shortcut exists. The head's error is tracked, not trusted.

  PROBE — the counterfactual, run live UNDER THE BLINDFOLD: the same
  window, once with true paint efference and once with it erased
  (which collapses the reconstruction to the stale visible base).
  ratio > 1 means the hand measurably matters to what it expects of
  its canvas. Published, watched, and checkpoint-carried.

The blindfolded mode is the doorway later eras need: predicting the
consequences of actions WITHOUT seeing them is what planning is.

Consumers: the Watch (stream "grip"), the Pulse (event "grip"), the
dashboard HUD. Falsifiable test: test_v3e7_grip.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from primordium.loom.tokenizer import SensoryTokenizer

# paint efference dims within PROPRIO: [gate,x,y,radius,r,g,b,alpha]
PAINT_SLICE = slice(15, 23)
N_PAINT = PAINT_SLICE.stop - PAINT_SLICE.start


class Grip(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_cnv = (cfg.canvas_res // cfg.canvas_patch) ** 2
        # what a hidden canvas feels like — learned, not zero
        self.mask = nn.Parameter(torch.randn(d) * 0.02)
        # the hand model: stroke -> per-token canvas latent delta
        self.hand = nn.Sequential(
            nn.Linear(N_PAINT, 128), nn.GELU(),
            nn.Linear(128, self.n_cnv * d))
        # reconstruction adapter into embed space; identity at birth
        self.rec_proj = nn.Linear(d, d)
        nn.init.eye_(self.rec_proj.weight)
        nn.init.zeros_(self.rec_proj.bias)
        # inverse dynamics: (state before, moment after) -> the action
        self.inv = nn.Sequential(
            nn.Linear(2 * d, 128), nn.GELU(),
            nn.Linear(128, SensoryTokenizer.PROPRIO_DIM))
        # measured signals ride in buffers so a checkpoint carries them
        self.register_buffer("ratio_ema", torch.tensor(0.0))
        self.register_buffer("inv_ema", torch.tensor(0.0))
        self.register_buffer("grip_ema", torch.tensor(0.0))
        self.register_buffer("hand_ema", torch.tensor(0.0))
        self.register_buffer("probes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    def deltas(self, paints: torch.Tensor) -> torch.Tensor:
        """(n, 8) strokes -> (n, n_cnv, d) predicted latent deltas."""
        return self.hand(paints).view(paints.shape[0], self.n_cnv, -1)

    def hand_loss(self, paints: torch.Tensor,
                  true_deltas: torch.Tensor) -> torch.Tensor:
        """Supervised forward dynamics: what did the stroke actually do
        to the canvas latents? Zero-action pairs included by the caller
        so hand(0) is pulled toward 'no change'."""
        return F.smooth_l1_loss(self.deltas(paints), true_deltas.detach())

    def mask_canvas(self, tokens: torch.Tensor, group_size: int,
                    n_cnv: int, m_groups: int,
                    paints: torch.Tensor = None,
                    base: torch.Tensor = None) -> torch.Tensor:
        """Blindfold: hide the canvas tokens of the m groups ENDING at
        the prediction-source group (the second-to-last). The final
        group's canvas is untouched — causally it cannot leak into the
        prediction, and masking it would only corrupt the online
        latents. Canvases older than the horizon stay visible.

        With `paints` (n_groups, 8) and `base` (n_cnv, d — latents of
        the last VISIBLE canvas), each masked slot receives mask + the
        running reconstruction: base plus the hand model's accumulated
        stroke deltas up to that group. Erased efference collapses the
        reconstruction to the stale base — exactly what the
        counterfactual measures."""
        n_groups = tokens.shape[0] // group_size
        masked = tokens.clone()
        lo = max(0, n_groups - 1 - m_groups)
        rec = base
        for g in range(lo, n_groups - 1):
            fill = self.mask
            if paints is not None and rec is not None:
                rec = rec + self.deltas(paints[g: g + 1])[0]
                fill = fill + self.rec_proj(rec)
            c0 = (g + 1) * group_size - n_cnv
            masked[c0: (g + 1) * group_size] = fill
        return masked

    def inverse(self, h_before: torch.Tensor,
                target_after: torch.Tensor) -> torch.Tensor:
        return self.inv(torch.cat([h_before, target_after.detach()], dim=-1))

    @staticmethod
    def cnv_loss(pred_cnv: torch.Tensor, target_cnv: torch.Tensor,
                 weights: torch.Tensor = None) -> torch.Tensor:
        """Same geometry as the objective's canvas term: normalized
        smooth-L1 in latent space; optionally delta-weighted per token."""
        p = F.normalize(pred_cnv, dim=-1)
        t = F.normalize(target_cnv, dim=-1)
        per_tok = F.smooth_l1_loss(p, t, reduction="none").mean(-1)
        if weights is None:
            return per_tok.mean()
        return (per_tok * weights).sum()

    @staticmethod
    @torch.no_grad()
    def delta_weights(base_cnv: torch.Tensor,
                      target_cnv: torch.Tensor) -> torch.Tensor:
        """Per-token weights ∝ how much each canvas token changed from
        the last VISIBLE canvas to the predicted one — the strokes are
        where the action signal lives. A floor keeps a quarter of the
        mass on the static background so it is never unlearned."""
        d = (F.normalize(target_cnv, dim=-1)
             - F.normalize(base_cnv, dim=-1)).norm(dim=-1)
        n = d.shape[0]
        delta = d / d.sum() if float(d.sum()) > 1e-8 \
            else torch.full_like(d, 1.0 / n)
        return 0.25 / n + 0.75 * delta

    # ------------------------------------------------------------------
    @torch.no_grad()
    def note_grip(self, loss: float) -> None:
        self.steps += 1
        a = 0.05 if int(self.steps) > 1 else 1.0
        self.grip_ema.mul_(1 - a).add_(a * loss)

    @torch.no_grad()
    def note_hand(self, loss: float) -> None:
        a = 0.05 if float(self.hand_ema) != 0.0 else 1.0
        self.hand_ema.mul_(1 - a).add_(a * loss)

    @torch.no_grad()
    def note_inv(self, loss: float) -> None:
        a = 0.02 if float(self.inv_ema) != 0.0 else 1.0
        self.inv_ema.mul_(1 - a).add_(a * loss)

    @torch.no_grad()
    def note_probe(self, ratio: float) -> None:
        self.probes += 1
        a = 0.2 if int(self.probes) > 1 else 1.0
        self.ratio_ema.mul_(1 - a).add_(a * ratio)

    def snapshot(self) -> dict:
        return {
            "ratio": round(float(self.ratio_ema), 4),
            "probes": int(self.probes),
            "grip_loss": round(float(self.grip_ema), 4),
            "hand_loss": round(float(self.hand_ema), 5),
            "inv_loss": round(float(self.inv_ema), 4),
            "steps": int(self.steps),
        }
