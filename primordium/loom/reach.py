"""
Reach — attention over a lifetime, not a context window.

An LLM attends within its context, over other people's text. The Reach
lets the Loom attend over ITS OWN lived history: salient moments are
banked as latents (no words, ever), and each live tick the present
queries the bank; the closest distant memories enter the sequence as
extra attendable tokens ahead of the causal mask, so every real token
can lean on them. Recent ticks are excluded — the window already holds
the near past; the Reach exists to bring back the far one.

Honesty rules, same as everything here:
- Born empty. No past, no tokens, bit-identical forward (tested).
- Every retrieval is measurable (count, similarity, age), and the
  bank's worth is a counterfactual: the same tick re-scored without
  memory, gain published as an EMA — not a story.
- The bank stores what the encoders said THEN. Encoders drift as it
  develops, so old entries slowly go out of focus — a documented
  limit, softened by salience-gated writing and finite capacity.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Reach(nn.Module):
    def __init__(self, cfg, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        cap = cfg.reach_capacity
        self.value = nn.Linear(d, d)         # how a memory speaks now
        self.mem_pos = nn.Parameter(torch.randn(d) * 0.01)
        # the bank: plain tensors, managed as a ring (not parameters)
        self._vecs = torch.zeros(cap, d, device=device)
        self._ticks = torch.full((cap,), -1, dtype=torch.long,
                                 device=device)
        self._salience = torch.zeros(cap, device=device)
        self._next = 0
        self.size = 0
        # measured life
        self.writes = 0
        self.hits_total = 0
        self.gain_ema = 0.0
        self.probes = 0
        self.last_sim = 0.0
        self.last_ages: list = []

    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(self, vec: torch.Tensor, tick: int, salience: float) -> None:
        v = vec.detach().float().flatten()
        n = float(v.norm())
        if n < 1e-6:
            return
        i = self._next
        self._vecs[i] = v / n
        self._ticks[i] = int(tick)
        self._salience[i] = float(salience)
        self._next = (i + 1) % self._vecs.shape[0]
        self.size = min(self.size + 1, self._vecs.shape[0])
        self.writes += 1

    def retrieve(self, query: torch.Tensor, now_tick: int
                 ) -> Tuple[Optional[torch.Tensor], Dict]:
        """Top-k distant memories as finished tokens (k, d), with the
        measured facts of the retrieval. None when there is nothing
        old enough to reach for."""
        cfg = self.cfg
        if self.size == 0:
            return None, {"n": 0}
        q = query.detach().float().flatten()
        qn = float(q.norm())
        if qn < 1e-6:
            return None, {"n": 0}
        q = q / qn
        vecs = self._vecs[:self.size]
        ticks = self._ticks[:self.size]
        old = ticks <= (now_tick - cfg.reach_exclude_recent)
        if not bool(old.any()):
            return None, {"n": 0}
        sims = vecs @ q
        sims = torch.where(old, sims, torch.full_like(sims, -2.0))
        k = min(cfg.reach_topk, int(old.sum()))
        top_sims, top_idx = sims.topk(k)
        keep = top_sims > cfg.reach_min_sim
        if not bool(keep.any()):
            return None, {"n": 0}
        top_idx, top_sims = top_idx[keep], top_sims[keep]
        # grads flow into value/mem_pos; the stored past stays as it was
        tokens = self.value(vecs[top_idx]) + self.mem_pos
        ages = (now_tick - ticks[top_idx]).tolist()
        self.hits_total += len(ages)
        self.last_sim = float(top_sims.mean())
        self.last_ages = ages
        return tokens, {"n": len(ages), "sim": round(self.last_sim, 4),
                        "ages": ages}

    def observe_gain(self, gain: float) -> None:
        self.probes += 1
        k = min(self.probes, 16)
        self.gain_ema += (float(gain) - self.gain_ema) / k

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"size": self.size, "writes": self.writes,
                "hits": self.hits_total, "sim": round(self.last_sim, 4),
                "ages": self.last_ages[:8],
                "gain": round(self.gain_ema, 6), "probes": self.probes}

    def bank_state(self) -> dict:
        return {"vecs": self._vecs[:self.size].detach().cpu(),
                "ticks": self._ticks[:self.size].detach().cpu(),
                "salience": self._salience[:self.size].detach().cpu(),
                "next": self._next, "writes": self.writes,
                "gain_ema": self.gain_ema, "probes": self.probes,
                "hits": self.hits_total}

    @torch.no_grad()
    def load_bank_state(self, st: dict) -> None:
        v = st.get("vecs")
        if v is None or v.numel() == 0:
            return
        n = min(v.shape[0], self._vecs.shape[0])
        dev = self._vecs.device
        self._vecs[:n] = v[:n].to(dev)
        self._ticks[:n] = st["ticks"][:n].to(dev)
        self._salience[:n] = st["salience"][:n].to(dev)
        self.size = n
        self._next = int(st.get("next", n % self._vecs.shape[0]))
        self.writes = int(st.get("writes", n))
        self.gain_ema = float(st.get("gain_ema", 0.0))
        self.probes = int(st.get("probes", 0))
        self.hits_total = int(st.get("hits", 0))

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev = next(self.value.parameters()).device
        self._vecs = self._vecs.to(dev)
        self._ticks = self._ticks.to(dev)
        self._salience = self._salience.to(dev)
        return out
