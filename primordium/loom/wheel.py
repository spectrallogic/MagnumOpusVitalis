"""
Wheel — prediction within prediction, at the scale of moments.

The Loom predicts the next TICK (~150ms). The Wheel rides above it:
every `wheel_window_ticks` ticks it pools the lived latents into one
summary — a chunk of experience a few seconds wide — and a small
predictor learns to guess the NEXT summary from the last two. Fast
wheel inside slow wheel; each turns at its own rate.

What this buys, mechanically:
  - SLOW SURPRISE: the tick predictor recovers from a scene change in
    a few ticks; the Wheel's error says whether the new CHAPTER is
    still unexpected. A spike (z-scored against its own history, after
    warmup) raises arousal with its cause on the record, feeds the
    Watch as the "story" stream, and lands in the Pulse.
  - EXPECTATION AS CONTEXT: the predicted next summary re-enters the
    sequence as one extra token ahead of the causal mask (the Reach's
    own mechanism, reused) — the present can lean on what the coming
    stretch should feel like. Gradients flow into how the expectation
    SPEAKS (value/pos), never into the prediction it carries.

Honesty notes: the predictor trains on its own tiny optimizer against
detached summaries (targets never depend on it — no collapse path for
the predictor itself); born empty means no token and a bit-identical
forward until the first full window has been lived; the window length
is an authored clock constant, disclosed as such.
"""

from collections import deque
from statistics import median
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Wheel(nn.Module):
    def __init__(self, cfg, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.pred = nn.Sequential(
            nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, d))
        self.value = nn.Linear(d, d)      # how the expectation speaks
        self.wheel_pos = nn.Parameter(torch.randn(d) * 0.01)
        self.device_str = device
        self._opt = torch.optim.AdamW(self.pred.parameters(), lr=1e-3)

        self._acc: List[torch.Tensor] = []
        self._prev: Optional[torch.Tensor] = None      # summary t-1
        self._prev2: Optional[torch.Tensor] = None     # summary t-2
        self._expect: Optional[torch.Tensor] = None    # predicted t (detached)
        # calibration (Era 6): before each turn the wheel STATES its
        # confidence that the coming chapter will be at least as
        # predictable as its typical one (realized err <= tau, tau
        # frozen from the warmup median); Brier scores the statements.
        self._err_hist: deque = deque(maxlen=64)
        self._tau: Optional[float] = None
        self._stated_conf: Optional[float] = None
        self.calib_n = 0
        self._brier_sum = 0.0

        # measured life
        self.turns = 0
        self.loss_ema: Optional[float] = None
        self._err_m, self._err_v, self._err_n = 0.0, 1.0, 0
        self.last_err = 0.0
        self.last_z = 0.0

    # ------------------------------------------------------------------
    def spin(self, z_t: torch.Tensor) -> Optional[dict]:
        """Feed one tick's pooled latent. Returns a turn report every
        wheel_window_ticks ticks, else None."""
        self._acc.append(z_t.detach().float())
        if len(self._acc) < self.cfg.wheel_window_ticks:
            return None
        summary = torch.stack(self._acc).mean(0)
        self._acc = []
        self.turns += 1
        report = None

        # judge the expectation the previous turn committed to
        if self._expect is not None:
            err = float(1.0 - F.cosine_similarity(
                self._expect.unsqueeze(0), summary.unsqueeze(0)))
            self.last_err = err
            self.loss_ema = (err if self.loss_ema is None
                             else 0.9 * self.loss_ema + 0.1 * err)
            self._err_n = min(self._err_n + 1, 2000)
            self._err_m += (err - self._err_m) / self._err_n
            self._err_v += ((err - self._err_m) ** 2 - self._err_v) / self._err_n
            self.last_z = (err - self._err_m) / max(self._err_v ** 0.5, 1e-6)
            spiked = (self._err_n >= self.cfg.wheel_min_turns
                      and self.last_z >= self.cfg.wheel_spike_z)
            report = {"err": round(err, 5), "z": round(self.last_z, 2),
                      "turn": self.turns, "spiked": spiked}

            # score the confidence stated BEFORE this outcome existed
            if self._stated_conf is not None and self._tau is not None:
                outcome = 1.0 if err <= self._tau else 0.0
                self._brier_sum += (self._stated_conf - outcome) ** 2
                self.calib_n += 1
            self._err_hist.append(err)
            if self._tau is None and len(self._err_hist) >= 16:
                self._tau = float(median(self._err_hist))
            if self._tau is not None and len(self._err_hist) >= 8:
                recent = list(self._err_hist)[-16:]
                self._stated_conf = (
                    sum(1 for e in recent if e <= self._tau) / len(recent))

        # one gradient step for the predictor (inputs detached, tiny)
        if self._prev is not None and self._prev2 is not None:
            x = torch.cat([self._prev2, self._prev]).unsqueeze(0)
            self._opt.zero_grad(set_to_none=True)
            loss = 1.0 - F.cosine_similarity(
                self.pred(x), summary.unsqueeze(0))
            loss.mean().backward()
            self._opt.step()

        # commit the next expectation
        self._prev2, self._prev = self._prev, summary
        if self._prev2 is not None:
            with torch.no_grad():
                self._expect = self.pred(
                    torch.cat([self._prev2, self._prev]).unsqueeze(0)
                )[0].detach()
        return report

    def expectation_token(self) -> Optional[torch.Tensor]:
        """(1, d) token for the sequence, or None before the first
        expectation exists (born empty = bit-identical forward)."""
        if self._expect is None:
            return None
        return (self.value(self._expect) + self.wheel_pos).unsqueeze(0)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        out = {"turns": self.turns,
               "loss": (round(self.loss_ema, 5)
                        if self.loss_ema is not None else None),
               "err": round(self.last_err, 5),
               "z": round(self.last_z, 2)}
        if self.calib_n:
            out["calib"] = {"brier": round(self._brier_sum / self.calib_n, 4),
                            "n": self.calib_n,
                            "tau": round(self._tau, 5),
                            "conf": (round(self._stated_conf, 3)
                                     if self._stated_conf is not None
                                     else None)}
        return out

    def bank_state(self) -> Dict:
        return {"turns": self.turns, "loss_ema": self.loss_ema,
                "err_stats": [self._err_m, self._err_v, self._err_n],
                "calib": {"hist": list(self._err_hist), "tau": self._tau,
                          "conf": self._stated_conf, "n": self.calib_n,
                          "brier_sum": self._brier_sum},
                "prev": (self._prev.cpu() if self._prev is not None else None),
                "prev2": (self._prev2.cpu() if self._prev2 is not None else None),
                "expect": (self._expect.cpu()
                           if self._expect is not None else None)}

    def load_bank_state(self, st: Dict) -> None:
        self.turns = int(st.get("turns", 0))
        self.loss_ema = st.get("loss_ema")
        m, v, n = st.get("err_stats", [0.0, 1.0, 0])
        self._err_m, self._err_v, self._err_n = float(m), float(v), int(n)
        cal = st.get("calib") or {}
        self._err_hist = deque(cal.get("hist", []), maxlen=64)
        self._tau = cal.get("tau")
        self._stated_conf = cal.get("conf")
        self.calib_n = int(cal.get("n", 0))
        self._brier_sum = float(cal.get("brier_sum", 0.0))
        dev = self.wheel_pos.device
        for name in ("prev", "prev2", "expect"):
            t = st.get(name)
            setattr(self, f"_{name}", t.to(dev) if t is not None else None)
