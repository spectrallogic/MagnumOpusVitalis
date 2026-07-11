"""
Fringe — the soft, fast-plastic rim around the hard core.

Design observation: brains stay quick because the edges stay soft — a
hive of small eager learners around the stable middle, always trying to
pick something up. Here that is mechanism, not metaphor:

- SPROUTS are tiny low-rank adapters (rank ~4) attached in parallel to
  the EDGE blocks' down-projections (first and last block; the middle
  layers are the hard core). Born silent (B = 0), they ride the same
  backward pass as the core but at ~20x learning rate, scaled further
  by surprise — the fringe learns hardest exactly where prediction
  fails. Heavy weight decay makes an unused sprout fade back to silence.

- UTILITY is measured, never assumed: on replay, one sprout at a time
  is ablated and the same window is re-scored — utility is the loss it
  was actually saving. An EMA of that counterfactual is the only judge.

- CONSOLIDATION: during sleep, a sprout with sustained positive utility
  HARDENS INTO THE CORE — its low-rank delta is added exactly into the
  host weight (W += B@A, exact for a parallel adapter on a Linear) and
  the sprout is reborn empty, eager again. Harmful or long-idle sprouts
  are recycled. Discoveries move inward; the edge stays soft.

Unlike biology, every consolidation here is an exact algebraic merge
with a measured justification in the Pulse.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class Sprout(nn.Module):
    """One eager little learner: out += B(A(x)), born silent."""

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.A = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)
        self.reinit()
        # measured life
        self.util_ema = 0.0
        self.probes = 0
        self.age_ticks = 0
        self.merges = 0
        self.recycles = 0
        self.enabled = True          # probe gate, not a parameter

    def reinit(self) -> None:
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)             # silent at birth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))

    def delta_weight(self) -> torch.Tensor:
        """The exact dense weight this sprout adds: (d_out, d_in)."""
        return self.B.weight @ self.A.weight

    def observe_utility(self, util: float) -> None:
        self.probes += 1
        k = min(self.probes, 8)
        self.util_ema += (float(util) - self.util_ema) / k

    def reset_life(self) -> None:
        self.util_ema = 0.0
        self.probes = 0
        self.age_ticks = 0


class Fringe(nn.Module):
    """Manages the sprout hive on the core's edge layers."""

    def __init__(self, cfg, core):
        super().__init__()
        self.cfg = cfg
        per = int(cfg.fringe_sprouts_per_site)
        self.sprouts = nn.ModuleList()
        self._site_of: List[Tuple[str, nn.Linear]] = []   # parallel list
        self._handles = []
        if per <= 0 or not core.blocks:
            return
        edge_ids = sorted({0, len(core.blocks) - 1})
        for bi in edge_ids:
            host: nn.Linear = core.blocks[bi].mlp[2]      # down-projection
            site_name = f"block{bi}.mlp.down"
            group: List[Sprout] = []
            for _ in range(per):
                s = Sprout(host.in_features, host.out_features,
                           cfg.fringe_rank)
                self.sprouts.append(s)
                self._site_of.append((site_name, host))
                group.append(s)
            self._handles.append(host.register_forward_hook(
                self._make_hook(group)))

    @staticmethod
    def _make_hook(group: List[Sprout]):
        def hook(_module, args, output):
            x = args[0]
            for s in group:
                if s.enabled:
                    output = output + s(x)
            return output
        return hook

    # ------------------------------------------------------------------
    def detach(self) -> None:
        """Remove every hook: the fringe lets go of the core so the
        Bloom can operate (or a checkpoint can regrow the body)."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __len__(self) -> int:
        return len(self.sprouts)

    def tick(self) -> None:
        for s in self.sprouts:
            s.age_ticks += 1

    def probe_target(self, step: int) -> int:
        return step % max(1, len(self.sprouts))

    def consolidate(self, ema_slow, cfg) -> Tuple[List[dict], List[dict]]:
        """Sleep-time judgement. Returns (merged, recycled) reports."""
        merged, recycled = [], []
        if ema_slow is None or not self.sprouts:
            return merged, recycled
        theta = cfg.fringe_merge_frac * max(float(ema_slow), 1e-8)
        for i, s in enumerate(self.sprouts):
            site, host = self._site_of[i]
            if s.probes < cfg.fringe_min_probes:
                continue
            if s.util_ema > theta:
                with torch.no_grad():             # the edge hardens inward
                    host.weight += s.delta_weight()
                    s.reinit()
                s.merges += 1
                merged.append({"sprout": i, "site": site,
                               "util": round(s.util_ema, 6)})
                s.reset_life()
            elif (s.util_ema < -theta
                    or (s.age_ticks > cfg.fringe_idle_age
                        and abs(s.util_ema) < 0.25 * theta)):
                with torch.no_grad():
                    s.reinit()
                s.recycles += 1
                recycled.append({"sprout": i, "site": site,
                                 "util": round(s.util_ema, 6)})
                s.reset_life()
        return merged, recycled

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "sprouts": [
                {"site": self._site_of[i][0],
                 "util": round(s.util_ema, 6), "probes": s.probes,
                 "age": s.age_ticks, "merges": s.merges,
                 "recycles": s.recycles,
                 "norm": round(float(s.B.weight.norm()), 5)}
                for i, s in enumerate(self.sprouts)
            ],
        }

    def stats_state(self) -> List[dict]:
        return [{"util_ema": s.util_ema, "probes": s.probes,
                 "age_ticks": s.age_ticks, "merges": s.merges,
                 "recycles": s.recycles} for s in self.sprouts]

    def load_stats_state(self, stats: List[dict]) -> None:
        for s, st in zip(self.sprouts, stats):
            s.util_ema = float(st.get("util_ema", 0.0))
            s.probes = int(st.get("probes", 0))
            s.age_ticks = int(st.get("age_ticks", 0))
            s.merges = int(st.get("merges", 0))
            s.recycles = int(st.get("recycles", 0))
