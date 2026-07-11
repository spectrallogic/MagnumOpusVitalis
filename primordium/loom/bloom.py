"""
Bloom — the core grows, and every growth is exact.

When measured learning saturates (progress flatlined while loss is
still high — the brain is full, not done), the Bloom adds capacity
during sleep with function-preserving surgery:

- DEPTH: a new block whose residual branches are born at exactly zero
  (attn.out_proj and mlp-down zeroed) — the block is the identity on
  the day of its birth, and starts learning the moment gradients flow.
  (Zero-init means the block's inner attention learns only after its
  output gate moves off zero — a documented, gentle ramp-in.)
- WIDTH: new hidden units whose output columns are zero — silent at
  birth, same trick as the Fringe's B=0.

Where to grow is measured, not guessed: per-block strain gauges (EMA
of gradient norm) point at the block working hardest. How much room
there is comes from the actual GPU (`torch.cuda.mem_get_info`) — the
same life resumed on a bigger machine simply has more headroom and
keeps growing. No ceiling constant exists anywhere in this file.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from primordium.loom.core import Block, LoomCore

# fp32 training footprint per parameter: weight + grad + 2 AdamW moments
BYTES_PER_PARAM = 16


@torch.no_grad()
def insert_block(core: LoomCore, idx: int) -> dict:
    """Insert an identity-at-birth block at `idx`. Exactly function-
    preserving: both residual branches start at zero."""
    cfg = core.cfg
    dev = next(core.parameters()).device
    blk = Block(cfg.d_model, cfg.n_heads, cfg.d_model * cfg.mlp_ratio)
    blk.apply(LoomCore._init)
    nn.init.zeros_(blk.attn.out_proj.weight)
    if blk.attn.out_proj.bias is not None:
        nn.init.zeros_(blk.attn.out_proj.bias)
    nn.init.zeros_(blk.mlp[2].weight)
    nn.init.zeros_(blk.mlp[2].bias)
    core.blocks.insert(idx, blk.to(dev))
    return core.anatomy()


@torch.no_grad()
def widen_mlp(core: LoomCore, bi: int, k: int) -> dict:
    """Add k hidden units to block bi's MLP. New up-rows are live,
    new down-COLUMNS are zero — the new units are silent at birth."""
    blk = core.blocks[bi]
    up: nn.Linear = blk.mlp[0]
    down: nn.Linear = blk.mlp[2]
    dev, m = up.weight.device, up.out_features

    new_up = nn.Linear(up.in_features, m + k).to(dev)
    nn.init.normal_(new_up.weight, std=0.02)
    nn.init.zeros_(new_up.bias)
    new_up.weight[:m] = up.weight
    new_up.bias[:m] = up.bias

    new_down = nn.Linear(m + k, down.out_features).to(dev)
    nn.init.zeros_(new_down.weight)          # silence first...
    new_down.weight[:, :m] = down.weight     # ...then the old voice back
    new_down.bias.copy_(down.bias)

    blk.mlp[0] = new_up
    blk.mlp[2] = new_down
    return core.anatomy()


class Bloom:
    """Strain gauges + growth decisions. Owns no schedule of its own:
    the Loom asks, during sleep, when the capacity gates all hold."""

    def __init__(self, cfg, core: LoomCore):
        self.cfg = cfg
        self.core = core
        self.strain: List[float] = [0.0] * len(core.blocks)
        self.blooms_total = 0
        self.last_bloom_tick = 0

    def rebind(self, core: LoomCore) -> None:
        """After a checkpoint regrows the body, point at the new one."""
        self.core = core
        if len(self.strain) != len(core.blocks):
            self.strain = [0.0] * len(core.blocks)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def observe_strain(self) -> None:
        """Called right after backward: EMA of per-block gradient norm.
        One GPU sync per call, nothing else."""
        sq = []
        for blk in self.core.blocks:
            s = None
            for p in blk.parameters():
                if p.grad is not None:
                    g = p.grad.pow(2).sum()
                    s = g if s is None else s + g
            sq.append(s if s is not None else torch.zeros((), device=next(
                self.core.parameters()).device))
        norms = torch.stack(sq).sqrt().tolist()
        if len(self.strain) != len(norms):
            self.strain = [0.0] * len(norms)
        for i, n in enumerate(norms):
            self.strain[i] += 0.05 * (float(n) - self.strain[i])

    # ------------------------------------------------------------------
    def headroom(self, delta_params: int) -> bool:
        """Is there honestly room to grow by this much, here?"""
        dev = next(self.core.parameters()).device
        if dev.type != "cuda":
            return True
        free, _total = torch.cuda.mem_get_info(dev)
        return free > delta_params * BYTES_PER_PARAM * \
            self.cfg.bloom_vram_headroom

    def _plan(self) -> Tuple[str, int, int]:
        """(kind, site, delta_params) for the next growth quantum."""
        cfg, d = self.cfg, self.cfg.d_model
        if (self.blooms_total + 1) % cfg.bloom_block_every == 0:
            site = len(self.core.blocks) - 1     # edges stay edges
            m = d * cfg.mlp_ratio
            delta = (4 * d * d + 4 * d) + (2 * d * m + m + d) + 4 * d
            return "block", site, delta
        inner = [s for s in self.strain]
        site = max(range(len(inner)), key=lambda i: inner[i])
        k = cfg.bloom_widen_k
        blk = self.core.blocks[site]
        delta = k * (blk.mlp[0].in_features + 1) \
            + k * blk.mlp[2].out_features
        return "width", site, delta

    def can_grow(self) -> bool:
        _kind, _site, delta = self._plan()
        return self.headroom(delta)

    def grow(self, tick_id: int) -> Optional[dict]:
        """Perform the planned quantum. Returns an honest report, or
        None if the room ran out between planning and cutting."""
        kind, site, delta = self._plan()
        if not self.headroom(delta):
            return None
        before = self.core.anatomy()["params"]
        if kind == "block":
            insert_block(self.core, site)
            self.strain.insert(site, 0.0)
        else:
            widen_mlp(self.core, site, self.cfg.bloom_widen_k)
        self.blooms_total += 1
        self.last_bloom_tick = tick_id
        return {"growth": kind, "site": site,
                "params_before": before,
                "params_after": self.core.anatomy()["params"]}

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        a = self.core.anatomy()
        return {"blocks": a["n_blocks"], "mlp_dims": a["mlp_dims"],
                "params": a["params"], "blooms": self.blooms_total,
                "strain": [round(s, 4) for s in self.strain]}

    def state_dict(self) -> dict:
        return {"strain": list(self.strain),
                "blooms_total": self.blooms_total,
                "last_bloom_tick": self.last_bloom_tick}

    def load_state_dict(self, st: Dict) -> None:
        self.strain = list(st.get("strain", self.strain))
        self.blooms_total = int(st.get("blooms_total", 0))
        self.last_bloom_tick = int(st.get("last_bloom_tick", 0))
