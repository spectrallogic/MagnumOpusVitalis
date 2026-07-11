"""Measure ms/tick and VRAM at each developmental stage (synthetic womb)."""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def main():
    cfg = OrganismConfig()
    cfg.run_name = "_bench"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea)
    w = ctx.worker

    for stage in range(len(cfg.stages)):
        if stage > 0:
            w._advance_stage()  # noqa: SLF001 — bench pokes on purpose
        for _ in range(5):
            world.step(); w.step_once()
        t0 = time.monotonic()
        n = 30
        for _ in range(n):
            world.step()
            w.step_once()
        ms = (time.monotonic() - t0) / n * 1000
        vram = (torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else 0)
        print(f"  stage {stage} ({cfg.stages[stage].label:>12}): "
              f"{ms:6.1f} ms/tick   peak VRAM {vram:7.1f} MB")
    w.stop()


if __name__ == "__main__":
    main()
