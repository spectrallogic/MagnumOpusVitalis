"""M1 — it learns its womb: loss falls, latents don't collapse."""

import numpy as np

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_online_learning_reduces_loss():
    cfg = OrganismConfig()
    cfg.run_name = "_test_m1"
    ctx, board, flow = build(cfg)          # no flow.start(): worker only
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=3)
    w = ctx.worker

    losses = []
    for i in range(600):
        world.step(0.15)
        w.step_once()
        pub = w.get_published()
        if pub.get("loss") is not None:
            losses.append(pub["loss"])

    assert len(losses) > 400
    early = float(np.mean(losses[20:80]))
    late = float(np.mean(losses[-60:]))
    assert late < early * 0.75, f"loss did not fall: {early:.4f} -> {late:.4f}"

    std = w.get_published().get("latent_std", 0)
    assert std > 0.05, f"latent collapse: std={std}"
    w.stop()
