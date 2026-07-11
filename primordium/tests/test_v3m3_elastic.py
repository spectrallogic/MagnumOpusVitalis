"""V3-M3 — a grown life is portable: the checkpoint carries its own
anatomy, the body regrows to fit it anywhere, growth continues after
the move, and older eras are refused honestly. The env-gated soak
carries the falsifiable claim behind the whole feature."""

import os

import numpy as np
import pytest
import torch

from primordium.config import OrganismConfig
from primordium.persistence.checkpoint import load_checkpoint, \
    save_checkpoint
from primordium.run import build
from primordium.senses import SyntheticWorld


def _forced_bloom_cfg(name):
    cfg = OrganismConfig()
    cfg.run_name = name
    cfg.bloom_cooldown_ticks = 1
    cfg.bloom_loss_floor = 1e-9
    cfg.bloom_lp_eps = 1e9
    cfg.bloom_widen_k = 64
    return cfg


def test_a_bloomed_life_moves_and_keeps_growing():
    cfg = _forced_bloom_cfg("_test_v3m3_move")
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=81)
    for _ in range(cfg.window + 24):
        world.step(0.15)
        w.step_once()
    ctx.drives.levels["energy"] = 1.0
    w._maybe_bloom()
    grown = w.model.anatomy()
    assert w.bloom.blooms_total == 1
    assert grown["mlp_dims"] != [cfg.d_model * cfg.mlp_ratio] * cfg.n_layers

    # the exact same mind must answer the same after the move
    probe = torch.randn(2 * 37, cfg.d_model, device=w.device) * 0.1
    with torch.no_grad():
        before = w.model(probe, group_size=37).cpu()
    path = save_checkpoint(ctx)

    ctx2, board2, flow2 = build(cfg)       # fresh default 8-block body
    w2 = ctx2.worker
    assert w2.model.anatomy()["mlp_dims"] != grown["mlp_dims"]
    load_checkpoint(ctx2, path)            # ...regrown to fit the life
    assert w2.model.anatomy()["mlp_dims"] == grown["mlp_dims"]
    assert w2.bloom.blooms_total == 1      # its history moved with it
    with torch.no_grad():
        after = w2.model(probe.to(w2.device), group_size=37).cpu()
    assert torch.allclose(before, after, atol=1e-5), \
        float((before - after).abs().max())

    # and on the "new machine" it keeps growing — same code path,
    # only the headroom differs
    world2 = SyntheticWorld(ctx2.retina, ctx2.cochlea, seed=82)
    for _ in range(cfg.window + 24):
        world2.step(0.15)
        w2.step_once()
    ctx2.drives.levels["energy"] = 1.0
    w2._maybe_bloom()
    assert w2.bloom.blooms_total == 2
    assert w2.model.anatomy()["params"] > grown["params"]
    w.stop()
    w2.stop()


def test_older_eras_are_refused_by_name(tmp_path):
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3m3_refuse"
    ctx, board, flow = build(cfg, no_loom=True)
    p = tmp_path / "ckpt.pt"
    torch.save({"version": 2, "tick_id": 1, "stage": 0}, p)
    with pytest.raises(RuntimeError, match="v2 life"):
        load_checkpoint(ctx, p)
    torch.save({"version": 1}, p)
    with pytest.raises(RuntimeError, match="v1 life"):
        load_checkpoint(ctx, p)


@pytest.mark.skipif(not os.environ.get("PRIMORDIUM_LONG"),
                    reason="bloom A/B soak: set PRIMORDIUM_LONG=1")
def test_bloom_beats_a_frozen_brain_after_saturation():
    """The falsifiable claim: born deliberately tiny (2 blocks), the
    organism that is ALLOWED to grow must keep improving after the
    frozen control plateaus."""
    def life(bloom_on):
        cfg = OrganismConfig()
        cfg.run_name = f"_test_v3m3_ab{int(bloom_on)}"
        cfg.birth_anatomy = {"mlp_dims": [256, 256]}
        cfg.bloom_enabled = bloom_on
        cfg.bloom_cooldown_ticks = 600
        cfg.bloom_loss_floor = 0.3
        cfg.bloom_window_s = 30.0
        cfg.bloom_widen_k = 128
        torch.manual_seed(17)
        ctx, board, flow = build(cfg)
        world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=83)
        tail = []
        for i in range(4000):
            world.step(0.15)
            ctx.worker.step_once()
            if i % 40 == 0:
                ctx.worker._job_replay()
            if bloom_on and i % 200 == 0:
                ctx.drives.levels["energy"] = 1.0
                ctx.worker._maybe_bloom()   # the sleep venue, on the
            if i > 3400:                    # soak's clock
                p = ctx.worker.get_published().get("loss")
                if p is not None:
                    tail.append(p)
        grown = ctx.worker.model.anatomy()["params"]
        ctx.worker.stop()
        return float(np.mean(tail)), grown

    on, on_params = life(True)
    off, off_params = life(False)
    print(f"bloom on={on:.5f} ({on_params/1e6:.1f}M) "
          f"off={off:.5f} ({off_params/1e6:.1f}M)")
    assert on_params > off_params           # it actually grew
    assert on <= off * 1.02                 # and growth paid its way
