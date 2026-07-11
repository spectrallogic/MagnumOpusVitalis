"""V2-M6 — the Fringe: eager low-rank sprouts on the edge blocks,
judged by counterfactual ablation, hardened into the core by exact
merge. The soft edge is mechanism, not metaphor."""

import os

import numpy as np
import pytest
import torch

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_sprout_merge_is_exact():
    """W += B@A must be indistinguishable from the live adapter."""
    cfg = OrganismConfig()
    from primordium.loom.core import LoomCore
    from primordium.loom.fringe import Fringe
    torch.manual_seed(3)
    core = LoomCore(cfg)
    fringe = Fringe(cfg, core)
    assert len(fringe) == 2 * cfg.fringe_sprouts_per_site
    s = fringe.sprouts[0]
    with torch.no_grad():                      # give it something to say
        s.B.weight.normal_(std=0.05)

    tokens = torch.randn(cfg.window * 37, cfg.d_model) * 0.1
    before = core(tokens, group_size=37)

    _site, host = fringe._site_of[0]
    with torch.no_grad():                      # the exact hardening step
        host.weight += s.delta_weight()
        s.reinit()
    after = core(tokens, group_size=37)
    assert torch.allclose(before, after, atol=1e-5), \
        float((before - after).abs().max())
    assert float(s.B.weight.abs().max()) == 0.0    # reborn silent


def test_fringe_lives_probes_and_consolidates():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m6"
    ctx, board, flow = build(cfg)
    w = ctx.worker
    assert len(w.fringe) > 0 and w.opt_fringe is not None
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=61)

    b0 = [s.B.weight.detach().clone() for s in w.fringe.sprouts]
    for _ in range(cfg.window + 6):
        world.step(0.15)
        w.step_once()

    # the eager edge actually moved off silence in a handful of ticks
    moved = [float((s.B.weight.detach() - b).abs().max())
             for s, b in zip(w.fringe.sprouts, b0)]
    assert max(moved) > 0, "no sprout learned anything"
    assert all(s.age_ticks > 0 for s in w.fringe.sprouts)

    # replay runs a real counterfactual probe on a lived window
    w._job_replay()
    probed = [s for s in w.fringe.sprouts if s.probes > 0]
    assert probed and all(np.isfinite(s.util_ema) for s in probed)

    # published honestly
    snap = w.get_published().get("fringe", {})
    assert len(snap.get("sprouts", [])) == len(w.fringe)

    # consolidation judgement: a proven-useful sprout hardens inward
    # (exactly), a proven-harmful one is recycled
    good, bad = w.fringe.sprouts[0], w.fringe.sprouts[-1]
    with torch.no_grad():
        good.B.weight.normal_(std=0.02)
    good.util_ema, good.probes = 1.0, cfg.fringe_min_probes
    bad.util_ema, bad.probes = -1.0, cfg.fringe_min_probes
    _gsite, ghost = w.fringe._site_of[0]
    host_before = ghost.weight.detach().clone()
    delta = good.delta_weight().detach().clone()
    merged, recycled = w.fringe.consolidate(w.objective.ema_slow, cfg)
    assert [m["sprout"] for m in merged] == [0]
    assert [r["sprout"] for r in recycled] == [len(w.fringe) - 1]
    assert torch.allclose(ghost.weight.detach(), host_before + delta,
                          atol=1e-6)
    assert good.merges == 1 and float(good.B.weight.abs().max()) == 0.0
    assert bad.recycles == 1

    # the whole hive survives a checkpoint
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    with torch.no_grad():
        w.fringe.sprouts[1].B.weight.normal_(std=0.03)
    w.fringe.sprouts[1].util_ema = 0.123
    w.fringe.sprouts[1].probes = 7
    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert torch.allclose(ctx2.worker.fringe.sprouts[1].B.weight.cpu(),
                          w.fringe.sprouts[1].B.weight.cpu())
    assert ctx2.worker.fringe.sprouts[1].probes == 7
    assert ctx2.worker.fringe.sprouts[0].merges == 1
    w.stop()
    ctx2.worker.stop()


def test_fringe_absent_when_disabled():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m6_off"
    cfg.fringe_sprouts_per_site = 0
    ctx, board, flow = build(cfg)
    w = ctx.worker
    assert len(w.fringe) == 0 and w.opt_fringe is None
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=62)
    for _ in range(4):
        world.step(0.15)
        w.step_once()                          # life goes on without it
    assert w.get_published().get("fringe", {}) == {}
    w.stop()


@pytest.mark.skipif(not os.environ.get("PRIMORDIUM_LONG"),
                    reason="fringe A/B learning-speed soak: PRIMORDIUM_LONG=1")
def test_fringe_speeds_early_learning():
    """The claim behind the feature, falsifiably: with the soft edge on,
    early womb learning is at least not slower, and typically faster."""
    def life(sprouts):
        cfg = OrganismConfig()
        cfg.run_name = f"_test_v2m6_ab{sprouts}"
        cfg.fringe_sprouts_per_site = sprouts
        torch.manual_seed(9)
        ctx, board, flow = build(cfg)
        world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=63)
        tail = []
        for i in range(1500):
            world.step(0.15)
            ctx.worker.step_once()
            if i % 40 == 0:
                ctx.worker._job_replay()
            if i > 1200:
                p = ctx.worker.get_published().get("loss")
                if p is not None:
                    tail.append(p)
        ctx.worker.stop()
        return float(np.mean(tail))

    on, off = life(3), life(0)
    print(f"fringe on={on:.5f} off={off:.5f}")
    assert on <= off * 1.05
