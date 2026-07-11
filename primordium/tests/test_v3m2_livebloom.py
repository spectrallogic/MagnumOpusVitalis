"""V3-M2 — the core blooms in a real life: measured gates, sleep-venue
surgery, honest Pulse record, loss continuity across the cut."""

import numpy as np

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_cap_gates_are_measured_and_conservative():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3m2_gates"
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=71)

    g = w._cap_gates()
    assert set(g) == {"stuck", "plateau", "cooldown", "energy", "room"}
    assert all(v == 0.0 for v in g.values())        # no life, no verdicts

    for _ in range(cfg.window + 12):
        world.step(0.15)
        w.step_once()
    g = w._cap_gates()
    assert g["cooldown"] < 1.0                       # 30k ticks is a childhood
    assert g["room"] == 1.0                          # 12GB has room to grow
    assert 0 <= g["stuck"] <= 1.0

    # strain gauges saw real gradients on every block
    assert len(w.bloom.strain) == len(w.model.blocks)
    assert all(s > 0 for s in w.bloom.strain)

    # published for the HUD, verbatim
    pub = w.get_published()
    assert pub["anatomy"]["params"] == w.model.anatomy()["params"]
    assert set(pub["cap_gates"]) == set(g)

    # _maybe_bloom respects honest gates: nothing grows before its time
    before = w.model.anatomy()["params"]
    w._maybe_bloom()
    assert w.model.anatomy()["params"] == before
    w.stop()


def test_the_core_blooms_in_a_real_life():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3m2_bloom"
    cfg.bloom_cooldown_ticks = 1        # every gate opened by config,
    cfg.bloom_loss_floor = 1e-9         # not by faking any measurement
    cfg.bloom_lp_eps = 1e9
    cfg.bloom_widen_k = 64
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=72)

    for _ in range(cfg.window + 24):    # >=20 verdicts in the window
        world.step(0.15)
        w.step_once()
    ctx.drives.levels["energy"] = 1.0
    assert all(v >= 1.0 for v in w._cap_gates().values()), w._cap_gates()

    before = w.model.anatomy()
    pre_loss = w.get_published()["loss"]
    w._maybe_bloom()                    # the sleep venue calls exactly this
    after = w.model.anatomy()

    assert after["params"] > before["params"]
    assert w.bloom.blooms_total == 1
    # the surgery was recorded honestly, with true param counts
    blooms = [e for e in ctx.pulse.since(0, limit=500)
              if e["kind"] == "bloom"]
    assert len(blooms) == 1
    assert blooms[0]["meta"]["params_before"] == before["params"]
    assert blooms[0]["meta"]["params_after"] == after["params"]
    # a prebloom checkpoint exists: surgery is reversible
    assert (cfg.run_dir() / "ckpt_prebloom0.pt").exists()
    # the fringe re-attached to the (possibly new) edges
    assert len(w.fringe) == 2 * cfg.fringe_sprouts_per_site

    # life continues across the cut: same answers scale, still learning
    losses = []
    for _ in range(8):
        world.step(0.15)
        w.step_once()
        losses.append(w.get_published()["loss"])
    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < max(pre_loss * 5.0, 1e-3)   # no discontinuity blowup

    # cooldown honestly resets: it cannot bloom again this tick
    w.bloom.last_bloom_tick = w.tick_id
    cfg.bloom_cooldown_ticks = 10_000
    w._maybe_bloom()
    assert w.bloom.blooms_total == 1
    w.stop()


def test_bloom_disabled_means_absent():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3m2_off"
    cfg.bloom_enabled = False
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=73)
    for _ in range(cfg.window + 4):
        world.step(0.15)
        w.step_once()
    assert all(s == 0.0 for s in w.bloom.strain)     # gauges never ran
    assert len(w._cap_window) == 0
    w._maybe_bloom()
    assert w.bloom.blooms_total == 0
    w.stop()
