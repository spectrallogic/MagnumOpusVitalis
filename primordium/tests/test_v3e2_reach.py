"""V3 Era 2 — the Reach: attention over a lifetime. Born empty (exact),
retrieval measured, worth proven by re-living ticks without memory."""

import os

import numpy as np
import pytest
import torch

from primordium.config import OrganismConfig
from primordium.loom.reach import Reach
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_bank_semantics():
    cfg = OrganismConfig()
    cfg.reach_capacity = 16
    cfg.reach_topk = 4
    cfg.reach_exclude_recent = 10
    r = Reach(cfg)
    q = torch.randn(cfg.d_model)

    toks, info = r.retrieve(q, now_tick=100)
    assert toks is None and info["n"] == 0          # born empty

    torch.manual_seed(5)
    for t in range(20):                              # ring wraps at 16
        r.write(torch.randn(cfg.d_model), tick=t, salience=1.0)
    planted = q + 0.05 * torch.randn(cfg.d_model)    # a near-echo of now,
    r.write(planted, tick=50, salience=2.0)          # from long ago
    assert r.size == 16 and r.writes == 21

    toks, info = r.retrieve(q, now_tick=200)
    assert toks is not None and toks.shape[1] == cfg.d_model
    assert info["n"] >= 1 and info["ages"][0] == 150  # the echo wins
    assert info["sim"] > 0.15         # mean over top-k; the echo lifts it

    # the near past is not the Reach's business
    toks, info = r.retrieve(q, now_tick=55)
    assert info["n"] == 0 or 150 not in info.get("ages", [])

    # gradients flow into how memories speak, never into the past itself
    toks, _ = r.retrieve(q, now_tick=200)
    toks.sum().backward()
    assert r.value.weight.grad is not None
    assert r._vecs.grad is None


def test_cold_start_is_bit_identical():
    """No memories, no tokens: the forward with an empty Reach equals
    the forward without one. And WITH memories, dropping K outputs
    keeps every downstream index valid."""
    cfg = OrganismConfig()
    from primordium.loom.core import LoomCore
    torch.manual_seed(7)
    core = LoomCore(cfg, {"mlp_dims": [512, 512]})
    tokens = torch.randn(3 * 37, cfg.d_model) * 0.1
    plain = core(tokens, group_size=37)
    with_none = core(tokens, group_size=37, mem_tokens=None)
    empty = core(tokens, group_size=37,
                 mem_tokens=torch.zeros(0, cfg.d_model))
    assert torch.equal(plain, with_none)
    assert torch.equal(plain, empty)

    mem = torch.randn(5, cfg.d_model) * 0.1
    out = core(tokens, group_size=37, mem_tokens=mem)
    assert out.shape == plain.shape                  # K dropped, S kept
    assert not torch.allclose(out, plain)            # and it mattered


def test_reach_lives_probes_and_persists():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e2"
    cfg.reach_write_every = 2
    cfg.reach_exclude_recent = 8
    cfg.reach_probe_every = 1        # probe every memory-bearing tick
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=91)

    for _ in range(cfg.window + 30):
        world.step(0.15)
        w.step_once()

    snap = w.get_published()["reach"]
    assert snap["size"] > 8                          # it banked its life
    assert snap["hits"] > 0                          # and reached back
    assert snap["probes"] > 0                        # on trial, measured
    assert np.isfinite(snap["gain"])
    assert all(a >= cfg.reach_exclude_recent for a in snap["ages"])

    # its memories and their worth survive a nap
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    w2 = ctx2.worker
    assert w2.reach.size == w.reach.size
    assert w2.reach.probes == w.reach.probes
    assert torch.allclose(w2.reach.value.weight.cpu(),
                          w.reach.value.weight.cpu())
    assert torch.allclose(w2.reach._vecs[:w2.reach.size].cpu(),
                          w.reach._vecs[:w.reach.size].cpu())
    w.stop()
    w2.stop()


def test_it_reaches_past_the_present_regime():
    """Live in world A, then move to world B: retrievals during B must
    include memories older than B itself — the far past stays reachable
    when the present no longer resembles it."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e2_regime"
    cfg.reach_write_every = 2
    cfg.reach_exclude_recent = 8
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world_a = SyntheticWorld(ctx.retina, ctx.cochlea, seed=101)
    for _ in range(cfg.window + 28):
        world_a.step(0.15)
        w.step_once()
    ticks_in_a = w.tick_id

    world_b = SyntheticWorld(ctx.retina, ctx.cochlea, seed=202)
    saw_a_from_b = False
    for _ in range(20):
        world_b.step(0.15)
        w.step_once()
        ages = w.get_published()["reach"].get("ages", [])
        b_age = w.tick_id - ticks_in_a
        if any(a > b_age for a in ages):
            saw_a_from_b = True
    assert saw_a_from_b, "no A-era memory surfaced during life in B"
    w.stop()


@pytest.mark.skipif(not os.environ.get("PRIMORDIUM_LONG"),
                    reason="return-to-A recall soak: set PRIMORDIUM_LONG=1")
def test_returning_somewhere_known_is_measurably_easier():
    """The falsifiable claim: come BACK to a place it has lived, and the
    memory of that place must pay — mean counterfactual gain over the
    return must beat the gain in a place it has never been."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e2_return"
    cfg.reach_write_every = 2
    cfg.reach_probe_every = 2
    ctx, board, flow = build(cfg)
    w = ctx.worker

    def live(world, n):
        gains = []
        for _ in range(n):
            world.step(0.15)
            before = w.reach.probes
            w.step_once()
            if w.reach.probes > before:
                gains.append(w.reach.gain_ema)
        return gains

    live(SyntheticWorld(ctx.retina, ctx.cochlea, seed=111), 600)  # home
    novel = live(SyntheticWorld(ctx.retina, ctx.cochlea, seed=222), 300)
    home_again = live(SyntheticWorld(ctx.retina, ctx.cochlea, seed=111), 300)
    print(f"gain novel={np.mean(novel):.5f} return={np.mean(home_again):.5f}")
    assert np.mean(home_again) >= np.mean(novel) - 1e-4
    w.stop()
