"""V3 Era 3 — the Gaze: looking is an action. The eye chases measured
prediction error, its state is felt (proprioception), staring that
teaches nothing is released, and the whole loop is visible end-to-end:
plant chaos in one corner of the world and the eye goes there."""

import numpy as np
import pytest

from primordium.body.gaze import Gaze
from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import LightPort, SyntheticWorld


def test_crop_is_identity_at_full_zoom_and_honest_at_corner():
    port = LightPort(96)
    frame = np.random.default_rng(3).integers(
        0, 255, (96, 96, 3), dtype=np.uint8)
    port.offer_array(frame)

    whole = port.stage_view(96, 3, crop=None)
    full = port.stage_view(96, 3, crop=(0.3, -0.7, 1.0))
    assert np.array_equal(whole, full)          # zoom 1 = the whole scene

    corner = port.stage_view(48, 3, crop=(-1.0, -1.0, 0.5))
    expect = frame[:48, :48]                     # top-left 48px window
    assert np.array_equal(corner, expect)


def test_reflex_chases_error_and_boredom_releases():
    cfg = OrganismConfig()
    cfg.gaze_noise = 0.0                         # deterministic reflex
    cfg.gaze_release_hold_s = 100.0              # inspect the released state
    np.random.seed(0)
    gz = Gaze(cfg)
    hot = np.zeros((4, 4), dtype=np.float32)
    hot[0, 3] = 1.0                              # top-right defies it
    for _ in range(40):
        gz.update(hot, lp=0.05, dt=0.15)         # learning: keep chasing
    assert gz.x > 0.5 and gz.y < -0.5            # it went there
    assert gz.zoom < 0.9                         # and leaned in
    assert gz.saccades > 0

    # the noisy-TV valve: same stare, but nothing is being learned
    for _ in range(200):
        gz.update(hot, lp=0.0, dt=0.15)
    assert gz.releases >= 1                      # it let go
    assert gz.zoom > 0.9                         # back toward the whole
    assert abs(gz.x) < 0.2 and abs(gz.y) < 0.2   # and stayed wide (hold)


def test_the_eye_is_felt_and_survives_naps():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e3"
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=131)
    for _ in range(cfg.window + 8):
        world.step(0.15)
        w.step_once()

    # efference: the last three proprio dims ARE the gaze, verbatim
    from primordium.loom.tokenizer import SensoryTokenizer
    assert SensoryTokenizer.PROPRIO_DIM == 26
    ep = ctx.chronicle.tail(1)[0]
    assert ep.motor.shape[0] == 26
    # the episode holds the gaze of ITS tick; the eye has drifted one
    # reflex step since — close, and honestly not identical
    assert np.allclose(ep.motor[-3:], ctx.gaze.efference(), atol=0.1)
    assert 0.4 <= float(ep.motor[-1]) <= 1.01    # zoom, in its real range

    # where it looks changes what it sees
    ctx.gaze.x, ctx.gaze.y, ctx.gaze.zoom = -0.9, -0.9, 0.5
    a = ctx.retina.stage_view(8, 1, crop=ctx.gaze.crop_params())
    ctx.gaze.x, ctx.gaze.y = 0.9, 0.9
    b = ctx.retina.stage_view(8, 1, crop=ctx.gaze.crop_params())
    assert not np.array_equal(a, b)

    assert "zoom" in w.get_published()["gaze"]
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    ctx.gaze.saccades = 7
    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert ctx2.gaze.saccades == 7
    assert abs(ctx2.gaze.x - ctx.gaze.x) < 1e-6

    # earlier bodies are refused by name
    import torch as _t
    p3 = cfg.run_dir() / "old_v3.pt"
    _t.save({"version": 3, "tick_id": 1, "stage": 0}, p3)
    with pytest.raises(RuntimeError, match="pre-Gaze v3 life"):
        load_checkpoint(ctx2, p3)
    p3.unlink(missing_ok=True)
    w.stop()
    ctx2.worker.stop()


def test_it_looks_at_the_part_of_the_world_that_defies_it():
    """End to end, no shortcuts: a static scene with an unpredictable
    square in the top-left. The per-patch error map is hot there, so
    the gaze must go there — through the real learn loop.

    The chaos flickers as a WHOLE block: per-pixel noise would simply
    average away at newborn 8x8 acuity (the eye correctly ignores what
    it cannot resolve — we learned that from this test's first run)."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e3_chaos"
    cfg.gaze_noise = 0.01
    ctx, board, flow = build(cfg)
    w = ctx.worker
    rng = np.random.default_rng(9)

    yy, xx = np.mgrid[0:96, 0:96]
    base = np.stack([np.full_like(xx, 60), np.full_like(xx, 120),
                     np.full_like(xx, 90)], axis=-1).astype(np.uint8)
    for i in range(110):
        frame = base.copy()
        frame[:32, :32] = rng.integers(0, 255, (3,))   # the chaos beacon
        ctx.retina.offer_array(frame)
        w.step_once()

    gz = ctx.gaze
    assert gz.saccades > 0, "the eye never moved"
    assert gz.x < -0.1 and gz.y < -0.1, \
        f"gaze ended at ({gz.x:.2f}, {gz.y:.2f}), not in the chaos corner"
    kinds = {e["kind"] for e in ctx.pulse.since(0, limit=500)}
    assert "gaze_shift" in kinds
    w.stop()
