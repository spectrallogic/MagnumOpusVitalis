"""V2-M3 — Lodestar (annealed frozen grounding) + Instincts (retirable
innate priors). The teacher is real (local Florence-2 DaViT); the
contract is that its influence is measured, annealed, and removable."""

import os

import numpy as np
import pytest
import torch

from primordium.config import OrganismConfig
from primordium.mind.coupling import (CompassRegion, InstinctRegion,
                                      LodestarRegion)
from primordium.mind.valence import ValenceCompass
from primordium.run import build
from primordium.senses import SyntheticWorld
from primordium.teachers.instincts import PROBES, all_probe_frames, \
    probe_frames


def test_probes_deterministic_and_distinct():
    a = probe_frames("face_like", 4, 96)
    b = probe_frames("face_like", 4, 96)
    assert np.array_equal(a, b)                    # same seed, same probes
    frames = all_probe_frames(96, 4)
    assert set(frames) == set(PROBES)
    means = {k: float(v.mean()) for k, v in frames.items()}
    assert means["darkness"] < means["looming"] < means["open_bright"]
    for k, v in frames.items():
        assert v.shape == (4, 96, 96, 3) and v.dtype == np.uint8


def test_compass_priors_and_retirement():
    c = ValenceCompass(dim=384, min_samples=10)
    v = torch.randn(384)
    c.install_prior("face_like", v, sig={"origin": "procedural probe"})
    c.install_prior("innate:looming", v)
    snap = c.snapshot()["installed"]
    assert {a["id"] for a in snap} == {"innate:face_like", "innate:looming"}
    assert all(a["provenance"] == "innate" for a in snap)
    assert c.lived_count() == 0
    # roundtrip keeps provenance
    c2 = ValenceCompass(dim=384)
    c2.load_state_dict(c.state_dict())
    assert c2.snapshot()["installed"][0]["provenance"] == "innate"
    gone = c2.retire_innate()
    assert len(gone) == 2 and c2.vectors() == {}


def test_lodestar_and_instincts_live():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m3"
    cfg.needle_every_s = 0.0                       # recompute immediately
    ctx, board, flow = build(cfg, teachers=True)
    teacher = ctx.vision_teacher
    assert teacher is not None and teacher.ok, teacher.load_error
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=41)
    w = ctx.worker

    # the teacher sees one frame and answers in bounded time
    world.step(0.15)
    frame, _ = ctx.retina.latest()
    feat = teacher.embed_frame(frame)              # warmup (cudnn, alloc)
    feat = teacher.embed_frame(frame)
    assert feat is not None and feat.shape == (cfg.lodestar_feat_dim,)
    assert torch.isfinite(feat).all()
    assert teacher.last_ms < 1000.0, f"teacher pass {teacher.last_ms:.0f}ms"

    lodestar = LodestarRegion(cfg, w, teacher, ctx.retina, ctx.sleep,
                              ctx.compass, board, pulse=ctx.pulse)
    instincts = InstinctRegion(cfg, w, ctx.compass, ctx.mood, board)
    compass_r = CompassRegion(ctx.compass, ctx.mood, ctx.subc, ctx.bus,
                              board, cfg=cfg, worker=w, pulse=ctx.pulse)

    # a few lived moments, then the star feeds the scaffold
    for _ in range(6):
        world.step(0.15)
        w.step_once()
    w.tick_id = 505                    # past the L_birth freeze point
    lodestar.step(ctx.bus, ctx.neuromod, 1.0)      # feat + needles job
    w._drain_jobs(5.0)                 # the loop thread would do this
    for _ in range(4):
        world.step(0.15)
        w.step_once()

    pub = w.get_published()
    assert pub.get("w_distill", 0) > 0             # annealed term is live
    kinds = {e["kind"] for e in ctx.pulse.since(0, limit=500)}
    assert "distill" in kinds and "needles" in kinds

    innate = {n for n in ctx.compass.vectors() if n.startswith("innate:")}
    assert innate == {f"innate:{k}" for k in PROBES}
    for e in ctx.compass.snapshot()["installed"]:
        if e["id"].startswith("innate:"):
            assert e["provenance"] == "innate"

    # the compass installs them into the mood field at reduced weight...
    compass_r.step(ctx.bus, ctx.neuromod, 1.0)
    # ...and the instincts feel motion along them
    instincts.step(ctx.bus, ctx.neuromod, 0.2)
    world.step(0.15); w.step_once()
    instincts.step(ctx.bus, ctx.neuromod, 0.2)
    acts = board.get("affect_acts", {})
    assert any(k.startswith("innate:") for k in acts)

    # retirement: stage 2 removes every inherited needle, measurably
    w.stage = 2
    compass_r.step(ctx.bus, ctx.neuromod, 1.0)
    assert not any(n.startswith("innate:") for n in ctx.compass.vectors())
    assert board.get("instincts_retired") is True
    lodestar.step(ctx.bus, ctx.neuromod, 1.0)      # and the star lets go
    assert not teacher.ok
    kinds = {e["kind"] for e in ctx.pulse.since(0, limit=500)}
    assert "instincts_retired" in kinds and "lodestar_released" in kinds
    w.stop()


def test_word_grounding_path():
    from primordium.teachers.lodestar import TextTeacher
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m3_word"
    tt = TextTeacher(cfg)
    assert tt.ok, tt.load_error
    e = tt.embed_text("hello little one")
    assert e.shape == (384,) and abs(float(np.linalg.norm(e)) - 1.0) < 1e-3

    ctx, board, flow = build(cfg)                  # no vision teacher needed
    ctx.word_teacher = tt
    from primordium.server.app import OrganismServer
    server = OrganismServer(cfg, ctx, board)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=43)
    for _ in range(cfg.window + 3):                # let the ring warm up
        world.step(0.15)
        ctx.worker.step_once()
    server.handle_chat("hello little one")         # now the words arrive
    world.step(0.15)
    ctx.worker.step_once()
    parts = ctx.worker.get_published().get("loss_parts", {})
    assert "w_word" in parts and parts["w_word"] > 0
    ctx.worker.stop()


@pytest.mark.skipif(not os.environ.get("PRIMORDIUM_LONG"),
                    reason="falsifiability soak: set PRIMORDIUM_LONG=1")
def test_lodestar_dissolves_falsifiably():
    """The dissolvability contract: once annealed (r>0.6), turning the
    teacher OFF must change prediction loss by <5% over a womb run."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m3_long"
    cfg.apply_dev_fast()
    ctx, board, flow = build(cfg, teachers=True)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=47)
    w = ctx.worker
    lodestar = LodestarRegion(cfg, w, ctx.vision_teacher, ctx.retina,
                              ctx.sleep, ctx.compass, board, pulse=ctx.pulse)
    losses_on = []
    for i in range(2400):
        world.step(0.15)
        if i % 20 == 0:
            lodestar.step(ctx.bus, ctx.neuromod, 1.0)
        w.step_once()
        if i > 2200:
            losses_on.append(w.get_published().get("loss") or 0.0)
    ema, birth = w.objective.ema_slow, w._L_birth
    r = 0.0 if not birth else max(0.0, 1.0 - ema / birth)
    ctx.vision_teacher.unload()                    # the star goes dark
    w._teacher_feat = None
    losses_off = []
    for i in range(600):
        world.step(0.15)
        w.step_once()
        if i > 400:
            losses_off.append(w.get_published().get("loss") or 0.0)
    on, off = float(np.mean(losses_on)), float(np.mean(losses_off))
    print(f"r={r:.2f} pred on={on:.4f} off={off:.4f}")
    if r > 0.6:
        assert abs(off - on) / max(on, 1e-6) < 0.05
    w.stop()
