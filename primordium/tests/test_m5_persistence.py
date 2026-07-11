"""M5 — a restart is a nap, not a death; and the calibration record holds."""

import json

import torch

from primordium.config import OrganismConfig
from primordium.persistence import calib
from primordium.persistence.checkpoint import save_checkpoint, load_checkpoint
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_checkpoint_roundtrip(tmp_path):
    cfg = OrganismConfig()
    cfg.run_name = "_test_m5"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=9)
    w = ctx.worker
    for _ in range(30):
        world.step(0.15)
        w.step_once()
    ctx.self_model.felt_time = 123.4
    ctx.imprint.imprint(torch.randn(cfg.d_model))
    path = save_checkpoint(ctx)
    tick_before = w.tick_id

    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert ctx2.worker.tick_id == tick_before
    assert abs(ctx2.self_model.felt_time - 123.4) < 1e-3
    assert ctx2.imprint.count() == 1
    # weights actually round-trip
    a = ctx.worker.model.state_dict()["blocks.0.mlp.0.weight"]
    b = ctx2.worker.model.state_dict()["blocks.0.mlp.0.weight"]
    assert torch.allclose(a.cpu(), b.cpu())
    ctx.worker.stop()
    ctx2.worker.stop()


def test_calib_seal_and_open():
    rec = calib._seal("hunter2", b"the sealed record survives the tensor path")
    out = calib._open("hunter2", rec)
    assert out == b"the sealed record survives the tensor path"
    assert calib._open("wrong", rec) is None
    # survives the tensor embedding path
    t = torch.tensor(list(json.dumps(rec, sort_keys=True).encode()),
                     dtype=torch.uint8)
    rec2 = calib.tensor_record(t)
    assert calib._open("hunter2", rec2) is not None
