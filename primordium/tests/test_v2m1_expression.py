"""V2-M1 — the expression organs: layout v2, Wordstream, Easel, heads."""

import numpy as np
import torch

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_group_layout_v2():
    cfg = OrganismConfig()
    from primordium.loom.tokenizer import SensoryTokenizer
    for stage, expect in ((0, 37), (1, 37), (2, 85)):
        tok = SensoryTokenizer(cfg, stage=stage)
        assert tok.group_size == expect, (stage, tok.group_size)
        s = tok.sensory_slice
        n_sens = (cfg.audio_tokens + tok.n_vis
                  + cfg.text_tokens + tok.n_cnv)
        assert s.stop - s.start == n_sens
    assert cfg.window * 85 <= cfg.max_seq   # seq-length cliff guarded


def test_wordstream_and_easel_flow():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m1"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=21)
    w = ctx.worker

    # a caregiver-shaped message enters the stream
    ctx.wordstream.push("hello little one", source="human")
    # urge gates open: it may type and paint
    ctx.router.impulse()

    for _ in range(30):
        world.step(0.15)
        w.step_once()

    pub = w.get_published()
    assert pub.get("loss") is not None
    ws = pub.get("wordstream", {})
    assert ws.get("messages_in", 0) >= 1
    # the caregiver message was consumed; only fresh self-echoes may remain
    assert ws.get("pending", 99) < 16

    # its keystrokes echo into its own stream (self-hearing for text)
    assert ctx.wordstream.chars_typed_total > 0
    # painting mutated the framebuffer away from the birth color
    fb = ctx.easel.view()
    assert ctx.easel.strokes_total > 0
    assert float(np.abs(fb.astype(int) - 16).mean()) > 0.5

    # pulse recorded real events
    events = ctx.pulse.since(0, limit=200)
    kinds = {e["kind"] for e in events}
    assert "tick" in kinds
    assert "babble_out" in kinds or "paint" in kinds
    w.stop()


def test_checkpoint_v2_roundtrip_with_expression():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m1_ck"
    from primordium.persistence.checkpoint import save_checkpoint, load_checkpoint
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=22)
    ctx.wordstream.push("hi", source="human")
    ctx.router.impulse()
    for _ in range(15):
        world.step(0.15)
        ctx.worker.step_once()
    strokes = ctx.easel.strokes_total
    path = save_checkpoint(ctx)

    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert ctx2.worker.tick_id == ctx.worker.tick_id
    assert ctx2.easel.strokes_total == strokes
    a = ctx.worker.keys_head.state_dict()["0.weight"]
    b = ctx2.worker.keys_head.state_dict()["0.weight"]
    assert torch.allclose(a.cpu(), b.cpu())
    ctx.worker.stop()
    ctx2.worker.stop()
