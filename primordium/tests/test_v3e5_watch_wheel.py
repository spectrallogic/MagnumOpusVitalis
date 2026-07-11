"""V3 Era 5 — the Watch (many cheap eyes, one deliberate look) and the
Wheel (prediction at the scale of moments). Both born silent, both
judged by mechanism: a cross-modal spike must release the fovea, and a
chapter change must register where tick-level surprise has moved on."""

import numpy as np
import torch

from primordium.config import OrganismConfig
from primordium.loom.watch import Watch
from primordium.loom.wheel import Wheel
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_watchers_spike_only_against_their_own_history():
    cfg = OrganismConfig()
    cfg.watch_min_n = 32
    cfg.watch_spike_z = 3.0
    rng = np.random.default_rng(5)

    # a newborn watcher stays silent, even for a wild value — it has no
    # history to defy yet (and that wild first value BECOMES history:
    # the first version of this test fed 99.0 then expected a spike at
    # 2.0, and the poisoned statistics rightly refused)
    w0 = Watch(cfg)
    for _ in range(10):
        assert w0.observe({"sound": 99.0 + rng.normal(0, 1)}) == []

    # a steady life, then the world jumps: exactly one hand goes up
    w = Watch(cfg)
    for _ in range(64):
        assert w.observe({"sound": 0.5 + rng.normal(0, 0.02)}) == []
    spikes = w.observe({"sound": 2.0})
    assert len(spikes) == 1
    assert spikes[0]["stream"] == "sound" and spikes[0]["z"] > 3.0
    assert w.snapshot()["spikes"] == 1

    # a dead-constant stream that jumps is a spike, not an overflow
    wc = Watch(cfg)
    for _ in range(40):
        assert wc.observe({"flow": 0.0}) == []
    sp = wc.observe({"flow": 1.0})
    assert len(sp) == 1 and abs(sp[0]["z"]) <= 999.0

    # history survives a nap
    w2 = Watch(cfg)
    w2.load_state(w.state())
    assert w2.observe({"sound": 2.0})[0]["stream"] == "sound"


def test_wheel_turns_learns_and_notices_chapter_changes():
    cfg = OrganismConfig()
    cfg.wheel_window_ticks = 4
    cfg.wheel_min_turns = 4
    cfg.wheel_spike_z = 2.0
    torch.manual_seed(3)
    wh = Wheel(cfg)
    assert wh.expectation_token() is None      # born empty

    d = cfg.d_model
    a = torch.randn(d)                         # chapter A, slightly noisy
    for i in range(4 * 30):
        wh.spin(a + torch.randn(d) * 0.05)
    assert wh.turns == 30
    assert wh.expectation_token() is not None
    assert wh.loss_ema is not None and wh.loss_ema < 0.1   # A is learnable

    b = torch.randn(d)                         # the chapter CHANGES
    spiked = False
    for i in range(4 * 3):
        r = wh.spin(b + torch.randn(d) * 0.05)
        if r is not None:
            spiked = spiked or r["spiked"]
    assert spiked, "a regime change did not register at the moments scale"

    wh2 = Wheel(cfg)                           # the wheel survives a nap
    wh2.load_state_dict(wh.state_dict())
    wh2.load_bank_state(wh.bank_state())
    assert wh2.turns == wh.turns
    assert torch.allclose(wh2.expectation_token(), wh.expectation_token())


def test_cross_modal_spike_releases_the_fovea():
    """The falsifiable claim of the Watch: while the eye stares at a
    corner, a burst on the WORDS stream must make it look up — and with
    the Watch disabled, the identical burst must not."""
    def life(watch_on):
        cfg = OrganismConfig()
        cfg.run_name = f"_test_v3e5_{int(watch_on)}"
        cfg.watch_enabled = watch_on
        cfg.watch_min_n = 24
        cfg.watch_spike_z = 3.0
        cfg.gaze_chase = 0.0               # the eye holds where we put it
        cfg.gaze_noise = 0.0
        ctx, board, flow = build(cfg)
        w = ctx.worker
        world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=171)
        for i in range(cfg.window + 40):   # quiet life; watchers learn it
            world.step(0.15)
            w.step_once()
        ctx.gaze.x, ctx.gaze.y, ctx.gaze.zoom = -0.8, -0.8, 0.5
        ctx.wordstream.push("SOMETHING LOUD ARRIVES IN THE SILENT ROOM",
                            source="human")
        for i in range(10):
            world.step(0.15)
            w.step_once()
        zoom, interrupts = ctx.gaze.zoom, ctx.gaze.interrupts
        w.stop()
        return zoom, interrupts

    zoom_on, intr_on = life(True)
    zoom_off, intr_off = life(False)
    print(f"watch on: zoom={zoom_on:.2f} interrupts={intr_on} | "
          f"off: zoom={zoom_off:.2f} interrupts={intr_off}")
    assert intr_on >= 1, "the words spike never released the fovea"
    assert zoom_on > 0.8, "the eye did not open back up"
    assert intr_off == 0 and zoom_off < 0.6, \
        "control failed: gaze moved without the Watch"


def test_organs_live_in_the_body_and_survive_naps():
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e5_body"
    cfg.wheel_window_ticks = 4
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=173)
    for _ in range(cfg.window + 20):
        world.step(0.15)
        w.step_once()

    pub = w.get_published()
    assert pub["watch"]["streams"] >= 9        # senses + needs at least
    assert pub["wheel"]["turns"] >= 4          # it has been turning
    assert np.isfinite(pub["loss"])            # expectation token is safe

    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert ctx2.worker.wheel.turns == w.wheel.turns
    assert ctx2.worker.watch.snapshot()["streams"] == \
        w.watch.snapshot()["streams"]
    w.stop()
    ctx2.worker.stop()
