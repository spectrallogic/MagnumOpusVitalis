"""V2-M5 — a whole (short) life with every organ on, and the honest
refusal of v1 bodies. The 30-minute soak is env-gated: PRIMORDIUM_LONG."""

import os
import time

import numpy as np
import pytest
import torch

from primordium.config import OrganismConfig
from primordium.persistence.checkpoint import load_checkpoint
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_v1_life_is_refused(tmp_path):
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m5_v1"
    ctx, board, flow = build(cfg, no_loom=True)
    p = tmp_path / "ckpt_latest.pt"
    torch.save({"version": 1, "tick_id": 12345, "stage": 2}, p)
    with pytest.raises(RuntimeError, match="v1 life"):
        load_checkpoint(ctx, p)


def test_full_life_all_organs_on():
    """~30 real seconds of actual life: Loom thread + substrate clocks +
    womb + stub caregiver, all concurrent, nothing mocked inside the
    organism. Asserts the realtime invariant (Hz), learning, expression,
    caregiving, and a clean pulse."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m5_life"
    cfg.caregiver = True
    cfg.caregiver_stub = True
    cfg.caregiver_reply_min_s = 3.0
    cfg.caregiver_checkin_s = 8.0
    (cfg.run_dir() / "gates.json").unlink(missing_ok=True)
    ctx, board, flow = build(cfg)
    # the caregiver gate stands open for this life (ceremony tested in M4)
    ctx.gatehouse.load_state_dict(
        {"gates": {"caregiver": {"state": "unlocked"}}})
    world = SyntheticWorld(ctx.retina, ctx.cochlea)

    import threading
    stop = threading.Event()

    def womb():
        while not stop.is_set():
            world.step(0.15)
            time.sleep(0.15)

    t = threading.Thread(target=womb, daemon=True)
    try:
        t.start()
        flow.start()
        ctx.worker.start()
        deadline = time.monotonic() + 30.0
        nudged = False
        while time.monotonic() < deadline:
            time.sleep(1.0)
            pub = ctx.worker.get_published()
            # 10s in, fire the urge the Executive would build up to over
            # a longer life — same code path, just on the test's clock
            if not nudged and time.monotonic() > deadline - 20.0:
                nudged = True
                ctx.router.impulse()
        pub = ctx.worker.get_published()

        # lived rate: an unfed newborn paces itself down BY DESIGN
        # (tick_ms 150 -> tick_ms_max 400 as energy falls); the floor is
        # 2.5 Hz — anything below that means compute, not tiredness
        assert pub.get("hz", 0) >= 2.5, f"hz={pub.get('hz')}"
        assert pub.get("loss") is not None and np.isfinite(pub["loss"])
        assert "job_error" not in pub or not pub["job_error"]

        # it expressed something, unprompted or answered
        ws = pub.get("wordstream", {})
        assert ws.get("chars_typed", 0) > 0 or ctx.easel.strokes_total > 0

        # the caregiver actually spoke into its stream
        assert any(m["source"] == "foster"
                   for m in ctx.wordstream.snapshot()["transcript"]), \
            "no caregiver word arrived in 30s of open-gate life"

        kinds = {e["kind"] for e in ctx.pulse.since(0, limit=500)}
        assert "tick" in kinds and "caregiver_msg" in kinds

        # compute capability with EVERY organ wired: >=5 Hz (plan bar).
        # Measured directly, unpaced, after the live threads stand down.
        stop.set()
        ctx.worker.stop()
        ctx.worker.join(timeout=10)
        t0 = time.monotonic()
        n = 15
        for _ in range(n):
            world.step(0.15)
            ctx.worker.step_once()
        per = (time.monotonic() - t0) / n
        assert per < 0.2, f"step takes {per*1000:.0f}ms — under 5 Hz"
    finally:
        stop.set()
        ctx.worker.stop()
        flow.stop()
        fr = getattr(ctx, "foster", None)
        if fr is not None and getattr(fr, "foster", None) is not None:
            fr.foster.stop()


@pytest.mark.skipif(not os.environ.get("PRIMORDIUM_LONG"),
                    reason="30-min all-on soak: set PRIMORDIUM_LONG=1")
def test_thirty_minute_life():
    from primordium.persistence.checkpoint import save_checkpoint
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m5_soak"
    cfg.apply_dev_fast()
    cfg.caregiver = True
    cfg.caregiver_stub = True
    (cfg.run_dir() / "gates.json").unlink(missing_ok=True)
    ctx, board, flow = build(cfg, teachers=True)
    ctx.gatehouse.load_state_dict(
        {"gates": {"caregiver": {"state": "unlocked"}}})
    world = SyntheticWorld(ctx.retina, ctx.cochlea)
    import threading
    stop = threading.Event()

    def womb():
        while not stop.is_set():
            world.step(0.15)
            time.sleep(0.15)

    t = threading.Thread(target=womb, daemon=True)
    hz_samples = []
    try:
        t.start()
        flow.start()
        ctx.worker.start()
        deadline = time.monotonic() + 1800.0
        while time.monotonic() < deadline:
            time.sleep(10.0)
            pub = ctx.worker.get_published()
            if pub.get("hz"):
                hz_samples.append(pub["hz"])
        pub = ctx.worker.get_published()
        assert float(np.mean(hz_samples[3:])) >= 5.0
        assert pub.get("stage", 0) >= 1              # it grew (dev-fast)
        assert len(ctx.compass.vectors()) >= 1       # it felt something
        path = save_checkpoint(ctx)
        assert path.exists()
    finally:
        stop.set()
        ctx.worker.stop()
        flow.stop()
        fr = getattr(ctx, "foster", None)
        if fr is not None and getattr(fr, "foster", None) is not None:
            fr.foster.stop()
