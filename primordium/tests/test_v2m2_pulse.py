"""V2-M2 — Pulse feed + server protocol: honest blocks, chat, canvas."""

import numpy as np

from primordium.config import OrganismConfig
from primordium.events.pulse import Pulse, ZONES
from primordium.run import build
from primordium.senses import SyntheticWorld
from primordium.server.app import OrganismServer, _canvas_png


def test_pulse_ring_semantics():
    p = Pulse(capacity=8)
    for i in range(12):
        p.emit("tick", "SENSE", "CORE", loss=float(i))
    evs = p.since(0, limit=64)
    assert len(evs) == 8                      # ring capacity honoured
    assert evs[-1]["id"] == 12                # ids keep counting
    assert p.since(evs[-1]["id"]) == []       # drained means drained
    assert all(e["zone_from"] in ZONES and e["zone_to"] in ZONES
               for e in evs)


def test_live_pulse_and_server_payload():
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m2"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=31)
    server = OrganismServer(cfg, ctx, board)

    # chat uplink: text enters the Wordstream, Pulse records the arrival
    assert server.handle_chat("  hello there  ")
    assert not server.handle_chat("   ")          # whitespace is not speech
    assert ctx.wordstream.n_messages == 1

    for _ in range(12):
        world.step(0.15)
        ctx.worker.step_once()

    # every rendered block is a computation that happened
    evs = ctx.pulse.since(0, limit=500)
    kinds = {e["kind"] for e in evs}
    assert "tick" in kinds and "chat_in" in kinds
    ticks = [e for e in evs if e["kind"] == "tick"]
    assert all("loss" in t["meta"] and "shape" in t["meta"] for t in ticks)

    # the tooltip and the HUD tell the same story: last tick's loss is
    # exactly the published world-prediction loss
    payload = server.state_payload()
    assert payload["loss"] == ticks[-1]["meta"]["loss"]
    assert payload["wordstream"]["messages_in"] == 1
    assert "strokes" in payload["easel"]
    # dormant organs are visible and shut from birth (Gatehouse, M4)
    assert set(payload["gates"]) == {"caregiver", "webtext"}
    assert all(g["state"] == "locked" for g in payload["gates"].values())

    ctx.worker.stop()


def test_canvas_png_roundtrip():
    import cv2
    fb = np.zeros((32, 32, 3), dtype=np.uint8)
    fb[4:9, 20:26] = (250, 40, 120)                # an RGB blot
    png = _canvas_png(fb)
    assert png[:4] == b"\x89PNG"
    back = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB)
    assert np.array_equal(back, fb)                # lossless, channels kept
