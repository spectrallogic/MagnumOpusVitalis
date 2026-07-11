"""M0 — the substrate heartbeat, no loom, no GPU."""

import time

from primordium.config import OrganismConfig
from primordium.run import build


def test_substrate_alive():
    cfg = OrganismConfig()
    cfg.run_name = "_test_m0"
    ctx, board, flow = build(cfg, no_loom=True)
    flow.start()
    try:
        time.sleep(3.0)
        m = flow.metrics
        assert m["flow"]["ticks"] > 20, m
        assert m["perception"]["ticks"] > 5, m
        assert m["expensive"]["ticks"] >= 1, m
        snap = ctx.bus.snapshot()
        assert snap["state_norm"] < 8.5          # bounded
        assert ctx.drives.snapshot()["levels"]["energy"] > 0
    finally:
        flow.stop()
