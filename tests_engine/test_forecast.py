"""Era 6-M4 — the ForecastLedger: imagined futures become accountable.
Stable ids, deadlines, resolution against reality, Brier/ECE, and a
calibration map that feeds back into how futures are ranked."""

import time

import torch

from magnum_opus_v2.forecast import ForecastLedger


def _future(vec, p, mode="world", name="x"):
    return {"vec": vec, "probability": p, "mode": mode, "name": name,
            "source": "test", "utility": 0.0}


def test_forecasts_resolve_against_reality():
    d = 32
    led = ForecastLedger(horizon_s=0.05, hit_cos=0.35)
    truth = torch.zeros(d); truth[0] = 1.0
    ortho = torch.zeros(d); ortho[1] = 1.0

    led.record([_future(truth.clone(), 0.9, name="right"),
                _future(ortho.clone(), 0.9, name="wrong")], tick=1)
    assert led.metrics()["open"] == 2

    time.sleep(0.06)                       # the horizon passes
    n = led.resolve(truth)
    assert n == 2
    m = led.metrics()
    assert m["resolved"] == 2 and m["open"] == 0
    statuses = {f["phrase"]: f["status"] for f in led.resolved}
    assert statuses["right"] == "hit" and statuses["wrong"] == "miss"
    assert "brier" in m and "ece" in m and m["hit_rate"] == 0.5


def test_calibration_earns_an_opinion_then_corrects():
    """Feed the ledger a systematically overconfident forecaster
    (p=0.9, hits 20% of the time). calibrated(0.9) must learn ~0.2,
    and ranking by calibrated probability must overturn the raw
    ranking — the ledger's consumer contract."""
    d = 16
    led = ForecastLedger(horizon_s=0.0, hit_cos=0.35, min_resolutions=30)
    truth = torch.zeros(d); truth[0] = 1.0
    ortho = torch.zeros(d); ortho[1] = 1.0

    assert led.calibrated(0.9, "world") is None    # no opinion yet

    for i in range(50):                    # overconfident: 20% hits
        vec = truth.clone() if i % 5 == 0 else ortho.clone()
        led.record([_future(vec, 0.9)], tick=i)
        led.resolve(truth)
    # honest: 80% hits at stated 0.3
    for i in range(50):
        vec = truth.clone() if i % 5 else ortho.clone()
        led.record([_future(vec, 0.3)], tick=100 + i)
        led.resolve(truth)

    c_hi = led.calibrated(0.9, "world")
    c_lo = led.calibrated(0.3, "world")
    assert c_hi is not None and c_lo is not None
    assert abs(c_hi - 0.2) < 0.1, f"calibrated(0.9)={c_hi}"
    assert abs(c_lo - 0.8) < 0.1, f"calibrated(0.3)={c_lo}"
    # the correction overturns the raw ranking (w_p * p, other terms =)
    assert c_lo > c_hi, "measured meaning of 'likely' did not win"

    m = led.metrics()
    assert m["brier"] > 0.3                # the overconfidence is VISIBLE
    assert m["ece"] > 0.2


def test_open_set_is_bounded_honestly():
    d = 8
    led = ForecastLedger(horizon_s=999.0, max_open=10)
    for i in range(25):
        led.record([_future(torch.randn(d), 0.5)], tick=i)
    m = led.metrics()
    assert m["open"] == 10
    assert m["expired"] == 15              # evictions counted, not hidden
