"""Engine persistence — a restart is a nap, not a death.

Model-free: the codecs and the forecast ledger's serialization are
exercised directly on region objects, and monotonic-clock rebasing is
asserted to keep continuity without making a woken engine blurt."""

import time

import torch

from magnum_opus_v2 import persistence as P
from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.config import BusConfig
from magnum_opus_v2.neuromod import NeuromodState
from magnum_opus_v2.forecast import ForecastLedger
from magnum_opus_v2.regions.memory import Memory
from magnum_opus_v2.regions.abstraction import AbstractionLadder
from magnum_opus_v2.regions.self_model import SelfModel
from magnum_opus_v2.regions.situation import SituationModel
from magnum_opus_v2.regions.subconscious import Candidate

D = 32


def _bus():
    return LatentBus(hidden_dim=D, device="cpu", config=BusConfig())


def test_bus_and_neuromod_roundtrip():
    b = _bus()
    b.state = torch.randn(D)
    b.velocity = torch.randn(D)
    b.tick_count = 4321
    b.temperature = 0.7
    blk = P.encode_bus(b)

    b2 = _bus()
    P.apply_bus(b2, blk)
    assert torch.allclose(b2.state, b.state)
    assert torch.allclose(b2.velocity, b.velocity)
    assert b2.tick_count == 4321
    assert abs(b2.temperature - 0.7) < 1e-6

    nm = NeuromodState()
    nm.stress, nm.reward, nm.calm, nm.arousal = 1.3, 0.4, 0.9, 0.2
    nm2 = NeuromodState()
    P.apply_neuromod(nm2, P.encode_neuromod(nm))
    assert (nm2.stress, nm2.reward, nm2.calm, nm2.arousal) == (1.3, 0.4, 0.9, 0.2)


def test_memory_pool_roundtrip_keeps_false_memories():
    mem = Memory(device="cpu")
    mem.pool.append(Candidate(vec=torch.randn(D), source="memory",
                              confidence=1.0, meta={"tag": "heard:hi"}))
    mem.pool.append(Candidate(vec=torch.randn(D), source="confab",
                              confidence=0.6, meta={"tag": "confabulated"}))
    blk = P.encode_memory_pool(mem)

    mem2 = Memory(device="cpu")
    P.apply_memory_pool(mem2, blk)
    assert len(mem2.pool) == 2
    assert torch.allclose(mem2.pool[0].vec, mem.pool[0].vec)
    # the false memory survives with its sub-1.0 confidence and tag
    false = [c for c in mem2.pool if c.confidence < 1.0]
    assert len(false) == 1 and false[0].meta["tag"] == "confabulated"


def test_abstraction_and_situation_roundtrip():
    ab = AbstractionLadder(hidden_dim=D, device="cpu")
    ab.observations = 200
    ab.levels[0].centroids = [torch.randn(D), torch.randn(D)]
    ab.levels[0].counts = [5, 3]
    ab.levels[0].updates = 8
    ab2 = AbstractionLadder(hidden_dim=D, device="cpu")
    P.apply_abstraction(ab2, P.encode_abstraction(ab))
    assert ab2.observations == 200
    assert ab2.levels[0].counts == [5, 3]
    assert torch.allclose(ab2.levels[0].centroids[0], ab.levels[0].centroids[0])

    sit = SituationModel(device="cpu")
    sit.vec = torch.randn(D)
    sit.narrative = "The user is driving at night."
    sit.shift_count = 2
    sit2 = SituationModel(device="cpu")
    P.apply_situation(sit2, P.encode_situation(sit))
    assert torch.allclose(sit2.vec, sit.vec)
    assert sit2.narrative == "The user is driving at night."
    assert sit2.shift_count == 2


def test_self_model_felt_time_persists():
    sm = SelfModel(device="cpu")
    sm.identity = torch.randn(D)
    sm.felt_time = 123.4
    sm.continuity = 0.8
    sm2 = SelfModel(device="cpu")
    P.apply_self_model(sm2, P.encode_self_model(sm))
    assert abs(sm2.felt_time - 123.4) < 1e-4
    assert torch.allclose(sm2.identity, sm.identity)


def test_forecast_ledger_nap_keeps_earned_calibration():
    led = ForecastLedger(min_resolutions=30)
    # earn an opinion: 40 resolved world forecasts, systematically over-
    # confident (stated 0.9, hit only ~20%)
    for i in range(40):
        led._bins.setdefault("world", [[0, 0] for _ in range(10)])
        led._bins["world"][9][0] += 1
        led._bins["world"][9][1] += 1 if i % 5 == 0 else 0
        led.resolved.append({"mode": "world", "status":
                             "hit" if i % 5 == 0 else "miss",
                             "probability": 0.9})
    before = led.calibrated(0.9, "world")
    assert before is not None and before < 0.4     # measured, not the raw 0.9

    led2 = ForecastLedger(min_resolutions=30)
    led2.load_state_dict(led.state_dict())
    after = led2.calibrated(0.9, "world")
    assert after is not None
    assert abs(after - before) < 1e-9              # the opinion survived


def test_forecast_open_deadlines_rebased_not_in_the_past():
    led = ForecastLedger(horizon_s=45.0)
    led.record([{"vec": torch.randn(D), "mode": "world",
                 "name": "x", "probability": 0.5, "utility": 0.1}], tick=1)
    st = led.state_dict()
    assert st["open"] and st["open"][0]["remaining_s"] > 0
    led2 = ForecastLedger(horizon_s=45.0)
    led2.load_state_dict(st)
    # the restored deadline is in the FUTURE, so it won't instantly resolve
    assert led2.open and led2.open[0]["deadline"] > time.monotonic()


def test_rebase_monotonic_policy():
    now = time.monotonic()
    # wall-start clocks pin to now (felt-time itself persists separately)
    assert P._rebase_monotonic(999.0, now, reset=True) == now
    # cooldown clocks become now - age, capped, never in the future
    r = P._rebase_monotonic(30.0, now)
    assert now - 31 < r <= now
    # an ancient age is capped, not runaway
    r2 = P._rebase_monotonic(10 ** 9, now)
    assert r2 <= now and (now - r2) <= P._AGE_CAP_S + 1
    # a None age (never happened) yields 0.0 (falsy "never")
    assert P._rebase_monotonic(None, now) == 0.0
