"""M2 — the body: deprivation raises pressure, sleep consolidates,
stimulus wakes."""

from primordium.config import OrganismConfig
from primordium.body.sleep import SleepController
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_sleep_pressure_cycle():
    cfg = OrganismConfig()
    sc = SleepController(cfg)
    # exhausted, alone, plateaued -> pressure climbs until sleep
    for _ in range(40):
        sc.update(energy=0.05, plateau=1.0, presence=0.0, surprise=1.0)
        if sc.asleep:
            break
    assert sc.asleep, sc.snapshot()
    # a loud strange thing happens -> it wakes, groggy
    sc.update(energy=0.5, plateau=0.0, presence=0.0, surprise=4.0)
    assert not sc.asleep
    assert sc.snapshot()["groggy"]


def test_replay_trains_from_lived_experience():
    cfg = OrganismConfig()
    cfg.run_name = "_test_m2"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=5)
    w = ctx.worker
    for _ in range(40):
        world.step(0.15)
        w.step_once()
    assert len(ctx.chronicle) >= 30
    w._job_replay()  # noqa: SLF001 — direct exercise
    assert w.get_published().get("replays", 0) >= 1
    w.stop()
