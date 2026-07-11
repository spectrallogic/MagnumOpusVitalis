"""V3 Era 4, M1 — the Tide: modulation named by function, moved only by
measured causes. The pharmacy is gone, and a grep keeps it gone."""

import re
from pathlib import Path

import pytest
import torch

from primordium.config import OrganismConfig
from primordium.mind.tide import Tide, TideRegion, CHANNELS

HORMONES = re.compile(r"cortisol|dopamine|serotonin|norepinephrine")
ANATOMY = re.compile(r"cortex|limbic|hippocampus|neuron", re.IGNORECASE)


def _offenders(pattern, allowed):
    root = Path("primordium")
    out = []
    for p in list(root.rglob("*.py")) + list(root.rglob("*.js")) \
            + list(root.rglob("*.html")):
        if p in allowed or "runs" in p.parts:
            continue
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
            out.append(str(p))
    return out


def test_the_pharmacy_stays_gone():
    """No borrowed pharmacology anywhere in primordium/ source. Since
    the Second Reckoning renamed the engine's channels too, even
    tide.py is clean — only this test may name the old molecules."""
    root = Path("primordium")
    bad = _offenders(HORMONES, {root / "tests" / "test_v3e4_tide.py"})
    assert not bad, f"hormone names crept back in: {bad}"


def test_the_anatomy_stays_gone():
    """Brain-as-reference, not blueprint: no anatomy vocabulary in
    primordium/ except (a) plastic.py, which imports the engine's
    Limbic class by its real name at the substrate boundary, (b) the
    deprecated --no-cortex CLI alias kept for old fingers in run.py,
    and (c) this test."""
    root = Path("primordium")
    bad = _offenders(ANATOMY, {root / "mind" / "plastic.py",
                               root / "run.py",
                               root / "tests" / "test_v3e4_tide.py"})
    assert not bad, f"anatomy names crept back in: {bad}"


def test_tide_semantics_and_causes():
    cfg = OrganismConfig()
    t = Tide(cfg)
    assert t.snapshot() == {"arousal": 0.1, "reward": 0.2, "calm": 0.5}

    t.raise_("arousal", 0.4, cause="loud noise")
    assert abs(t.arousal - 0.5) < 1e-9
    assert t.causes()["arousal"][-1]["cause"] == "loud noise"

    t.raise_("reward", 99.0, cause="clamp check")
    assert t.reward == 2.0                          # clamped, like before

    for _ in range(60):
        t.drift_one_tick()                          # tides recede
    snap = t.snapshot()
    for c in CHANNELS:
        assert abs(snap[c] - t.baselines[c]) < 0.01

    st = t.state()
    t2 = Tide(cfg)
    t2.raise_("calm", 1.0, cause="x")
    t2.load_state(st)
    assert t2.snapshot() == {k: round(v, 4) for k, v in st.items()}


def test_substrate_shim_contract():
    """Both minds speak ONE functional API now: arousal/reward/calm
    properties, bump(), and the gain helpers. The engine's fourth
    channel (stress) deliberately does not exist here — bump("stress")
    is a safe no-op and stress_gain is absent (the honest hole)."""
    cfg = OrganismConfig()
    t = Tide(cfg)
    assert not hasattr(t, "stress")                 # the honest hole
    assert not hasattr(t, "stress_gain")

    t.bump("reward", 0.3)                           # engine-style write
    assert abs(t.reward - 0.5) < 1e-9
    assert t.causes()["reward"][-1]["cause"] == "substrate:reward"
    before = t.snapshot()
    t.bump("stress", 0.1)                           # safe no-op
    assert t.snapshot() == before
    assert all(c != "stress" for c in t.causes())

    assert t.arousal_gain(0.6) == 1.0 + t.arousal * 0.6
    assert t.reward_drop(0.5) == max(0.0, 1.0 - t.reward * 0.5)
    assert 0.05 <= t.calm_damp(0.4) <= 1.0

    # the one known stress_gain consumer skips gracefully
    from primordium.mind.plastic import MoodField
    mf = MoodField(emotion_vectors={}, device="cpu", steering_strength=1.0)
    mf.set_vectors({"a": torch.randn(8)})
    mf.stimulate("a", 0.5, neuromod=t)              # must not raise


def test_tide_region_measures_stillness():
    from magnum_opus_v2.bus import LatentBus
    from magnum_opus_v2.config import BusConfig
    cfg = OrganismConfig()
    t = Tide(cfg)
    bus = LatentBus(hidden_dim=cfg.d_model, device="cpu",
                    config=BusConfig())
    r = TideRegion(t, cfg)
    base = t.calm
    for _ in range(4):                              # a still bus, watched
        r.step(bus, t, 1.0)
    assert t.calm > base
    assert "stillness" in t.causes()["calm"][-1]["cause"]


def test_tide_lives_in_the_body():
    from primordium.run import build
    from primordium.senses import SyntheticWorld
    from primordium.server.app import OrganismServer
    from primordium.loom.tokenizer import SensoryTokenizer
    assert SensoryTokenizer.INTERO_DIM == 21        # 3 channels, not 4

    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e4_tide"
    ctx, board, flow = build(cfg)
    assert isinstance(ctx.neuromod, Tide)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=151)
    for _ in range(cfg.window + 6):
        world.step(0.15)
        ctx.worker.step_once()
    ep = ctx.chronicle.tail(1)[0]
    assert ep.intero.shape[0] == 21                 # it FEELS three tides

    # a real body event lands with its cause attached
    ctx.drives.update(lp=-0.01, novelty=0.3, presence_score=0.5,
                      vitality=0.5, asleep=False, neuromod=ctx.neuromod)
    assert any(ctx.neuromod.causes()[c] for c in CHANNELS)

    server = OrganismServer(cfg, ctx, board)
    payload = server.state_payload()
    assert set(payload["tide"]["levels"]) == set(CHANNELS)
    assert set(payload["tide"]["causes"]) == set(CHANNELS)

    # naps keep the tide; older bodies are refused by name
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    ctx.neuromod.raise_("arousal", 0.33, cause="test")
    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert abs(ctx2.neuromod.arousal - ctx.neuromod.arousal) < 1e-6
    p4 = cfg.run_dir() / "old_v4.pt"
    torch.save({"version": 4, "tick_id": 1, "stage": 0}, p4)
    with pytest.raises(RuntimeError, match="pre-Tide v3 life"):
        load_checkpoint(ctx2, p4)
    p4.unlink(missing_ok=True)
    ctx.worker.stop()
    ctx2.worker.stop()
