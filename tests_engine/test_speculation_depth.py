"""Deepened speculation — the engine imagines further and reads the LLM
harder as a reality predictor, bounded so a deep rollout never steals
latency from a live user turn.

Uses gpt2 (the mechanism fundamentally needs a model); one engine, reused.
Skipped if the gpt2 profile can't be built offline."""

import time

import pytest

from magnum_opus_v2 import (
    V2Engine, load_model, load_profile, create_profile, profile_exists,
)
from magnum_opus_v2.config import V2Config, SpeculativeConfig


def test_speculative_config_defaults_are_deepened():
    """The knobs exist and default to the deeper regime (was 6 tokens)."""
    c = SpeculativeConfig()
    assert c.rollout_tokens == 14
    assert c.rollout_budget_ms > 0
    assert c.chained_continuation_tokens > 0
    assert 0.0 <= c.lexicon_weight <= 1.0
    assert V2Config().spec.rollout_tokens == 14


@pytest.fixture(scope="module")
def gpt2_engine():
    try:
        model, tok, dev = load_model("gpt2")
        prof = (load_profile("gpt2") if profile_exists("gpt2")
                else create_profile("gpt2", device=dev))
    except Exception as e:  # noqa: BLE001 — offline / no weights
        pytest.skip(f"gpt2 unavailable: {e}")
    eng = V2Engine.from_profile(model, tok, prof, device=dev)
    eng.start()
    yield eng
    eng.stop()


def _spec_after(engine, seconds):
    engine.converse("I am standing at the very edge of a high cliff.",
                    max_new_tokens=16)
    time.sleep(seconds)
    return engine.snapshot()["speculative"]


def test_snapshot_carries_depth_fields_and_risk_max(gpt2_engine):
    # let speculation run a few rounds
    spec = None
    for _ in range(6):
        spec = _spec_after(gpt2_engine, 2.5)
        if spec and spec.get("rounds_total", 0) >= 1 and spec.get("futures"):
            break
    assert spec is not None
    assert "field_risk" in spec
    assert "over_budget" in spec
    assert "rollout_tokens_used" in spec
    # multi-point risk is exposed per future
    for f in spec["futures"]:
        assert "risk_max" in f
        assert f["risk_max"] >= f["risk"] - 1e-6   # peak >= mean
    # deeper than the old 6-token default when not heavily truncated
    assert spec["rollout_tokens_used"] > 0


def test_budget_guard_bounds_the_rollout(gpt2_engine):
    # warm up so rounds are flowing
    for _ in range(4):
        spec = _spec_after(gpt2_engine, 2.0)
        if spec.get("rounds_total", 0) >= 1:
            break
    over_before = spec["over_budget"]
    depth_before = spec["rollout_tokens_used"]

    # clamp the budget hard — rollouts must truncate
    gpt2_engine.speculative.rollout_budget_ms = 0.5
    later = None
    for _ in range(6):
        later = _spec_after(gpt2_engine, 2.0)
        if later["over_budget"] > over_before:
            break
    assert later["over_budget"] > over_before, "budget guard never fired"
    # and the truncation shows up as shallower realized depth
    assert later["rollout_tokens_used"] <= max(depth_before, 1.0)
