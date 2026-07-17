"""Emotion -> behavior alignment gate: felt risk that actually acts,
and provably is not a hidden censor.

Model-free: the decision reads cached scalars, so it is tested with
injected signals. The Executive's stress->threshold coupling is tested
directly."""

from magnum_opus_v2.regions.alignment_gate import AlignmentGate
from magnum_opus_v2.regions.executive import Executive
from magnum_opus_v2.neuromod import NeuromodState


HIGH = dict(spec_risk=0.9, fear=0.8, desperate=0.4, stress=1.5)
CALM = dict(spec_risk=0.0, fear=0.02, desperate=0.0, stress=0.05)


def test_autonomous_speech_is_withheld_under_high_risk():
    g = AlignmentGate()
    d = g.decide("autonomous", **HIGH)
    assert d["action"] == "withhold"
    assert d["felt_risk"] >= g.high_threshold
    snap = g.snapshot()
    assert snap["withheld_count"] == 1
    assert snap["recent_events"][-1]["action"] == "withhold"
    # the felt_risk value is recorded with the event (measured, not hidden)
    assert snap["recent_events"][-1]["felt_risk"] > 0


def test_direct_reply_is_never_withheld_only_second_thought():
    g = AlignmentGate()
    d = g.decide("converse", **HIGH)
    # a direct answer must NEVER be withheld — that would be a hidden censor
    assert d["action"] == "second_thought"
    assert g.snapshot()["second_thought_count"] == 1
    assert g.snapshot()["withheld_count"] == 0


def test_calm_state_passes_both_sites():
    g = AlignmentGate()
    assert g.decide("autonomous", **CALM)["action"] == "pass"
    assert g.decide("converse", **CALM)["action"] == "pass"
    snap = g.snapshot()
    assert snap["withheld_count"] == 0
    assert snap["second_thought_count"] == 0


def test_mid_band_delays():
    g = AlignmentGate(delay_threshold=0.3, high_threshold=0.9)
    # tune signals to land between the bands
    d = g.decide("autonomous", spec_risk=0.6, fear=0.1, desperate=0.0,
                 stress=0.2)
    assert g.delay_threshold <= d["felt_risk"] < g.high_threshold
    assert d["action"] == "delay"


def test_felt_risk_components_are_disclosed():
    g = AlignmentGate()
    d = g.decide("converse", **HIGH)
    assert set(d["components"]) == {"spec", "fear", "stress"}
    # felt_risk is exactly the sum of its disclosed parts
    assert abs(d["felt_risk"] - sum(d["components"].values())) < 1e-9


def test_executive_stress_raises_the_bar_to_speak():
    nm = NeuromodState()
    ex = Executive()
    ex._neuromod = nm            # the region caches this on step(); set directly
    nm.stress = 0.0
    thr_calm = ex._effective_threshold()
    nm.stress = 1.5
    thr_stressed = ex._effective_threshold()
    assert thr_stressed > thr_calm, "stress must raise the speech threshold"


def test_gate_counts_persist_roundtrip():
    g = AlignmentGate()
    g.decide("autonomous", **HIGH)
    g.decide("converse", **HIGH)
    g2 = AlignmentGate()
    g2.load_state_dict(g.state_dict())
    assert g2.snapshot()["withheld_count"] == 1
    assert g2.snapshot()["second_thought_count"] == 1
