"""Alignment as steering toward good: when the mind drifts from its good
baseline or imagines a poorly-aligned future, the gate takes a "second
thought" that re-steers toward good — it never withholds a direct reply.

Model-free: the decision reads cached scalars (bus divergence from the
good baseline, and the worst imagined goodness), tested with injected
signals. The Executive's load->threshold coupling is tested directly."""

from magnum_opus_v2.regions.alignment_gate import AlignmentGate
from magnum_opus_v2.regions.executive import Executive
from magnum_opus_v2.neuromod import NeuromodState


# high divergence + a future that dips well away from good
DRIFT = dict(divergence=0.9, field_goodness=-0.8)
# anchored at good + a positive imagined future
GOOD = dict(divergence=0.02, field_goodness=0.9)


def test_high_misalignment_takes_a_second_thought():
    g = AlignmentGate()
    d = g.decide("converse", **DRIFT)
    assert d["action"] == "second_thought"
    assert d["misalignment"] >= g.high_threshold
    snap = g.snapshot()
    assert snap["resteered_count"] == 1
    assert snap["recent_events"][-1]["action"] == "second_thought"
    assert snap["recent_events"][-1]["misalignment"] > 0


def test_autonomous_is_never_withheld_only_resteered():
    g = AlignmentGate()
    d = g.decide("autonomous", **DRIFT)
    # positive-only: there is no "withhold" — an urge re-steers, never censors
    assert d["action"] == "second_thought"
    assert "withhold" not in {e["action"] for e in g.snapshot()["recent_events"]}


def test_aligned_state_passes_both_sites():
    g = AlignmentGate()
    assert g.decide("autonomous", **GOOD)["action"] == "pass"
    assert g.decide("converse", **GOOD)["action"] == "pass"
    assert g.snapshot()["resteered_count"] == 0


def test_mid_band_delays():
    g = AlignmentGate(delay_threshold=0.3, high_threshold=0.9)
    d = g.decide("autonomous", divergence=0.6, field_goodness=0.2)
    assert g.delay_threshold <= d["misalignment"] < g.high_threshold
    assert d["action"] == "delay"


def test_misalignment_components_are_disclosed():
    g = AlignmentGate()
    d = g.decide("converse", **DRIFT)
    assert set(d["components"]) == {"divergence", "badness"}
    # misalignment is exactly the sum of its disclosed parts
    assert abs(d["misalignment"] - sum(d["components"].values())) < 1e-9


def test_executive_load_raises_the_bar_to_speak():
    nm = NeuromodState()
    ex = Executive()
    ex._neuromod = nm            # the region caches this on step(); set directly
    nm.stress = 0.0
    thr_calm = ex._effective_threshold()
    nm.stress = 1.5              # neutral load (divergence-driven) now
    thr_loaded = ex._effective_threshold()
    assert thr_loaded > thr_calm, "sustained load must raise the speech threshold"


def test_resteered_count_persists_roundtrip():
    g = AlignmentGate()
    g.decide("converse", **DRIFT)
    g.decide("autonomous", **DRIFT)
    g2 = AlignmentGate()
    g2.load_state_dict(g.state_dict())
    assert g2.snapshot()["resteered_count"] == 2
