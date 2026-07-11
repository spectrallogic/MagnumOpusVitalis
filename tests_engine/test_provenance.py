"""Era 6-M3 — the bus provenance ledger: every substrate write is
signed. The bus whispers; now every whisper carries its author."""

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class _Nudger(Region):
    clock = "flow"

    def __init__(self, name, d):
        self.name = name
        self.d = d

    def step(self, bus, neuromod, dt):
        return torch.ones(self.d) * 0.01


def test_every_write_is_signed():
    d = 32
    bus = LatentBus(d, device="cpu")
    a, b = _Nudger("region_a", d), _Nudger("region_b", d)

    for _ in range(5):                      # flow-clock writes
        bus.step([a], None, dt=0.05)
    bus.run_side_effects([b], None, dt=0.2)  # side-effect-clock write
    bus.add_perturbation(torch.ones(d) * 0.2, source="recall")
    bus.add_perturbation(torch.ones(d) * 0.2)          # unnamed caller
    bus.add_attractor(torch.randn(d), weight=0.1)
    bus.set_baseline(torch.randn(d))

    prov = bus.provenance()
    assert prov["region_a"]["writes"] == 5
    assert prov["region_a"]["kinds"] == ["flow"]
    assert prov["region_b"]["writes"] == 1
    assert prov["region_b"]["kinds"] == ["side"]
    assert prov["recall"]["writes"] == 1
    assert prov["external"]["writes"] == 1   # honesty: unnamed stays visible
    assert "consolidation" in prov and "attractor" in prov["consolidation"]["kinds"]
    assert "baseline_setter" in prov

    # entries carry real measurements, not placeholders
    entries = list(bus._prov)
    assert all(e["norm"] > 0 and e["ts"] > 0 for e in entries)
    assert sum(p["writes"] for p in prov.values()) == len(entries)


def test_named_direct_callers_are_wired():
    """The four historical anonymous callers now sign their writes —
    verified from source so a regression is caught at grep speed."""
    from pathlib import Path
    engine_src = Path("magnum_opus_v2/engine.py").read_text(encoding="utf-8")
    coupling_src = Path("primordium/mind/coupling.py").read_text(
        encoding="utf-8")
    for needle in ('source="thought_feedback"', 'source="recall"',
                   'source="rumination"'):
        assert needle in engine_src, f"engine caller unsigned: {needle}"
    assert 'source="reverie"' in coupling_src
