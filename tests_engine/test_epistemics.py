"""Era 6-M2 — epistemic guards (ADR-002). The live bug this closes:
confabulated memories flowed unguarded through Consolidation into the
abstraction ladder (and could become permanent bus attractors), and the
SelfModel could leak a just-minted false memory into the substrate as
"what-just-was". Policy: canonical paths blocked, the undertow
keeps dreaming — BOTH directions are asserted here so neither half of
the design gets 'fixed' away later."""

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.regions.memory import Memory, FalseMemoryConfabulator
from magnum_opus_v2.regions.consolidation import Consolidation
from magnum_opus_v2.regions.self_model import SelfModel


def _seeded_memory(n_real=6, d=64):
    torch.manual_seed(11)
    mem = Memory(device="cpu", capacity=50)
    for i in range(n_real):
        mem.capture_experience(torch.randn(d), importance=1.0,
                               tag=f"real{i}")
    return mem


def _confabulate(mem, bus):
    confab = FalseMemoryConfabulator(mem, per_tick_count=1)
    before = len(mem.pool)
    confab.step(bus, None, 30.0)
    assert len(mem.pool) == before + 1, "confabulation did not fire"
    fake = mem.pool[-1]
    assert fake.confidence < 1.0
    assert fake.meta["epistemic"] == "confabulated"
    fake.meta["importance"] = 2.0          # worst case: vivid false memory
    return fake


class _LadderSpy:
    def __init__(self):
        self.seen = []

    def observe_external(self, vec, neuromod):
        self.seen.append(vec.detach().clone())

    def dominant_concept(self):
        return None


def test_confabulation_cannot_become_canonical():
    d = 64
    bus = LatentBus(d, device="cpu")
    mem = _seeded_memory(d=d)
    fake = _confabulate(mem, bus)

    spy = _LadderSpy()
    cons = Consolidation(memory=mem, abstraction=spy)
    cons.step(bus, None, 30.0)

    assert cons.last_replayed > 0, "consolidation replayed nothing at all"
    for vec in spy.seen:                    # the vivid lie taught nothing
        assert not torch.allclose(vec, fake.vec), \
            "a confabulated trace reached the abstraction ladder"


def test_self_leak_is_something_that_was():
    d = 64
    bus = LatentBus(d, device="cpu")
    mem = _seeded_memory(d=d)
    _confabulate(mem, bus)                  # lands at pool[-1]

    sm = SelfModel(memory=mem, device="cpu")
    for _ in range(3):
        sm.step(bus, None, 0.2)
    assert sm.last_leak_tag is not None
    assert sm.last_leak_tag != "confabulated", \
        "the self-model leaked a false memory as what-just-was"
    assert sm.last_leak_tag.startswith("real")


def test_the_undertow_still_dreams():
    """The other half of the policy: false memories REMAIN in
    subconscious sampling — marked, down-weighted, deliberately
    indistinguishable to the dreaming layers. That is design, not a bug,
    and this assertion protects it from future 'cleanups'."""
    d = 64
    bus = LatentBus(d, device="cpu")
    mem = _seeded_memory(d=d)
    fake = _confabulate(mem, bus)

    sampler = mem.make_sampler()
    hit = False
    for _ in range(60):
        for c in sampler.sample(4, bias=None):
            if c.confidence < 1.0 and torch.allclose(c.vec, fake.vec):
                hit = True
    assert hit, "the confabulation vanished from the subconscious sea"


def test_imagined_content_is_born_tagged():
    """Primordium's mind's-eye — the 'imagined' epistemic type in the
    flesh: every sample carries its source and reduced confidence."""
    from primordium.mind.minds_eye import MindsEyeRing, MindsEyeSampler
    ring = MindsEyeRing(maxlen=8)
    for kind in ("prediction", "dream"):
        ring.push(torch.randn(64), kind=kind)
    sampler = MindsEyeSampler(ring)
    got = sampler.sample(6)
    assert got, "no imagined samples emerged"
    for c in got:
        assert c.source.startswith("minds_eye:")
        assert c.confidence < 1.0
