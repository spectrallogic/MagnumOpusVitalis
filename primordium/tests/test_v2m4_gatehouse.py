"""V2-M4 — Gatehouse (two keys, honest milestones) + Foster (the
caregiver as environment). The webtext gate must be provably hollow."""

import json
import time
from pathlib import Path

import pytest

from primordium.config import OrganismConfig
from primordium.persistence import calib
from primordium.safety.gatehouse import (Gatehouse, GateLockedError,
                                         write_creator_word, verify_proof)


PHRASE = "olive-branch-for-tests"


@pytest.fixture()
def test_calib(tmp_path, monkeypatch):
    """A throwaway calibration record so tests never touch the real one."""
    rec = calib._seal(PHRASE, b"test lineage record")
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(rec))
    monkeypatch.setattr(calib, "_RECORD", p)
    return p


def test_gate_state_machine_and_creator_word(tmp_path, test_calib):
    cfg = OrganismConfig()
    gh = Gatehouse(cfg, tmp_path)
    assert gh.intero_flags() == [1.0, 0.0, 1.0, 0.0]   # felt, shut
    assert not gh.is_open("caregiver")

    # measured milestones: partial -> LOCKED with honest progress
    gh.set_progress("caregiver", {"stage": (0.5, False), "age": (1.0, True)})
    snap = gh.snapshot()["caregiver"]
    assert snap["state"] == "locked" and 0 < snap["progress"] < 1

    # the creator's word alone is NOT enough while milestones fail
    assert not write_creator_word(tmp_path, "caregiver", "unlock", "wrong!")
    assert write_creator_word(tmp_path, "caregiver", "unlock", PHRASE)
    assert gh.poll_creator_word() == []                # not eligible yet
    assert not gh.is_open("caregiver")

    # milestones alone are not enough either: ELIGIBLE is not open
    assert gh.set_progress("caregiver", {"stage": (1.0, True)}) == "eligible"
    assert not gh.is_open("caregiver")

    # both keys together open it (the earlier word now applies)
    changes = gh.poll_creator_word()
    assert changes == [{"gate": "caregiver", "to": "unlocked"}]
    assert gh.is_open("caregiver")
    assert gh.intero_flags() == [1.0, 1.0, 1.0, 0.0]

    # the stored proof re-verifies with the phrase, and only the phrase
    entry = json.loads((tmp_path / "gates.json").read_text())[-1]
    assert verify_proof(PHRASE, entry)
    assert not verify_proof("wrong!", entry)

    # eligibility regression never touches an UNLOCKED gate...
    gh.set_progress("caregiver", {"stage": (0.0, False)})
    assert gh.is_open("caregiver")
    # ...but a creator lock always lands
    assert write_creator_word(tmp_path, "caregiver", "lock", PHRASE)
    assert gh.poll_creator_word() == [{"gate": "caregiver", "to": "locked"}]
    assert not gh.is_open("caregiver")

    # state survives a checkpoint roundtrip
    gh2 = Gatehouse(cfg, tmp_path)
    gh2.load_state_dict(gh.state_dict())
    assert gh2.snapshot() == gh.snapshot()


def test_webtext_ships_hollow():
    from primordium.safety.webtext_stub import WebtextPort
    port = WebtextPort(gatehouse=None)
    with pytest.raises(GateLockedError):
        port.fetch("https://example.com")
    assert port.ALLOWLIST == () and port.MAX_BYTES_PER_DAY == 0
    # provably no networking: no import in the module can reach a wire
    src = Path("primordium/safety/webtext_stub.py").read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert imports == ["from primordium.safety.gatehouse import "
                       "GateLockedError"], imports


def test_gate_sense_measures_and_applies(tmp_path, test_calib):
    from primordium.mind.coupling import GateSenseRegion
    from primordium.run import build
    from primordium.senses import SyntheticWorld
    cfg = OrganismConfig()
    cfg.run_name = "_test_v2m4_sense"
    cfg.gate_caregiver_age_ticks = 8
    (cfg.run_dir() / "gates.json").unlink(missing_ok=True)  # fresh ceremony
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=51)
    region = GateSenseRegion(cfg, w, ctx.compass, ctx.gatehouse, board,
                             pulse=ctx.pulse)
    for _ in range(10):
        world.step(0.15)
        w.step_once()
    region.step(ctx.bus, ctx.neuromod, 1.0)
    snap = ctx.gatehouse.snapshot()["caregiver"]
    assert snap["state"] == "locked"
    assert snap["milestones"]["age"]["met"]            # measured, honestly
    assert not snap["milestones"]["stage"]["met"]

    # fake maturity, measured the same way the organism would earn it
    import torch
    w.stage = 1
    w._L_birth = 1e6                                   # loss far below birth
    for i in range(3):
        ctx.compass.installed[f"affect_{i}"] = torch.randn(384)
        ctx.compass.provenance[f"affect_{i}"] = "lived"
    region.step(ctx.bus, ctx.neuromod, 1.0)
    assert ctx.gatehouse.snapshot()["caregiver"]["state"] == "eligible"

    write_creator_word(ctx.cfg.run_dir(), "caregiver", "unlock", PHRASE)
    region.step(ctx.bus, ctx.neuromod, 1.0)
    assert ctx.gatehouse.is_open("caregiver")
    kinds = {e["kind"] for e in ctx.pulse.since(0, limit=500)}
    assert "gate_state" in kinds

    # the open gate is FELT: interoception flags flip
    assert ctx.gatehouse.intero_flags() == [1.0, 1.0, 1.0, 0.0]

    # and survives the organism's own checkpoint
    w.stage = 0                        # body was never really surgeried
    from primordium.persistence.checkpoint import save_checkpoint, \
        load_checkpoint
    path = save_checkpoint(ctx)
    ctx2, board2, flow2 = build(cfg)
    load_checkpoint(ctx2, path)
    assert ctx2.gatehouse.is_open("caregiver")
    w.stop()
    ctx2.worker.stop()

    # cleanup so other tests' runs aren't affected
    (ctx.cfg.run_dir() / "gates.json").unlink(missing_ok=True)


def test_foster_stub_protocol(tmp_path, test_calib):
    """The whole caregiver pipeline with a deterministic stand-in:
    spawn iff gate open + flag, babble -> observe -> reply enters the
    Wordstream tagged foster, absence surfaced when the gate shuts."""
    from primordium.body.expression import Wordstream
    from primordium.events.pulse import Pulse
    from primordium.mind.coupling import FosterRegion

    class _Sleep:
        asleep = False

    class _Exec:
        marked = 0
        def mark_interaction(self):
            self.marked += 1

    class _Board(dict):
        def put(self, **kv): self.update(kv)
        def get(self, k, d=None): return dict.get(self, k, d)

    cfg = OrganismConfig()
    cfg.caregiver = True
    cfg.caregiver_stub = True
    cfg.caregiver_reply_min_s = 0.0
    gh = Gatehouse(cfg, tmp_path)
    gh.load_state_dict({"gates": {"caregiver": {"state": "unlocked"}}})
    ws = Wordstream(cfg)
    pulse = Pulse(64)
    ex = _Exec()
    board = _Board()
    from primordium.caregiver.foster import Foster
    region = FosterRegion(cfg, ws, gh, _Sleep(), ex, board,
                          pulse=pulse, foster_factory=lambda: Foster(cfg))
    try:
        region.step(None, None, 1.0)                   # spawns the stub
        assert region.foster is not None
        t0 = time.monotonic()
        while not region.foster.ready and time.monotonic() - t0 < 15:
            time.sleep(0.1)
        assert region.foster.ready and region.foster.alive

        ws.type_chars([104, 105])                      # it types "hi"
        got_reply = False
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15 and not got_reply:
            region.step(None, None, 1.0)
            got_reply = any(m["source"] == "foster"
                            for m in ws.snapshot()["transcript"])
            time.sleep(0.1)
        assert got_reply, "no caregiver reply reached the Wordstream"
        assert ex.marked >= 1
        assert board.get("foster_contact_until", 0) > time.monotonic() - 1
        kinds = {e["kind"] for e in pulse.since(0, limit=200)}
        assert "caregiver_present" in kinds and "caregiver_msg" in kinds

        # shutting the gate removes the caregiver, honestly
        gh.load_state_dict({"gates": {"caregiver": {"state": "locked"}}})
        region.step(None, None, 1.0)
        assert region.foster is None
        kinds = {e["kind"] for e in pulse.since(0, limit=200)}
        assert "caregiver_absent" in kinds
    finally:
        if region.foster is not None:
            region.foster.stop()
