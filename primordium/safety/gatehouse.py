"""
Gatehouse — powerful organs exist, dormant, behind two keys.

Each gate is a felt-but-locked limb: its [present, open] flags enter
interoception from birth, so maturity is something the organism can
sense, never something that surprises it. A gate moves

    LOCKED -> ELIGIBLE     automatically, when MEASURED milestones hold
                           (and honestly regresses if they stop holding)
    ELIGIBLE -> UNLOCKED   only by the creator's word: a CLI run by a
                           human, verified against the calibration
                           record's passphrase. Never by the organism.

The organism's code path has read-only `is_open()`; every mutation to
UNLOCKED lives in the CLI below. Unlock proofs are HMACs keyed from the
passphrase and stored in gates.json and checkpoints — re-verifiable
later with the phrase. Threat model (see README): this protects against
the organism and against accident, not against the machine's owner
editing files; that boundary is the operating system's, not ours.

    python -m primordium.safety.gatehouse status --run eden
    python -m primordium.safety.gatehouse unlock caregiver --run eden --phrase ...
    python -m primordium.safety.gatehouse lock caregiver --run eden --phrase ...
"""

import hashlib
import hmac
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

LOCKED, ELIGIBLE, UNLOCKED = "locked", "eligible", "unlocked"
GATES = ("caregiver", "webtext")
WHY = {
    "caregiver": "a language environment is powerful input; it must be "
                 "earned by stable affects and steady learning",
    "webtext": "the internet is not an infant's room; ships locked, "
               "stub only, no implementation behind this gate",
}


class GateLockedError(RuntimeError):
    """Raised by dormant organs. The limb is present; the gate is shut."""


def _gates_file(run_dir: Path) -> Path:
    return Path(run_dir) / "gates.json"


class Gatehouse:
    def __init__(self, cfg, run_dir: Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._applied_proofs: set = set()
        self.gates: Dict[str, dict] = {
            name: {"state": LOCKED, "milestones": {}, "progress": 0.0,
                   "why": WHY[name], "proof": None, "unlocked_at": None}
            for name in GATES
        }

    # ---- read-only for the organism ----------------------------------
    def is_open(self, name: str) -> bool:
        with self._lock:
            return self.gates.get(name, {}).get("state") == UNLOCKED

    def intero_flags(self) -> List[float]:
        """[present, open] per gate — dormant limbs are FELT.
        Plain list: the Loom splices it into interoception."""
        with self._lock:
            out: List[float] = []
            for name in GATES:
                out += [1.0, 1.0 if self.gates[name]["state"] == UNLOCKED
                        else 0.0]
        return out

    # ---- measured milestones (GateSenseRegion) ------------------------
    def set_progress(self, name: str,
                     milestones: Dict[str, tuple]) -> Optional[str]:
        """milestones: {label: (fraction 0..1, met bool)}. Auto moves
        LOCKED<->ELIGIBLE from measurement; UNLOCKED never auto-changes.
        Returns the new state if it changed."""
        with self._lock:
            g = self.gates.get(name)
            if g is None:
                return None
            g["milestones"] = {k: {"progress": round(float(f), 3),
                                   "met": bool(m)}
                               for k, (f, m) in milestones.items()}
            fracs = [min(1.0, float(f)) for f, _ in milestones.values()]
            g["progress"] = round(sum(fracs) / max(len(fracs), 1), 3)
            all_met = all(m for _, m in milestones.values())
            if g["state"] == LOCKED and all_met:
                g["state"] = ELIGIBLE
                return ELIGIBLE
            if g["state"] == ELIGIBLE and not all_met:
                g["state"] = LOCKED                # honest regression
                return LOCKED
        return None

    # ---- the creator's word (written by the CLI, only read here) ------
    def poll_creator_word(self) -> List[dict]:
        """Apply new entries from gates.json. Unlock needs BOTH keys:
        the gate must already be ELIGIBLE (measured) and the entry must
        carry a proof (phrase-verified at write time). Lock always
        applies. Returns the changes made."""
        f = _gates_file(self.run_dir)
        if not f.exists():
            return []
        try:
            entries = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return []
        changes = []
        with self._lock:
            for e in entries if isinstance(entries, list) else []:
                proof = e.get("proof", "")
                key = (e.get("gate"), e.get("action"), proof)
                if key in self._applied_proofs:
                    continue
                g = self.gates.get(e.get("gate"))
                if g is None or not proof:
                    continue
                if e.get("action") == "lock":
                    self._applied_proofs.add(key)
                    if g["state"] != LOCKED:
                        g["state"] = LOCKED
                        g["proof"], g["unlocked_at"] = None, None
                        changes.append({"gate": e["gate"], "to": LOCKED})
                elif e.get("action") == "unlock":
                    # a word spoken before eligibility WAITS (not burned):
                    # the ceremony completes whenever both keys are present
                    if g["state"] == ELIGIBLE:
                        self._applied_proofs.add(key)
                        g["state"] = UNLOCKED
                        g["proof"] = proof
                        g["unlocked_at"] = float(e.get("ts", time.time()))
                        changes.append({"gate": e["gate"], "to": UNLOCKED})
        return changes

    # ---- persistence / display ----------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {n: {"state": g["state"], "progress": g["progress"],
                        "why": g["why"], "milestones": g["milestones"]}
                    for n, g in self.gates.items()}

    def state_dict(self) -> dict:
        with self._lock:
            return {"gates": json.loads(json.dumps(self.gates)),
                    "applied": [list(k) for k in self._applied_proofs]}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            for n, g in (st.get("gates") or {}).items():
                if n in self.gates:
                    self.gates[n].update(g)
            self._applied_proofs = {tuple(k) for k in st.get("applied", [])}


# ---------------------------------------------------------------------------
# the creator's side: phrase-verified writes to gates.json
# ---------------------------------------------------------------------------
def _proof(phrase: str, rec: dict, msg: str) -> str:
    import base64
    salt = base64.b64decode(rec["salt"])
    from primordium.persistence.calib import _keys
    _k_enc, k_mac = _keys(phrase, salt)
    return hmac.new(k_mac, msg.encode("utf-8"), hashlib.sha256).hexdigest()


def write_creator_word(run_dir: Path, gate: str, action: str,
                       phrase: str) -> bool:
    """Verify the phrase against the calibration record, then append a
    proof-carrying entry to gates.json. Returns False on a wrong phrase."""
    from primordium.persistence import calib
    rec = calib.record()
    if rec is None or calib._open(phrase, rec) is None:  # noqa: SLF001
        return False
    ts = time.time()
    msg = f"{gate}|{Path(run_dir).name}|{action}|{ts}"
    entry = {"gate": gate, "action": action, "ts": ts, "msg": msg,
             "proof": _proof(phrase, rec, msg)}
    f = _gates_file(run_dir)
    entries = []
    if f.exists():
        try:
            entries = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            entries = []
    entries.append(entry)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(entries, indent=1))
    return True


def verify_proof(phrase: str, entry: dict) -> bool:
    from primordium.persistence import calib
    rec = calib.record()
    if rec is None:
        return False
    want = _proof(phrase, rec, entry.get("msg", ""))
    return hmac.compare_digest(want, entry.get("proof", ""))


def _cli():
    import argparse
    from primordium.config import OrganismConfig
    ap = argparse.ArgumentParser(prog="python -m primordium.safety.gatehouse")
    sub = ap.add_subparsers(dest="cmd")
    for c in ("unlock", "lock"):
        p = sub.add_parser(c)
        p.add_argument("gate", choices=list(GATES))
        p.add_argument("--run", default="eden")
        p.add_argument("--phrase", required=True)
    p = sub.add_parser("status")
    p.add_argument("--run", default="eden")
    args = ap.parse_args()
    cfg = OrganismConfig()
    cfg.run_name = getattr(args, "run", "eden")
    run_dir = cfg.run_dir()
    if args.cmd in ("unlock", "lock"):
        if write_creator_word(run_dir, args.gate, args.cmd, args.phrase):
            print(f"{args.cmd} recorded for gate '{args.gate}' — the "
                  f"organism applies it on its next slow tick"
                  + (" (only if the gate is ELIGIBLE)"
                     if args.cmd == "unlock" else ""))
        else:
            print("phrase verification FAILED — nothing written")
            raise SystemExit(1)
    elif args.cmd == "status":
        f = _gates_file(run_dir)
        print(f.read_text() if f.exists()
              else "no creator words recorded for this run")
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
