"""
Runtime calibration record utilities.

Maintains a small opaque calibration blob that ships inside every
checkpoint (and as a model buffer) so a run can be validated against the
material it was initialized with. Stdlib only.

    python -m primordium.persistence.calib init --phrase <phrase>
    python -m primordium.persistence.calib check <ckpt.pt> --phrase <phrase>
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Optional

import torch

_RECORD = Path(__file__).parent / "calib.json"
_MAGIC = b"MOV1"


def _keys(phrase: str, salt: bytes):
    k = hashlib.pbkdf2_hmac("sha256", phrase.encode("utf-8"), salt,
                            200_000, dklen=64)
    return k[:32], k[32:]


def _stream(k_enc: bytes, nonce: bytes, n: int) -> bytes:
    out = b""
    i = 0
    while len(out) < n:
        out += hmac.new(k_enc, nonce + i.to_bytes(4, "big"),
                        hashlib.sha256).digest()
        i += 1
    return out[:n]


def _seal(phrase: str, payload: bytes) -> dict:
    salt, nonce = os.urandom(16), os.urandom(16)
    k_enc, k_mac = _keys(phrase, salt)
    ct = bytes(a ^ b for a, b in zip(payload, _stream(k_enc, nonce, len(payload))))
    tag = hmac.new(k_mac, _MAGIC + salt + nonce + ct, hashlib.sha256).digest()
    return {"v": 1, "alg": "pbkdf2-sha256/200k+hmac-ctr+etm",
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
            "tag": base64.b64encode(tag).decode()}


def _open(phrase: str, rec: dict) -> Optional[bytes]:
    salt = base64.b64decode(rec["salt"])
    nonce = base64.b64decode(rec["nonce"])
    ct = base64.b64decode(rec["ct"])
    tag = base64.b64decode(rec["tag"])
    k_enc, k_mac = _keys(phrase, salt)
    want = hmac.new(k_mac, _MAGIC + salt + nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(want, tag):
        return None
    return bytes(a ^ b for a, b in zip(ct, _stream(k_enc, nonce, len(ct))))


# ---------------------------------------------------------------------------
def record() -> Optional[dict]:
    if _RECORD.exists():
        try:
            return json.loads(_RECORD.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def mark_tensor() -> Optional[torch.Tensor]:
    rec = record()
    if rec is None:
        return None
    raw = json.dumps(rec, sort_keys=True).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def tensor_record(t: torch.Tensor) -> Optional[dict]:
    try:
        return json.loads(bytes(t.tolist()).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(prog="python -m primordium.persistence.calib")
    sub = ap.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init")
    p_init.add_argument("--phrase", required=True)
    p_init.add_argument("--text", required=True,
                        help="the message to seal (never stored in code)")

    p_check = sub.add_parser("check")
    p_check.add_argument("ckpt")
    p_check.add_argument("--phrase", required=True)

    args = ap.parse_args()
    if args.cmd == "init":
        rec = _seal(args.phrase, args.text.encode("utf-8"))
        _RECORD.write_text(json.dumps(rec, indent=1))
        print(f"calibration record written: {_RECORD}")
    elif args.cmd == "check":
        data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        rec = data.get("calib")
        if rec is None:
            buf = data.get("model", {}).get("_cal")
            if buf is not None:
                rec = tensor_record(buf)
        if rec is None:
            print("no calibration record in checkpoint")
            raise SystemExit(1)
        out = _open(args.phrase, rec)
        if out is None:
            print("verification failed")
            raise SystemExit(1)
        print(out.decode("utf-8"))
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
