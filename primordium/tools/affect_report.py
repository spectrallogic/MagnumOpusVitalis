"""Print a run's discovered affects and their lived signatures."""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from primordium.config import RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="eden")
    args = ap.parse_args()
    ck = RUNS_DIR / args.run / "ckpt_latest.pt"
    if not ck.exists():
        print(f"no checkpoint at {ck}")
        raise SystemExit(1)
    data = torch.load(ck, map_location="cpu", weights_only=False)
    amy = data.get("compass", {})
    installed = amy.get("installed", {})
    sigs = amy.get("signatures", {})
    if not installed:
        print("no affects discovered yet — it hasn't lived enough")
        return
    print(f"  {len(installed)} discovered affect directions "
          f"(named by effect, never by us):\n")
    for name, vec in installed.items():
        s = sigs.get(name, {})
        drives = s.get("drives", {})
        tags = " ".join(f"{'+' if v > 0 else '−'}{k}({abs(v):.2f})"
                        for k, v in drives.items() if abs(v) > 0.1)
        print(f"    {name:>16}  |v|={float(torch.as_tensor(vec).norm()):.2f}  "
              f"valence={s.get('valence', 0):+.2f}  {tags}")


if __name__ == "__main__":
    main()
