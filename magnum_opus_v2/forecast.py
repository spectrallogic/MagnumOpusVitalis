"""
ForecastLedger — imagined futures become accountable predictions.

Before this, the engine's speculation produced vivid, scored futures
that vanished without ever being checked against what actually
happened. The blueprint's charge was fair: a sampled continuation is
not a calibrated forecast. This ledger makes every imagined future a
BELIEF WITH A DUE DATE:

  record   — each speculation round's winner and penumbra entrants get
             a stable id, timestamp, horizon, and their stated
             probability (the rollout's chain confidence).
  resolve  — once the horizon passes, the forecast is scored against
             what the situation actually became (cosine of the
             forecast's direction vs the situation vector at
             resolution, threshold theta): hit, miss, or expired
             (no evidence arrived).
  calibrate— running Brier score and 10-bin reliability table of
             stated probability vs realized hit rate, per mode
             (speech / world / user).
  correct  — once enough resolutions exist, a per-bin monotone map
             turns raw chain confidence into probability_cal: what
             "likely" has MEASURABLY meant. The consumer is
             speculation's own utility ranking (speculative.py), which
             uses the calibrated value when available.

Honesty notes: chain confidence was never designed to be a
probability — the whole point of calibration is to MEASURE how far
from one it is, and the reliability table ships to the dashboard
whichever way it comes out. The ledger is in-memory per session (the
engine is a per-process runtime; primordium is the persistent
organism, and its Wheel keeps its own calibration ring).
"""

import itertools
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class ForecastLedger:
    def __init__(self, horizon_s: float = 45.0, hit_cos: float = 0.35,
                 max_open: int = 64, min_resolutions: int = 30):
        self.horizon_s = float(horizon_s)
        self.hit_cos = float(hit_cos)
        self.max_open = int(max_open)
        self.min_resolutions = int(min_resolutions)

        self._ids = itertools.count(1)
        self.open: List[dict] = []
        self.resolved: List[dict] = []          # bounded below
        self._lock = threading.Lock()

        # per-mode calibration: 10 bins of (n, hits)
        self._bins: Dict[str, List[List[int]]] = {}

    # ------------------------------------------------------------------
    def record(self, futures: List[dict], tick: int) -> None:
        """Called once per speculation round with the scored futures
        (winner first). Each becomes an open forecast."""
        now = time.monotonic()
        with self._lock:
            for f in futures:
                if f.get("vec") is None:
                    continue
                self.open.append({
                    "id": next(self._ids),
                    "ts": now,
                    "tick": tick,
                    "mode": f.get("mode", "world"),
                    "source": f.get("source", "?"),
                    "phrase": (f.get("name") or "")[:48],
                    "vec": f["vec"].detach().float().cpu(),
                    "probability": float(f.get("probability", 0.0)),
                    "utility": float(f.get("utility", 0.0)),
                    "deadline": now + self.horizon_s,
                    "status": "open",
                })
            # bound the open set: oldest expire unresolved (honest count)
            while len(self.open) > self.max_open:
                stale = self.open.pop(0)
                stale["status"] = "expired"
                self._keep(stale)

    def resolve(self, reality_vec: Optional[torch.Tensor]) -> int:
        """Called when fresh evidence exists (a new percept landed or a
        new speculation round observed the situation). Forecasts past
        their deadline are scored against reality; returns how many
        resolved this call."""
        if reality_vec is None:
            return 0
        r = reality_vec.detach().float().cpu().flatten()
        rn = float(r.norm())
        if rn < 1e-6:
            return 0
        r = r / rn
        now = time.monotonic()
        n = 0
        with self._lock:
            still_open = []
            for fc in self.open:
                if now < fc["deadline"]:
                    still_open.append(fc)
                    continue
                v = fc["vec"]
                vn = float(v.norm())
                if vn < 1e-6:
                    fc["status"] = "expired"
                else:
                    cos = float(F.cosine_similarity(
                        (v / vn).unsqueeze(0), r.unsqueeze(0)))
                    fc["evidence_cos"] = round(cos, 4)
                    fc["status"] = "hit" if cos >= self.hit_cos else "miss"
                    self._score(fc)
                self._keep(fc)
                n += 1
            self.open = still_open
        return n

    # ------------------------------------------------------------------
    def _keep(self, fc: dict) -> None:
        fc.pop("vec", None)                     # ledger keeps facts, not
        self.resolved.append(fc)                # tensors, once closed
        if len(self.resolved) > 512:
            del self.resolved[:-512]

    def _score(self, fc: dict) -> None:
        mode = fc["mode"]
        bins = self._bins.setdefault(mode, [[0, 0] for _ in range(10)])
        p = min(max(fc["probability"], 0.0), 0.999)
        b = int(p * 10)
        bins[b][0] += 1
        bins[b][1] += 1 if fc["status"] == "hit" else 0

    # ------------------------------------------------------------------
    # calibration readout + the correction map (the ledger's consumer)
    # ------------------------------------------------------------------
    def calibrated(self, probability: float, mode: str) -> Optional[float]:
        """Observed hit-rate of this probability's bin, once enough
        resolutions exist. None until the ledger has earned an opinion."""
        bins = self._bins.get(mode)
        if bins is None:
            return None
        total = sum(n for n, _ in bins)
        if total < self.min_resolutions:
            return None
        b = int(min(max(probability, 0.0), 0.999) * 10)
        n, h = bins[b]
        if n == 0:                              # empty bin: pooled rate
            hits = sum(h2 for _, h2 in bins)
            return hits / total
        return h / n

    def metrics(self) -> dict:
        with self._lock:
            closed = [f for f in self.resolved if f["status"] in
                      ("hit", "miss")]
            out: dict = {"open": len(self.open),
                         "resolved": len(closed),
                         "expired": sum(1 for f in self.resolved
                                        if f["status"] == "expired")}
            if closed:
                briers, hits = [], 0
                for f in closed:
                    y = 1.0 if f["status"] == "hit" else 0.0
                    hits += y
                    briers.append((f["probability"] - y) ** 2)
                out["brier"] = round(sum(briers) / len(briers), 4)
                out["hit_rate"] = round(hits / len(closed), 4)
                # 10-bin ECE across modes pooled
                pooled = [[0, 0] for _ in range(10)]
                for mode_bins in self._bins.values():
                    for i, (n, h) in enumerate(mode_bins):
                        pooled[i][0] += n
                        pooled[i][1] += h
                total = sum(n for n, _ in pooled)
                ece = 0.0
                for i, (n, h) in enumerate(pooled):
                    if n == 0:
                        continue
                    conf = (i + 0.5) / 10
                    ece += (n / total) * abs(h / n - conf)
                out["ece"] = round(ece, 4)
            return out

    def snapshot(self) -> dict:
        m = self.metrics()
        with self._lock:
            m["recent"] = [
                {k: f[k] for k in ("phrase", "mode", "probability",
                                   "status") if k in f}
                | ({"evidence_cos": f["evidence_cos"]}
                   if "evidence_cos" in f else {})
                for f in self.resolved[-6:]
            ]
        return m
