"""
ForecastLedger — imagined futures become accountable predictions.

Before this, the engine's speculation produced vivid, scored futures
that vanished without ever being checked against what actually
happened. A sampled continuation is not a calibrated forecast. This
ledger makes every imagined future a
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
  correct  — once enough resolutions exist, a per-bin empirical map
             turns raw chain confidence into probability_cal: what
             "likely" has MEASURABLY meant. The consumer is
             speculation's own utility ranking (speculative.py), which
             uses the calibrated value when available.

Honesty notes: chain confidence was never designed to be a
probability — the whole point of calibration is to MEASURE how far
from one it is, and the reliability table ships to the dashboard
whichever way it comes out. Engine persistence stores the calibration
bins and pending forecasts. This remains a latent-similarity proxy,
not a check that the forecast's verbal proposition occurred.
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
        self.journal = None                     # wired by the engine; guarded

        # per-mode calibration: 10 bins of (n, hits)
        self._bins: Dict[str, List[List[int]]] = {}

    # ------------------------------------------------------------------
    def record(self, futures: List[dict], tick: int) -> None:
        """Called once per speculation round with the scored futures
        (winner first). Each becomes an open forecast."""
        now = time.monotonic()
        with self._lock:
            for f in futures:
                forecast_vec = f.get("predicted_state", f.get("vec"))
                if forecast_vec is None:
                    continue
                fid = next(self._ids)
                self.open.append({
                    "id": fid,
                    "ts": now,
                    "tick": tick,
                    "mode": f.get("mode", "world"),
                    "source": f.get("source", "?"),
                    "phrase": (f.get("name") or "")[:48],
                    "vec": forecast_vec.detach().float().cpu(),
                    "probability": float(f.get("probability", 0.0)),
                    "utility": float(f.get("utility", 0.0)),
                    "deadline": now + self.horizon_s,
                    "status": "open",
                })
                if self.journal is not None:
                    try:
                        self.journal.emit(
                            "forecast_opened", turn=tick, fid=fid,
                            phrase=(f.get("name") or "")[:48],
                            mode=f.get("mode", "world"),
                            probability=round(float(f.get("probability", 0.0)), 3))
                    except Exception:  # noqa: BLE001
                        pass
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
                if self.journal is not None:
                    try:
                        self.journal.emit(
                            "forecast_resolved", turn=fc.get("tick", 0),
                            fid=fc.get("id"), phrase=fc.get("phrase", ""),
                            status=fc["status"],
                            evidence_cos=fc.get("evidence_cos"))
                    except Exception:  # noqa: BLE001
                        pass
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
        with self._lock:
            bins = self._bins.get(mode)
            bins = [list(b) for b in bins] if bins is not None else None
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
                # Same resolved window as Brier; use actual mean confidence,
                # not bin centers or the lifetime calibration training counts.
                pooled = [[0, 0, 0.0] for _ in range(10)]
                for f in closed:
                    p = min(max(f["probability"], 0.0), 1.0)
                    b = min(int(p * 10), 9)
                    pooled[b][0] += 1
                    pooled[b][1] += int(f["status"] == "hit")
                    pooled[b][2] += p
                total = len(closed)
                ece = 0.0
                for n, h, p_sum in pooled:
                    if n == 0:
                        continue
                    conf = p_sum / n
                    ece += (n / total) * abs(h / n - conf)
                out["ece"] = round(ece, 4)
            return out

    def snapshot(self) -> dict:
        m = self.metrics()
        m["event_definition"] = "latent cosine >= threshold on a new percept after deadline"
        m["hit_cos"] = self.hit_cos
        with self._lock:
            m["recent"] = [
                {k: f[k] for k in ("phrase", "mode", "probability",
                                   "status") if k in f}
                | ({"evidence_cos": f["evidence_cos"]}
                   if "evidence_cos" in f else {})
                for f in self.resolved[-6:]
            ]
        return m

    # ------------------------------------------------------------------
    # persistence — the earned calibration map survives a nap. Open
    # deadlines are stored as remaining seconds (a monotonic-clock
    # timestamp is meaningless across a process restart).
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        now = time.monotonic()
        with self._lock:
            open_ = []
            for fc in self.open:
                remaining = fc["deadline"] - now
                if remaining <= 0:
                    continue                     # would resolve/expire on load
                d = {k: v for k, v in fc.items()
                     if k not in ("vec", "ts", "deadline")}
                d["remaining_s"] = float(remaining)
                d["vec"] = fc["vec"].detach().float().cpu()
                open_.append(d)
            return {
                "open": open_,
                "resolved": [dict(f) for f in self.resolved[-512:]],
                "bins": {m: [list(b) for b in bins]
                         for m, bins in self._bins.items()},
                "next_id": next(self._ids),      # consumes one; fine
            }

    def load_state_dict(self, st: dict) -> None:
        if not st:
            return
        now = time.monotonic()
        with self._lock:
            self.open = []
            for d in st.get("open", []):
                remaining = float(d.get("remaining_s", 0.0))
                if remaining <= 0:
                    continue
                fc = {k: v for k, v in d.items() if k != "remaining_s"}
                fc["ts"] = now
                fc["deadline"] = now + remaining
                if not isinstance(fc.get("vec"), torch.Tensor):
                    continue
                self.open.append(fc)
            self.resolved = [dict(f) for f in st.get("resolved", [])]
            self._bins = {m: [list(b) for b in bins]
                          for m, bins in st.get("bins", {}).items()}
            start = int(st.get("next_id", 1))
            self._ids = itertools.count(max(1, start))
