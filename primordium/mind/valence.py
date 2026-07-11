"""
ValenceCompass — affects discovered from a lived life, and the Hearth.

Nothing here imports human emotion names. Directions are estimated from
the organism's own history: latent motion that co-occurs with drive-error
change (appetite directions, one per drive, permutation-tested), plus the
principal axes of arousal-weighted residual motion (unnamed affects).
A direction is only installed after it proves STABLE across consecutive
re-estimations; each carries a signature (drive correlations, valence),
never a borrowed label.

The Hearth holds latent prototypes of the caregiver — consolidated
automatically during high-social-satisfaction presence, and deliberately
during the imprint ritual. Kinship = closeness of the present moment to
those anchors. Recognition is earned by being raised, not hardcoded.
"""

import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch


class Hearth:
    def __init__(self, dim: int = 384, max_anchors: int = 8):
        self.dim = dim
        self.max_anchors = max_anchors
        self.anchors: List[torch.Tensor] = []
        self._lock = threading.Lock()

    def imprint(self, z: torch.Tensor, weight: float = 1.0) -> None:
        v = z.detach().float().cpu()
        n = float(v.norm())
        if n < 1e-6:
            return
        v = v / n
        with self._lock:
            for i, a in enumerate(self.anchors):
                if float(torch.dot(a, v)) > 0.90:
                    self.anchors[i] = torch.nn.functional.normalize(
                        a * (1 - 0.2 * weight) + v * 0.2 * weight, dim=0)
                    return
            self.anchors.append(v)
            if len(self.anchors) > self.max_anchors:
                self.anchors.pop(0)

    def kinship(self, z: torch.Tensor) -> float:
        v = z.detach().float().cpu()
        n = float(v.norm())
        if n < 1e-6:
            return 0.0
        v = v / n
        with self._lock:
            if not self.anchors:
                return 0.0
            return max(float(torch.dot(a, v)) for a in self.anchors)

    def state_dict(self) -> dict:
        with self._lock:
            return {"anchors": [a.clone() for a in self.anchors]}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            self.anchors = [torch.as_tensor(a).float() for a in
                            st.get("anchors", [])]

    def count(self) -> int:
        with self._lock:
            return len(self.anchors)


def fit_dynamics(series, dt: float):
    """How a feeling actually moves, measured from its own activation
    history: decay from the lag-1 autocorrelation of the projection
    series (AR(1) view: rate = (1-rho)/dt), onset from the size of its
    positive jumps relative to its spread. Returns (onset, decay, meta)
    or None if the series is too flat to say anything honest."""
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 32:
        return None
    x = x - x.mean()
    sd = x.std()
    if sd < 1e-9:
        return None
    rho1 = float(np.clip((x[:-1] * x[1:]).mean() / (sd * sd), 0.0, 0.999))
    decay = float(np.clip((1.0 - rho1) / max(dt, 1e-3) * 0.05, 0.01, 0.5))
    jumps = np.diff(x)
    pos = jumps[jumps > 0]
    onset = float(np.clip((pos.mean() / sd if len(pos) else 0.5) * 0.8,
                          0.05, 0.8))
    return onset, decay, {"rho1": round(rho1, 4), "n": len(x)}


class ValenceCompass:
    DRIVES = ("energy", "competence", "novelty", "social", "vitality")

    def __init__(self, dim: int = 384, buffer_size: int = 20000,
                 min_samples: int = 5000, stability_cos: float = 0.75):
        self.dim = dim
        self.min_samples = min_samples
        self.stability_cos = stability_cos
        self.buf = deque(maxlen=buffer_size)   # (dz, de(5,), surprise, reward)
        self.installed: Dict[str, torch.Tensor] = {}
        self.signatures: Dict[str, dict] = {}
        self.provenance: Dict[str, str] = {}   # name -> "lived" | "innate"
        self._pending: Dict[str, dict] = {}    # name -> {vec, streak}
        self._act_hist: Dict[str, deque] = {}  # lived activation, per affect
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # lived dynamics: the AffectProjector reports each affect's raw
    # projection every perception tick; from that history the affect's
    # onset/decay are FITTED, not authored
    # ------------------------------------------------------------------
    def observe_activation(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._act_hist:
                self._act_hist[name] = deque(maxlen=600)
            self._act_hist[name].append(float(value))

    def fitted_dynamics(self, name: str, dt: float,
                        min_n: int = 200):
        with self._lock:
            hist = list(self._act_hist.get(name, ()))
        if len(hist) < min_n:
            return None
        return fit_dynamics(hist, dt)

    # ------------------------------------------------------------------
    def observe(self, dz: np.ndarray, drive_err_delta: np.ndarray,
                surprise: float, reward: float) -> None:
        with self._lock:
            self.buf.append((dz.astype(np.float16),
                             drive_err_delta.astype(np.float16),
                             np.float16(surprise), np.float16(reward)))

    # ------------------------------------------------------------------
    def estimate(self) -> Dict[str, dict]:
        """Slow-clock re-estimation. Returns candidates {name: {vec, sig}}."""
        with self._lock:
            if len(self.buf) < self.min_samples:
                return {}
            DZ = np.stack([b[0] for b in self.buf]).astype(np.float32)
            DE = np.stack([b[1] for b in self.buf]).astype(np.float32)
            S = np.array([float(b[2]) for b in self.buf], dtype=np.float32)
            R = np.array([float(b[3]) for b in self.buf], dtype=np.float32)

        mu, sd = DZ.mean(0, keepdims=True), DZ.std(0, keepdims=True) + 1e-6
        Z = (DZ - mu) / sd
        rng = np.random.default_rng(0)
        candidates: Dict[str, dict] = {}

        # appetite directions: latent motion co-occurring with drive change
        accepted_vecs = []
        for k, name in enumerate(self.DRIVES):
            e = DE[:, k]
            if e.std() < 1e-5:
                continue
            v = Z.T @ e
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            v = v / nv
            score = float(np.abs((Z @ v) * e).mean())
            null = []
            for _ in range(20):
                ee = rng.permutation(e)
                w = Z.T @ ee
                wn = np.linalg.norm(w) + 1e-9
                null.append(float(np.abs((Z @ (w / wn)) * ee).mean()))
            if score > np.percentile(null, 95):
                candidates[f"drive_{name}"] = {"vec": v}
                accepted_vecs.append(v)

        # unnamed axes: PCA of arousal-weighted residual motion
        Res = Z.copy()
        for v in accepted_vecs:
            Res -= np.outer(Res @ v, v)
        Wt = np.clip(S, 0.5, 4.0)[:, None]
        try:
            _, _, Vt = np.linalg.svd(Res * Wt, full_matrices=False)
            for i in range(min(3, Vt.shape[0])):
                candidates[f"affect_{i}"] = {"vec": Vt[i]}
        except np.linalg.LinAlgError:
            pass

        # signatures: what each direction co-occurs with, and its valence
        for name, c in candidates.items():
            proj = Z @ c["vec"]
            sig = {self.DRIVES[k]: round(float(np.corrcoef(proj, DE[:, k])[0, 1]), 3)
                   if DE[:, k].std() > 1e-6 else 0.0
                   for k in range(len(self.DRIVES))}
            val = (round(float(np.corrcoef(proj, R)[0, 1]), 3)
                   if R.std() > 1e-6 else 0.0)
            c["sig"] = {"drives": sig, "valence": val,
                        "arousal": round(float(np.abs(proj * S).mean()), 3)}
        return candidates

    # ------------------------------------------------------------------
    def stabilize_and_install(self, candidates: Dict[str, dict]) -> List[str]:
        """Install only directions stable across consecutive estimations."""
        newly = []
        with self._lock:
            for name, c in candidates.items():
                v = torch.from_numpy(c["vec"]).float()
                prev = self._pending.get(name)
                if prev is not None:
                    cos = float(torch.dot(
                        torch.nn.functional.normalize(prev["vec"], dim=0),
                        torch.nn.functional.normalize(v, dim=0)))
                    if abs(cos) > self.stability_cos:
                        if cos < 0:
                            v = -v
                        streak = prev["streak"] + 1
                    else:
                        streak = 0
                else:
                    streak = 0
                self._pending[name] = {"vec": v, "streak": streak,
                                       "sig": c["sig"]}
                # stability means SURVIVING re-estimation twice — the
                # docstring always said so; the code now agrees
                if streak >= 2 and name not in self.installed:
                    self.installed[name] = v
                    self.signatures[name] = c["sig"]
                    self.provenance[name] = "lived"
                    newly.append(name)
                elif name in self.installed and streak >= 2:
                    self.installed[name] = v          # track slow drift
                    self.signatures[name] = c["sig"]
        return newly

    # ------------------------------------------------------------------
    # innate priors — weak inherited leanings, annotated and retirable.
    # Unlike biology, every needle here knows where it came from.
    # ------------------------------------------------------------------
    def install_prior(self, name: str, vec: torch.Tensor,
                      sig: Optional[dict] = None) -> None:
        v = torch.nn.functional.normalize(
            torch.as_tensor(vec).float().flatten(), dim=0)
        if v.numel() != self.dim:
            return
        key = name if name.startswith("innate:") else f"innate:{name}"
        with self._lock:
            self.installed[key] = v
            self.signatures[key] = sig or {"origin": "inherited prior"}
            self.provenance[key] = "innate"

    def lived_count(self) -> int:
        with self._lock:
            return sum(1 for n in self.installed
                       if self.provenance.get(n, "lived") == "lived")

    def retire_innate(self) -> List[str]:
        """The training wheels come off: remove every inherited needle."""
        with self._lock:
            gone = [n for n in self.installed
                    if self.provenance.get(n) == "innate"]
            for n in gone:
                self.installed.pop(n, None)
                self.signatures.pop(n, None)
                self.provenance.pop(n, None)
        return gone

    def vectors(self) -> Dict[str, torch.Tensor]:
        with self._lock:
            return dict(self.installed)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "buffered": len(self.buf),
                "installed": [
                    {"id": n, "sig": self.signatures.get(n, {}),
                     "provenance": self.provenance.get(n, "lived"),
                     "stable": True}
                    for n in self.installed
                ],
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {"installed": {n: v.clone() for n, v in self.installed.items()},
                    "signatures": dict(self.signatures),
                    "provenance": dict(self.provenance)}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            self.installed = {n: torch.as_tensor(v).float()
                              for n, v in st.get("installed", {}).items()}
            self.signatures = dict(st.get("signatures", {}))
            self.provenance = {n: st.get("provenance", {}).get(n, "lived")
                               for n in self.installed}
