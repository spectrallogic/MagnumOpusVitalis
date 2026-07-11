"""
Gaze — attention as an action. It chooses where to look.

Until now the organism received whatever the camera gave, whole.
The Gaze makes vision a motor act: a movable, zoomable window onto the
source frame, driven by a measured reflex loop —

- CHASE ERROR: vision is tokenized on a patch grid, so the world-model's
  per-patch prediction error IS a map of where the world defies it. The
  gaze drifts toward the error's center of mass, and zooms in when the
  error is concentrated, out when it is diffuse.
- EXPLORE: smooth OU noise keeps the eye from freezing.
- THE BOREDOM VALVE: chasing error that yields no learning progress
  (the noisy-TV trap — unlearnable noise is maximally "interesting")
  builds a stuck timer; past it, the gaze releases back to the whole
  scene. Stare, fail to learn, look away. This is a mitigation, not a
  solution, and the README says so.

Movement costs energy (motion is metabolic), the gaze state enters
proprioception (efference copy — it can learn what its own looking
does to what it sees), and zoom can never pass 1.0: pulling back to
the full frame is always available, so nothing is ever hidden from it
by its own eye. The policy is procedural v0, like the voice reflex;
the learning lives in the world model predicting THROUGH its own eye
movements.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class Gaze:
    def __init__(self, cfg):
        self.cfg = cfg
        self.x = 0.0                 # center offset, [-1, 1]
        self.y = 0.0
        self.zoom = 1.0              # 1.0 = whole scene; floor cfg.gaze_zoom_min
        self._noise = np.zeros(2, dtype=np.float64)
        self._stuck_s = 0.0          # time spent staring without learning
        self._hold_s = 0.0           # post-release refractory: stay wide
        self.releases = 0            # boredom valve firings
        self.interrupts = 0          # cross-modal releases (the Watch)
        self.saccades = 0            # shifts big enough to count
        self.last_shift = 0.0

    # ------------------------------------------------------------------
    def crop_params(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.zoom)

    def efference(self) -> np.ndarray:
        """What the eye is doing, for proprioception: [x, y, zoom]."""
        return np.array([self.x, self.y, self.zoom], dtype=np.float32)

    # ------------------------------------------------------------------
    def update(self, err_grid: np.ndarray, lp: float,
               dt: float) -> Optional[Dict]:
        """One reflex step from the measured error map. Returns a report
        when the shift was a real saccade, else None."""
        cfg = self.cfg
        g = err_grid.shape[0]
        # only CONTRAST attracts: early life has a high uniform error
        # floor everywhere, and a center-of-mass over the raw map would
        # just point at the middle forever
        rel = err_grid - float(err_grid.min())
        total = float(rel.sum())

        tx, ty, tzoom = 0.0, 0.0, 1.0
        if total > 1e-9:
            w = rel / total
            ys, xs = np.mgrid[0:g, 0:g]
            # error center-of-mass in VIEW coordinates: the map is of
            # what it currently sees, so the world-target is the view
            # offset scaled by how much world the view spans (= zoom)
            vx = float((w * xs).sum()) / max(g - 1, 1) * 2.0 - 1.0
            vy = float((w * ys).sum()) / max(g - 1, 1) * 2.0 - 1.0
            tx = float(np.clip(self.x + vx * self.zoom, -1, 1))
            ty = float(np.clip(self.y + vy * self.zoom, -1, 1))
            conc = float(rel.max()) / (float(rel.mean()) + 1e-9)
            tzoom = float(np.clip(1.0 - cfg.gaze_zoom_rate * (conc - 1.0),
                                  cfg.gaze_zoom_min, 1.0))

        # the boredom valve: hard staring that teaches nothing lets go —
        # and STAYS let-go for a refractory hold, or it would snap
        # straight back to the same unlearnable spot next tick
        if self._hold_s > 0.0:
            self._hold_s = max(0.0, self._hold_s - dt)
            tx, ty, tzoom = 0.0, 0.0, 1.0
            self._stuck_s = 0.0
        else:
            if self.zoom < 0.95 and abs(lp) < cfg.gaze_boredom_lp:
                self._stuck_s += dt
            else:
                self._stuck_s *= 0.5
            if self._stuck_s > cfg.gaze_release_s:
                tx, ty, tzoom = 0.0, 0.0, 1.0
                self.releases += 1
                self._stuck_s = 0.0
                self._hold_s = cfg.gaze_release_hold_s

        self._noise = 0.9 * self._noise + \
            np.random.randn(2) * cfg.gaze_noise
        ox, oy, oz = self.x, self.y, self.zoom
        c = cfg.gaze_chase
        self.x = float(np.clip(ox + c * (tx - ox) + self._noise[0], -1, 1))
        self.y = float(np.clip(oy + c * (ty - oy) + self._noise[1], -1, 1))
        self.zoom = float(np.clip(oz + c * (tzoom - oz),
                                  cfg.gaze_zoom_min, 1.0))

        shift = abs(self.x - ox) + abs(self.y - oy) + abs(self.zoom - oz)
        self.last_shift = shift
        if shift >= cfg.gaze_shift_emit:
            self.saccades += 1
            return {"x": round(self.x, 3), "y": round(self.y, 3),
                    "zoom": round(self.zoom, 3), "shift": round(shift, 3)}
        return None

    # ------------------------------------------------------------------
    def interrupt(self) -> bool:
        """Cross-modal release: something not-visual demands attention.
        A startle is a SNAP, not smooth pursuit — the eye opens to the
        whole scene at once and holds there (same refractory hold the
        boredom valve uses). Returns True if it actually let go."""
        if self.zoom >= 0.95 and abs(self.x) < 0.1 and abs(self.y) < 0.1:
            return False                     # already looking at everything
        self.x *= 0.2
        self.y *= 0.2
        self.zoom = 1.0
        self._hold_s = self.cfg.gaze_release_hold_s
        self._stuck_s = 0.0
        self.interrupts += 1
        return True

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3),
                "zoom": round(self.zoom, 3), "saccades": self.saccades,
                "releases": self.releases, "interrupts": self.interrupts,
                "stuck_s": round(self._stuck_s, 2)}

    def state_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "zoom": self.zoom,
                "saccades": self.saccades, "releases": self.releases}

    def load_state_dict(self, st: dict) -> None:
        self.x = float(st.get("x", 0.0))
        self.y = float(st.get("y", 0.0))
        self.zoom = float(st.get("zoom", 1.0))
        self.saccades = int(st.get("saccades", 0))
        self.releases = int(st.get("releases", 0))
