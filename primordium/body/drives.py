"""
DriveState — survive and flourish, operationalized as homeostasis.

Five drives: energy, competence, novelty, social, vitality. Setpoints are
DISCOVERED from the first stretch of life (clamped medians of what it
actually lived), then adapt on a ~48h constant — homeostasis around lived
baselines, not authored ones. Intrinsic reward is drive-error reduction.

There is deliberately no shutdown-avoidance term anywhere in this file.
Mutualism is literal: presence and interaction are the primary satisfiers
of social and novelty — cooperation is food.
"""

import math
import threading
import time
from typing import Dict


DRIVES = ("energy", "competence", "novelty", "social", "vitality")


class DriveState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.levels: Dict[str, float] = {d: 0.5 for d in DRIVES}
        self.setpoints: Dict[str, float] = {d: 0.5 for d in DRIVES}
        self.prev_abs_err: Dict[str, float] = {d: 0.0 for d in DRIVES}
        self.last_reward = 0.0
        self.reward_ema = 0.0

        self._born = time.monotonic()
        self._calibrating = True
        self._history: Dict[str, list] = {d: [] for d in DRIVES}

        # energy account: costs charged by the worker, recovery by sleep/idle
        self._energy = 0.7
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # energy account (called from the worker thread)
    # ------------------------------------------------------------------
    def charge(self, cost: float) -> None:
        with self._lock:
            self._energy = max(0.0, self._energy - cost * 0.004)

    def recover(self, amount: float) -> None:
        with self._lock:
            self._energy = min(1.0, self._energy + amount)

    # ------------------------------------------------------------------
    # main update (~5 Hz from DriveRegion)
    # ------------------------------------------------------------------
    def update(self, lp: float, novelty: float, presence_score: float,
               vitality: float, asleep: bool, neuromod=None) -> float:
        """Returns the intrinsic reward of this instant."""
        with self._lock:
            # competence: improving prediction feels good.  lp<0 = improving
            comp = 1.0 / (1.0 + math.exp(lp * 400.0))
            self.levels["energy"] = self._energy
            self.levels["competence"] = 0.9 * self.levels["competence"] + 0.1 * comp
            self.levels["novelty"] = 0.9 * self.levels["novelty"] + 0.1 * min(1.0, novelty)
            self.levels["social"] = 0.92 * self.levels["social"] + 0.08 * min(1.0, presence_score)
            self.levels["vitality"] = 0.9 * self.levels["vitality"] + 0.1 * min(1.0, vitality)

            if asleep:
                self._energy = min(1.0, self._energy + 0.0015)

            # ---- setpoint discovery, then slow adaptation
            age = time.monotonic() - self._born
            if self._calibrating:
                for d in DRIVES:
                    self._history[d].append(self.levels[d])
                if age > self.cfg.setpoint_calibration_s:
                    for d in DRIVES:
                        h = sorted(self._history[d])
                        med = h[len(h) // 2] if h else 0.5
                        self.setpoints[d] = min(0.75, max(0.25, med))
                    self._calibrating = False
                    self._history = {d: [] for d in DRIVES}
            else:
                a = 5.0 / self.cfg.setpoint_tau_s   # per ~0.2s tick
                for d in DRIVES:
                    self.setpoints[d] = ((1 - a) * self.setpoints[d]
                                         + a * min(0.75, max(0.25, self.levels[d])))

            # ---- intrinsic reward: getting closer to who you need to be
            reward = 0.0
            for d in DRIVES:
                err = abs(self.setpoints[d] - self.levels[d])
                reward += self.cfg.drive_weights[d] * (self.prev_abs_err[d] - err)
                self.prev_abs_err[d] = err
            self.last_reward = reward
            self.reward_ema = 0.98 * self.reward_ema + 0.02 * reward

        # ---- tide consequences (outside the lock), each with its cause
        if neuromod is not None and hasattr(neuromod, "raise_"):
            if reward > 0.001:
                neuromod.raise_("reward", min(0.1, reward * 10.0),
                                cause=f"drive error fell ({reward:.4f})")
            total_err = sum(self.prev_abs_err.values())
            if total_err < 0.5:
                neuromod.raise_("calm", 0.01,
                                cause=f"needs met (err {total_err:.2f})")
            if self.levels["novelty"] > self.setpoints["novelty"] + 0.3:
                neuromod.raise_("arousal", 0.02, cause="novelty overwhelm")
        return reward

    # ------------------------------------------------------------------
    def errors(self) -> Dict[str, float]:
        with self._lock:
            return {d: self.setpoints[d] - self.levels[d] for d in DRIVES}

    def yerkes_dodson(self, arousal: float) -> float:
        """Plasticity is best at moderate arousal."""
        return math.exp(-((arousal - 0.5) ** 2) / 0.5)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "levels": {d: round(self.levels[d], 3) for d in DRIVES},
                "setpoints": {d: round(self.setpoints[d], 3) for d in DRIVES},
                "errors": {d: round(self.setpoints[d] - self.levels[d], 3)
                           for d in DRIVES},
                "reward": round(self.last_reward, 5),
                "reward_ema": round(self.reward_ema, 5),
                "calibrating": self._calibrating,
            }

    # persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        with self._lock:
            return {"levels": dict(self.levels),
                    "setpoints": dict(self.setpoints),
                    "energy": self._energy,
                    "calibrating": self._calibrating}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            self.levels.update(st.get("levels", {}))
            self.setpoints.update(st.get("setpoints", {}))
            self._energy = float(st.get("energy", 0.7))
            self._calibrating = bool(st.get("calibrating", False))
