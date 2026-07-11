"""
Foster — the parent-side handle on the caregiver process.

Spawns foster_proc as a subprocess (JSON lines over stdio), reads its
replies on a daemon thread, restarts it with backoff if it dies, and
surfaces death honestly (`alive`, `restarts`) instead of pretending.
All RATE LIMITING lives here, on the organism's side of the boundary:
the child answers every observe it receives; the parent decides how
often the room gets to speak.
"""

import json
import queue
import subprocess
import sys
import threading
import time
from typing import Optional


class Foster:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self.ready = False
        self.restarts = 0
        self._last_spawn = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        args = [sys.executable, "-u", "-m", "primordium.caregiver.foster_proc",
                "--model", self.cfg.caregiver_model,
                "--max-chars", str(self.cfg.caregiver_max_chars)]
        if self.cfg.caregiver_cpu:
            args.append("--cpu")
        if getattr(self.cfg, "caregiver_stub", False):
            args.append("--stub")
        self._last_spawn = time.monotonic()
        self.ready = False
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            bufsize=1)
        self._reader = threading.Thread(target=self._read_loop,
                                        name="foster-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        proc = self.proc
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if msg.get("t") == "ready":
                    self.ready = True
                elif msg.get("t") == "say":
                    self._replies.put(msg)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def maybe_restart(self, backoff_s: float = 30.0) -> bool:
        """Watchdog: returns True if it just restarted the caregiver."""
        if self.alive or self.proc is None:
            return False
        if time.monotonic() - self._last_spawn < backoff_s:
            return False
        self.restarts += 1
        self.start()
        return True

    def observe(self, babble: str, state: dict) -> bool:
        if not self.alive:
            return False
        try:
            self.proc.stdin.write(json.dumps(
                {"t": "observe", "babble": babble, "state": state}) + "\n")
            self.proc.stdin.flush()
            return True
        except Exception:  # noqa: BLE001
            return False

    def poll_reply(self) -> Optional[dict]:
        try:
            return self._replies.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self.proc = None
        self.ready = False
