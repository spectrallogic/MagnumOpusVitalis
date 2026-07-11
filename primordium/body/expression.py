"""
Expression organs — Wordstream and Easel, routed by urge.

Wordstream: a text sense and a keyboard motor. Incoming text (caregiver,
human chat) and its OWN emitted characters share one conversation stream
consumed a few bytes per tick — it reads the room including itself, and
the proprioceptive typing flag is how it learns which words were its.
Expect keyboard-babble first; that is the honest starting point of every
writer.

Easel: a small framebuffer that is part of its body. The paint motor
lays soft strokes when the urge gate opens; the canvas feeds back in as
a sensory modality, so it sees what it made. It repaints when IT
chooses; the dashboard only mirrors it.

ExpressionRouter: the Executive's single pressure-to-act fans out into
voice, typing, and painting impulses — one urge, three mouths.
"""

import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

PAD = 256
MSG_END = 257


class Wordstream:
    def __init__(self, cfg):
        self.cfg = cfg
        self._stream: deque = deque(maxlen=4096)   # (byte, is_self)
        self._pending: deque = deque()             # not yet consumed
        self.outbox: deque = deque(maxlen=64)      # chars it typed, unsent
        self._typed_recent: deque = deque(maxlen=160)  # UI mirror, non-destructive
        self.n_messages = 0
        self.chars_typed_total = 0
        self.last_in_at = 0.0
        self.last_typed_at = 0.0
        self.gate_until = 0.0
        self.transcript: deque = deque(maxlen=40)  # (source, text) for UI
        self._current_target: Optional[np.ndarray] = None  # MiniLM vec
        self._lock = threading.Lock()

    # ---- inputs -------------------------------------------------------
    def push(self, text: str, source: str = "human",
             target_vec: Optional[np.ndarray] = None) -> None:
        data = text.encode("utf-8", errors="replace")[:512]
        with self._lock:
            for b in data:
                self._pending.append((b, False))
            self._pending.append((MSG_END, False))
            self.n_messages += 1
            self.last_in_at = time.monotonic()
            self.transcript.append((source, text[:200]))
            if target_vec is not None:
                self._current_target = target_vec

    def echo_self(self, byte: int) -> None:
        """Its own keystroke enters the same stream it reads."""
        with self._lock:
            self._pending.append((int(byte) & 0xFF, True))

    # ---- per-tick consumption ----------------------------------------
    def consume(self) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Returns (txt_ids (text_tokens,) int64, self_fraction,
        active word-target vec or None)."""
        n = self.cfg.text_tokens
        ids = np.full(n, PAD, dtype=np.int64)
        selfish = 0.0
        with self._lock:
            got = 0
            while got < n and self._pending:
                b, is_self = self._pending.popleft()
                ids[got] = b
                selfish += 1.0 if is_self else 0.0
                self._stream.append((b, is_self))
                got += 1
            target = self._current_target if got > 0 else None
            if not self._pending:
                self._current_target = None
        return ids, (selfish / max(got, 1) if got else 0.0), target

    # ---- typing motor -------------------------------------------------
    def typing_impulse(self) -> None:
        with self._lock:
            self.gate_until = time.monotonic() + self.cfg.phonation_gate_s

    def gate_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self.gate_until

    def type_chars(self, byte_ids: List[int]) -> str:
        """Record emitted keystrokes: outbox for Foster/UI + self-echo."""
        out = []
        with self._lock:
            now = time.monotonic()
            for b in byte_ids:
                b = int(b) & 0xFF
                self.outbox.append(b)
                self._typed_recent.append(b)
                out.append(b)
                self.chars_typed_total += 1
                self.last_typed_at = now
        for b in out:
            self.echo_self(b)
        try:
            return bytes(out).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    def drain_outbox(self) -> str:
        with self._lock:
            if not self.outbox:
                return ""
            s = bytes(self.outbox).decode("utf-8", errors="replace")
            self.outbox.clear()
            return s

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "messages_in": self.n_messages,
                "chars_typed": self.chars_typed_total,
                "pending": len(self._pending),
                "typing_gate": time.monotonic() < self.gate_until,
                "typed_recent": bytes(self._typed_recent).decode(
                    "utf-8", errors="replace"),
                "transcript": [
                    {"source": s, "text": t} for s, t in list(self.transcript)[-8:]
                ],
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {"n_messages": self.n_messages,
                    "chars_typed": self.chars_typed_total}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            self.n_messages = int(st.get("n_messages", 0))
            self.chars_typed_total = int(st.get("chars_typed", 0))


class Easel:
    def __init__(self, cfg):
        self.cfg = cfg
        r = cfg.canvas_res
        self.fb = np.full((r, r, 3), 16, dtype=np.uint8)   # near-black birth
        self.strokes_total = 0
        self.repainted = False           # set on stroke; server clears on send
        self.gate_until = 0.0
        self._lock = threading.Lock()

    def paint_impulse(self) -> None:
        with self._lock:
            self.gate_until = time.monotonic() + self.cfg.phonation_gate_s

    def gate_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self.gate_until

    def stroke(self, params: np.ndarray) -> bool:
        """params (8,) in [-1,1]: [gate,x,y,radius,r,g,b,alpha].
        Lays one soft circular blot. Returns True if painted."""
        if params[0] <= 0.0 or not self.gate_open():
            return False
        r = self.cfg.canvas_res
        x = int((params[1] + 1) / 2 * (r - 1))
        y = int((params[2] + 1) / 2 * (r - 1))
        rad = 1.0 + (params[3] + 1) / 2 * 5.0
        color = ((params[4:7] + 1) / 2 * 255.0)
        alpha = 0.2 + (params[7] + 1) / 2 * 0.7
        yy, xx = np.ogrid[:r, :r]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        soft = np.exp(-d2 / (2 * rad * rad))[..., None]      # (r, r, 1)
        with self._lock:
            fb = self.fb.astype(np.float32)
            self.fb = np.clip(
                fb * (1 - soft * alpha) + color[None, None, :] * soft * alpha,
                0, 255).astype(np.uint8)
            self.strokes_total += 1
            self.repainted = True
        return True

    def view(self) -> np.ndarray:
        with self._lock:
            return self.fb.copy()

    def take_repaint(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self.repainted:
                return None
            self.repainted = False
            return self.fb.copy()

    def snapshot(self) -> dict:
        with self._lock:
            return {"strokes": self.strokes_total,
                    "paint_gate": time.monotonic() < self.gate_until}

    def state_dict(self) -> dict:
        with self._lock:
            return {"fb": self.fb.copy(), "strokes": self.strokes_total}

    def load_state_dict(self, st: dict) -> None:
        with self._lock:
            fb = st.get("fb")
            if fb is not None and fb.shape == self.fb.shape:
                self.fb = np.asarray(fb, dtype=np.uint8)
            self.strokes_total = int(st.get("strokes", 0))


class ExpressionRouter:
    """One urge, three mouths: Executive pressure fans out."""

    def __init__(self, voice, wordstream, easel):
        self.voice = voice
        self.wordstream = wordstream
        self.easel = easel

    def impulse(self) -> None:
        self.voice.vocal_impulse()
        self.wordstream.typing_impulse()
        self.easel.paint_impulse()
