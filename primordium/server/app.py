"""
Serving Primordium: Flask for the page, `websockets` for the senses.

Uplink (browser -> organism): binary frames tagged by first byte —
0x01 camera JPEG, 0x02 microphone PCM — plus small JSON control messages
(mute, imprint, hello, chat). Downlink: 5 Hz state JSON assembled from
the worker and every region, PNGs of reverie/dreams/mind (0x81–0x84),
the Easel when IT repaints (0x85), and Pulse event batches — the honest
thought traffic the data-block view renders 1:1. The first connection
that asks to be "embodied" feeds the senses; everyone else is a
read-only window. Both servers bind 127.0.0.1 — nothing leaves.
"""

import asyncio
import json
import struct
import threading

import cv2
import numpy as np
from flask import Flask, render_template

import websockets


def _canvas_png(fb: np.ndarray) -> bytes:
    """Easel framebuffer (RGB uint8) -> PNG bytes."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(fb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


class OrganismServer:
    def __init__(self, cfg, ctx, board):
        self.cfg = cfg
        self.ctx = ctx
        self.board = board
        self.flask = Flask(__name__)
        self._embodied_ws = None
        self._loop = None
        # birth canvas so a fresh window sees its body, not a blank
        self._canvas_png = (_canvas_png(ctx.easel.view())
                            if getattr(ctx, "easel", None) is not None
                            else None)

        @self.flask.route("/")
        def index():
            return render_template("organism.html",
                                   ws_port=cfg.ws_port,
                                   run_name=cfg.run_name)

        @self.flask.route("/health")
        def health():
            pub = ctx.worker.get_published() if ctx.worker else {}
            return {"ok": True, "tick": pub.get("tick", 0),
                    "hz": pub.get("hz", 0)}

    # ------------------------------------------------------------------
    def state_payload(self) -> dict:
        ctx = self.ctx
        pub = ctx.worker.get_published() if ctx.worker else {}
        board = self.board.all()
        try:
            bus_snap = ctx.bus.snapshot()
        except Exception:  # noqa: BLE001
            bus_snap = {}
        payload = {
            "t": "state",
            "tick": pub.get("tick", 0),
            "hz": pub.get("hz", 0),
            "stage": pub.get("stage", 0),
            "acuity": pub.get("stage_label", ""),
            "drives": ctx.drives.snapshot(),
            "tide": {"levels": ctx.neuromod.snapshot(),
                     "causes": ctx.neuromod.causes()},
            "affects": ctx.compass.snapshot().get("installed", []),
            "affect_acts": board.get("affect_acts", {}),
            "kinship": round(float(board.get("kinship", 0.0)), 3),
            "imprints": ctx.imprint.count(),
            "surprise": pub.get("surprise", 1.0),
            "loss": pub.get("loss"),
            "lp": pub.get("lp", 0),
            "latent_std": pub.get("latent_std", 0),
            "novelty": pub.get("novelty", 0),
            "vitality": pub.get("vitality", 0),
            "sleep": ctx.sleep.snapshot(),
            "voice": pub.get("voice", {}),
            "bus": bus_snap,
            "self": ctx.self_model.snapshot() if ctx.self_model else {},
            "exec_pressure": ctx.executive.snapshot().get("pressure", 0),
            "situation": ctx.situation.snapshot() if ctx.situation else {},
            "chronicle": ctx.chronicle.snapshot(),
            "dev": pub.get("dev", {}),
            "presence": board.get("presence", {}),
            "events": pub.get("events", []),
            "replays": pub.get("replays", 0),
            "decoder_mse": pub.get("decoder_mse"),
            "dreams": ctx.reverie.snapshot(),
            "minds_eye": bool(getattr(ctx.worker, "minds_eye_on", False)),
            "mind_mel": pub.get("mind_mel") if getattr(
                ctx.worker, "minds_eye_on", False) else None,
            "wordstream": (ctx.wordstream.snapshot()
                           if getattr(ctx, "wordstream", None) else {}),
            "easel": (ctx.easel.snapshot()
                      if getattr(ctx, "easel", None) else {}),
            "gates": (ctx.gatehouse.snapshot()
                      if getattr(ctx, "gatehouse", None) else {}),
            "fringe": pub.get("fringe", {}),
            "bus_provenance": ctx.bus.provenance(),
            "anatomy": pub.get("anatomy", {}),
            "cap_gates": pub.get("cap_gates", {}),
            "reach": pub.get("reach", {}),
            "watch": pub.get("watch", {}),
            "wheel": pub.get("wheel", {}),
            "grip": pub.get("grip", {}),
            "gaze": pub.get("gaze", {}),
        }
        return payload

    # ------------------------------------------------------------------
    def handle_chat(self, text: str) -> bool:
        """A human typed to it. Text enters the Wordstream like any
        other voice in the room; Pulse records the real arrival."""
        text = (text or "").strip()[:500]
        if not text or getattr(self.ctx, "wordstream", None) is None:
            return False
        target = None
        teacher = getattr(self.ctx, "word_teacher", None)
        if teacher is not None and teacher.ok:
            target = teacher.embed_text(text)
        self.ctx.wordstream.push(text, source="human", target_vec=target)
        if getattr(self.ctx, "pulse", None) is not None:
            self.ctx.pulse.emit("chat_in", "WORLD", "SENSE",
                                chars=len(text))
        return True

    # ------------------------------------------------------------------
    async def _ws_handler(self, ws):
        embodied = False
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    if not embodied or len(msg) < 5:
                        continue
                    tag = msg[0]
                    payload = msg[5:]
                    if tag == 0x01:
                        self.ctx.retina.offer(payload)
                    elif tag == 0x02:
                        pcm = np.frombuffer(payload, dtype="<i2")
                        self.ctx.cochlea.push(pcm)
                else:
                    try:
                        d = json.loads(msg)
                    except Exception:  # noqa: BLE001
                        continue
                    t = d.get("t")
                    if t == "hello":
                        if d.get("embodied") and self._embodied_ws is None:
                            self._embodied_ws = ws
                            embodied = True
                            # its voice lives in this browser now
                            self.ctx.voice.set_muted(False)
                        await ws.send(json.dumps(
                            {"t": "welcome", "embodied": embodied}))
                        asyncio.ensure_future(self._downlink(ws))
                    elif t == "mute":
                        self.ctx.voice.set_muted(bool(d.get("v", True)))
                    elif t == "imprint":
                        self.board.put(imprint_hold=bool(d.get("on", False)))
                    elif t == "minds_eye":
                        self.ctx.worker.set_minds_eye(bool(d.get("v", False)))
                    elif t == "chat":
                        self.handle_chat(d.get("text", ""))
        finally:
            if self._embodied_ws is ws:
                self._embodied_ws = None
                self.board.put(imprint_hold=False)

    async def _downlink(self, ws) -> None:
        last_img = None
        last_dream = None
        last_canvas = None
        pulse = getattr(self.ctx, "pulse", None)
        # start from now: a new window watches the present, not the past
        last_pulse = pulse.latest_id() if pulse else 0
        try:
            while True:
                await asyncio.sleep(0.2)
                try:
                    await ws.send(json.dumps(self.state_payload()))
                except websockets.ConnectionClosed:
                    return
                if pulse is not None:
                    evs = pulse.since(last_pulse, limit=150)
                    if evs:
                        last_pulse = evs[-1]["id"]
                        await ws.send(json.dumps(
                            {"t": "pulse", "events": evs}))
                easel = getattr(self.ctx, "easel", None)
                if easel is not None:
                    fb = easel.take_repaint()
                    if fb is not None:
                        self._canvas_png = _canvas_png(fb)
                    if self._canvas_png and self._canvas_png is not last_canvas:
                        last_canvas = self._canvas_png
                        await ws.send(struct.pack("B", 0x85) + last_canvas)
                pub = self.ctx.worker.get_published() if self.ctx.worker else {}
                img = pub.get("imagination_png")
                if img and img is not last_img:
                    last_img = img
                    await ws.send(struct.pack("B", 0x81) + img)
                mind = pub.get("mind_png")
                if (mind and getattr(self.ctx.worker, "minds_eye_on", False)
                        and mind is not getattr(self, "_last_mind", None)):
                    self._last_mind = mind
                    await ws.send(struct.pack("B", 0x84) + mind)
                frames = self.ctx.reverie.frames()
                if frames and frames[-1] is not last_dream:
                    last_dream = frames[-1]
                    await ws.send(struct.pack("B", 0x83) + frames[-1])
                jpeg = self.ctx.retina.latest_jpeg()
                if jpeg:
                    await ws.send(struct.pack("B", 0x82) + jpeg)
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------
    def start(self) -> None:
        def ws_main():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def serve():
                async with websockets.serve(
                        self._ws_handler, self.cfg.host, self.cfg.ws_port,
                        max_size=2 ** 20):
                    await asyncio.Future()

            try:
                self._loop.run_until_complete(serve())
            except Exception as e:  # noqa: BLE001
                print(f"[ws] server stopped: {e}")

        threading.Thread(target=ws_main, name="primordium-ws",
                         daemon=True).start()

    def run_flask(self) -> None:
        self.flask.run(host=self.cfg.host, port=self.cfg.http_port,
                       debug=False)
