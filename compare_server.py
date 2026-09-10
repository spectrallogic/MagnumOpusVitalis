"""
Magnum Opus Vitalis — A/B Comparison Server
=============================================
Side-by-side comparison: raw model vs engine-steered model.
Same input, same model, same sampling — only difference is the engine.

Usage:
    python compare_server.py
    python compare_server.py --model gpt2-medium --port 5001
    python run.py                            # select a local model; prepare its profile
"""

import json
import math
import threading
import time
import traceback
import uuid
import webbrowser

from flask import Flask, Response, jsonify, render_template, request, redirect
from werkzeug.serving import make_server

from magnum_opus_v2 import (
    V2Engine,
    load_model,
)
from magnum_opus_v2.startup import (
    argument_parser, choose_model, check_engine, prepare_profile, validate_model_boundary,
)
from magnum_opus_v2.model_sources import discover_cached_models


def _json_error(exc: Exception):
    """Return a JSON 500 with the full traceback for the browser to surface."""
    tb = traceback.format_exc()
    msg = f"{type(exc).__name__}: {exc}"
    print(f"  [API ERROR] {msg}\n{tb}", flush=True)
    return jsonify({"error": msg, "traceback": tb}), 500


app = Flask(__name__)

engine: V2Engine = None
raw_history: list = []
engine_history: list = []
_session_id = uuid.uuid4().hex
_broadcast_lock = threading.Lock()


def _json_safe(value):
    """Represent missing/unbounded telemetry as null, never invalid JSON Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _snapshot():
    state = engine.snapshot()
    if hasattr(engine, "bus"):
        state["latent_bands"] = engine.bus.visual_bands()
    state["runtime"] = {"session_id": _session_id,
                        "model": engine.profile.model_name if engine.profile else "unknown",
                        "device": str(engine.device), "server_time": time.time()}
    state["last_reply_context"] = engine._last_causes
    return _json_safe(state)


@app.route("/")
def index():
    return render_template("observatory.html")


@app.route("/compare")
def comparison():
    return render_template("compare.html")


@app.route("/face")
def face():
    return render_template("voxel.html")


@app.route("/face2d")
def face2d():
    # the 2D plate face was a strict subset of the voxel face; retired.
    return redirect("/face")


@app.route("/api/talk", methods=["POST"])
def talk():
    """Single-column conversation for the face experience: engine only,
    returns the reply stripped of the prompt plus a fresh snapshot."""
    try:
        data = request.json
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        max_tokens = int(data.get("max_tokens", 80))

        reply = engine.converse(user_input, max_new_tokens=max_tokens)
        engine_history.append({"user": user_input, "assistant": reply})

        return jsonify({
            "response": reply,
            "state": _snapshot(),
            "context": _json_safe(engine._last_causes),
            "turn": len(engine_history),
        })
    except Exception as e:
        return _json_error(e)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Send the same input to both raw model and engine, return both responses."""
    try:
        data = request.json
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        max_tokens = data.get("max_tokens", 100)

        # Use a shared seed so both columns see the same sampling noise.
        seed = int(time.time()) % 10000

        # Both calls return the reply only. generate_raw gets the raw
        # column's own history (chat template when the model has one).
        raw_response = engine.generate_raw(
            user_input, max_new_tokens=max_tokens, seed=seed,
            history=raw_history,
        )
        engine_response = engine.converse(
            user_input, max_new_tokens=max_tokens, seed=seed,
        )

        raw_history.append({"user": user_input, "assistant": raw_response})
        engine_history.append({"user": user_input, "assistant": engine_response})

        return jsonify({
            "raw_response": raw_response,
            "engine_response": engine_response,
            "engine_state": _snapshot(),
            # what shaped this reply, frozen at emission (reply <- cause)
            "engine_causes": _json_safe(engine._last_causes),
            "turn": len(engine_history),
        })
    except Exception as e:
        return _json_error(e)


@app.route("/api/stream")
def stream():
    """Server-sent events: full engine snapshot at ~5Hz plus any autonomous
    speech, so the dashboard breathes at the substrate's pace instead of
    polling every 2 seconds."""
    try:
        resume_id = max(0, int(request.headers.get("Last-Event-ID", "0")))
    except ValueError:
        resume_id = 0

    def gen():
        # start at the present — a fresh connection shows the mind's
        # thinking from now on, never an invented past
        latest = engine.journal.latest_id()
        last_jid = resume_id if 0 < resume_id <= latest else max(0, latest - 40)
        while True:
            try:
                payload = _snapshot()
                payload["turn"] = len(engine_history)
                with _broadcast_lock:
                    payload["autonomous"] = engine.drain_autonomous_messages()
                    for message in payload["autonomous"]:
                        engine.journal.emit("autonomous_reply", reply=message)
                engine.maybe_log_emotion(payload)     # honest, change-triggered
                payload["journal"] = engine.journal.since(last_jid, limit=120)
                if payload["journal"]:
                    last_jid = payload["journal"][-1]["id"]
                yield f"id: {last_jid}\ndata: {json.dumps(_json_safe(payload), allow_nan=False)}\n\n"
            except GeneratorExit:
                return
            except Exception as e:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            time.sleep(0.2)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/status")
def status():
    """Get full engine snapshot for live dashboard updates."""
    try:
        s = _snapshot()
        s["turn"] = len(engine_history)
        return jsonify(s)
    except Exception as e:
        return _json_error(e)


@app.route("/api/conversation")
def conversation():
    """Restore the current engine conversation without triggering generation."""
    with engine._history_lock:
        messages = [dict(m) for m in engine.chat_history[-40:]]
    return jsonify({"messages": messages, "session_id": _session_id})


@app.route("/api/journal")
def journal():
    """The cognition journal since a given id — for timeline reconnect
    backfill. Starts empty for a caller with no id."""
    try:
        since = int(request.args.get("since", engine.journal.latest_id()))
        return jsonify({"events": engine.journal.since(since, limit=500),
                        "latest_id": engine.journal.latest_id()})
    except Exception as e:
        return _json_error(e)


@app.route("/api/memory")
def memory():
    """Browsable recent memory traces (kept off the 5Hz snapshot)."""
    try:
        return jsonify({"traces": engine.memory.recent(200)})
    except Exception as e:
        return _json_error(e)


@app.route("/api/autonomous")
def get_autonomous():
    """Drain queued autonomous messages for the UI."""
    try:
        msgs = engine.drain_autonomous_messages()
        return jsonify({"messages": msgs})
    except Exception as e:
        return _json_error(e)


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset engine soft state and conversation histories."""
    global raw_history, engine_history
    try:
        engine.reset()
        raw_history = []
        engine_history = []
        return jsonify({"status": "reset"})
    except Exception as e:
        return _json_error(e)


@app.route("/api/export")
def export():
    """Export full conversation data as JSON."""
    try:
        return jsonify({
            "raw_history": raw_history,
            "engine_history": engine_history,
            "engine_state": _snapshot(),
            "timestamp": time.time(),
        })
    except Exception as e:
        return _json_error(e)


def main(argv=None):
    global engine
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.list_models:
        choices = discover_cached_models()
        for item in choices:
            print(f"{item['label']}\n  {item['source']}")
        if not choices:
            print("No cached causal LMs found. Use --model with a Transformers checkpoint folder.")
        return 0
    try:
        source = choose_model(args.model)
        model, tokenizer, device = load_model(
            source, args.device, local_files_only=not args.download,
            trust_remote_code=args.trust_remote_code,
        )
        validate_model_boundary(model, tokenizer, device)
        profile, rebuilt = prepare_profile(
            source, (model, tokenizer, device), profiles_dir=args.profiles_dir,
            profile_path=args.profile_path, rebuild=args.rebuild_profile,
        )
        if args.resume and rebuilt:
            raise ValueError("The profile was recalibrated. Omit --resume to start fresh state; the previous checkpoint has been retained.")
        engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
        if args.resume:
            if engine.load_state():
                print("  Resumed saved runtime state.")
            else:
                print("  No saved runtime state; starting fresh.")
        if args.check:
            check_engine(engine)
            return 0

        # Bind successfully before starting background workers or opening a browser.
        with make_server(args.host, args.port, app, threaded=True) as httpd:
            engine.start()
            try:
                browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
                if ":" in browser_host:
                    browser_host = f"[{browser_host}]"
                url = f"http://{browser_host}:{args.port}/"
                print(f"\n  Vitalis is ready: {url}", flush=True)
                print("  Stop with Ctrl+C. First-run setup is saved for next time.", flush=True)
                if not args.no_browser:
                    try:
                        webbrowser.open(url)
                    except webbrowser.Error:
                        print(f"  Open the dashboard manually: {url}")
                httpd.serve_forever()
            finally:
                save_path = None
                if args.resume:
                    from magnum_opus_v2.persistence import default_path
                    save_path = default_path(source)
                engine.stop(save_path=save_path)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n  Stopped.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        hint = ""
        if isinstance(exc, OSError) and not args.download:
            hint = "\nUse a complete local Transformers folder, or add --download to allow fetching the selected Hub model."
        parser.exit(2, f"\nVitalis could not start: {exc}{hint}\n")


if __name__ == "__main__":
    raise SystemExit(main())
