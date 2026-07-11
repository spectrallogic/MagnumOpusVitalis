"""
Magnum Opus Vitalis — A/B Comparison Server
=============================================
Side-by-side comparison: raw model vs engine-steered model.
Same input, same model, same sampling — only difference is the engine.

Usage:
    python compare_server.py
    python compare_server.py --model gpt2-medium --port 5001
    python compare_server.py --profile        # use saved profile (auto-create if missing)
"""

import argparse
import json
import time
import traceback

from flask import Flask, Response, jsonify, render_template, request

from magnum_opus_v2 import (
    V2Engine,
    load_model,
    load_profile,
    create_profile,
    profile_exists,
)


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


@app.route("/")
def index():
    return render_template("compare.html")


@app.route("/face")
def face():
    return render_template("voxel.html")


@app.route("/face2d")
def face2d():
    return render_template("face.html")


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
            "state": engine.snapshot(),
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
            "engine_state": engine.snapshot(),
            "turn": len(engine_history),
        })
    except Exception as e:
        return _json_error(e)


@app.route("/api/stream")
def stream():
    """Server-sent events: full engine snapshot at ~5Hz plus any autonomous
    speech, so the dashboard breathes at the substrate's pace instead of
    polling every 2 seconds."""
    def gen():
        while True:
            try:
                payload = engine.snapshot()
                payload["turn"] = len(engine_history)
                payload["autonomous"] = engine.drain_autonomous_messages()
                yield f"data: {json.dumps(payload)}\n\n"
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
        s = engine.snapshot()
        s["turn"] = len(engine_history)
        return jsonify(s)
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
            "engine_state": engine.snapshot(),
            "timestamp": time.time(),
        })
    except Exception as e:
        return _json_error(e)


def main():
    global engine

    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis A/B Comparison")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model name (use gpt2 for quick tests)")
    parser.add_argument("--profile", action="store_true",
                        help="Use saved profile (auto-creates if missing)")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    print("=" * 50)
    print("  Magnum Opus Vitalis — Research Compare")
    print("=" * 50)

    model, tokenizer, device = load_model(args.model)

    if args.profile:
        if profile_exists(args.model):
            print(f"  Loading saved profile for {args.model}...")
            profile = load_profile(args.model)
        else:
            print(f"  No profile found — creating one for {args.model}...")
            profile = create_profile(args.model, device=device)
    else:
        print(f"  Creating fresh profile for {args.model}...")
        profile = create_profile(args.model, device=device)

    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
    engine.start()
    print("  V2 substrate started (FlowRunner active)")

    print(f"\n  Server starting at http://{args.host}:{args.port}")
    print(f"  A/B research view:   http://{args.host}:{args.port}/")
    print(f"  Face experience:     http://{args.host}:{args.port}/face")
    print(f"  Model: {args.model} | Device: {device}")
    print()

    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
