"""
v2 dashboard — Flask server that lets you actually interact with V2Engine.

Mirrors v1's compare_server shape: two columns (raw model vs v2-engine
output), plus live readouts of bus / neuromod / limbic / subconscious /
memory / executive that auto-refresh while the system is alive between
turns.

Run:
    python -m magnum_opus_v2.server                  # gpt2, port 5001
    python -m magnum_opus_v2.server --model gpt2 --port 5001
"""

import argparse
import threading
import time
import traceback
from collections import deque

import torch
from flask import Flask, jsonify, render_template, request

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile, profile_exists, create_profile

from magnum_opus_v2 import V2Engine, V2Config, ClockConfig


# ---------------------------------------------------------------------------
# Globals (single-process Flask is fine for this; the engine is the heavy
# state). All access goes through the engine's internal locks.
# ---------------------------------------------------------------------------
engine: V2Engine = None
raw_model = None
raw_tokenizer = None
raw_device = "cpu"

# Conversation history (display only — engine has its own state)
history = deque(maxlen=50)

# Autonomous speech queue (callback enqueues; UI polls and drains)
autonomous_queue = deque(maxlen=20)


def make_autonomous_callback():
    """Edge-trigger from Executive — generate now and enqueue text."""
    def cb():
        try:
            text = engine.speak_autonomously(max_new_tokens=25)
            autonomous_queue.append({"text": text, "t": time.monotonic()})
        except Exception:
            traceback.print_exc()
    return cb


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="../templates",  # reuse top-level templates/ dir
)


@app.route("/")
def index():
    return render_template("v2_compare.html")


@app.route("/api/snapshot")
def api_snapshot():
    if engine is None:
        return jsonify({"error": "engine not loaded"}), 503
    snap = engine.snapshot()
    snap["history"] = list(history)
    snap["autonomous_pending"] = len(autonomous_queue)
    return jsonify(snap)


@app.route("/api/converse", methods=["POST"])
def api_converse():
    if engine is None:
        return jsonify({"error": "engine not loaded"}), 503
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400
    max_tokens = int(data.get("max_tokens", 50))

    # Raw model run (no steering) for the comparison column
    raw_text = ""
    try:
        torch.manual_seed(42)
        enc = raw_tokenizer(prompt, return_tensors="pt").to(raw_device)
        with torch.no_grad():
            out = raw_model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=True,
                top_p=0.92, temperature=0.9,
                pad_token_id=raw_tokenizer.eos_token_id,
            )
        raw_text = raw_tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        raw_text = f"[raw model error: {e}]"

    # Engine run (steered by live bus)
    engine_text = ""
    try:
        engine_text = engine.converse(prompt, max_new_tokens=max_tokens, seed=42)
    except Exception as e:
        traceback.print_exc()
        engine_text = f"[engine error: {e}]"

    history.append({
        "prompt": prompt,
        "raw": raw_text,
        "engine": engine_text,
        "t": time.monotonic(),
    })
    return jsonify({
        "raw": raw_text,
        "engine": engine_text,
        "snapshot": engine.snapshot(),
    })


@app.route("/api/autonomous")
def api_autonomous():
    """Drain pending autonomous utterances (the UI polls this)."""
    out = list(autonomous_queue)
    autonomous_queue.clear()
    return jsonify({"messages": out})


@app.route("/api/stimulate", methods=["POST"])
def api_stimulate():
    """Manual emotion stimulation for testing the dashboard."""
    if engine is None:
        return jsonify({"error": "engine not loaded"}), 503
    data = request.get_json(force=True) or {}
    stim = data.get("emotions") or {}
    if not isinstance(stim, dict):
        return jsonify({"error": "emotions must be an object"}), 400
    engine.limbic.stimulate_many(stim, neuromod=engine.neuromod)
    return jsonify({"ok": True, "snapshot": engine.snapshot()})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset history and a few engine fields. Doesn't recreate the engine."""
    history.clear()
    autonomous_queue.clear()
    if engine is not None:
        # Zero the bus + velocity, clear the memory pool, reset emotional state.
        with engine.bus._lock:  # noqa: SLF001
            engine.bus.state.zero_()
            engine.bus.velocity.zero_()
            engine.bus.tick_count = 0
        engine.memory.pool.clear()
        # Re-init Limbic emotional state at baselines
        for emo, cfg in engine.limbic._state.configs.items():  # noqa: SLF001
            engine.limbic._state.fast[emo] = cfg.baseline       # noqa: SLF001
            engine.limbic._state.medium[emo] = cfg.baseline     # noqa: SLF001
            engine.limbic._state.slow[emo] = cfg.baseline       # noqa: SLF001
        engine.executive.pressure = 0.0
        engine.executive._wall_at_speech = 0.0      # noqa: SLF001
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-default-mode", action="store_true")
    ap.add_argument("--no-sparks", action="store_true")
    ap.add_argument("--slow-dt", type=float, default=5.0,
                    help="Slow clock period in seconds (lower = faster neuromod)")
    args = ap.parse_args()

    global engine, raw_model, raw_tokenizer, raw_device

    print(f"  Loading {args.model}...")
    raw_model, raw_tokenizer, raw_device = load_model(args.model)
    if not profile_exists(args.model):
        print(f"  No profile — creating one for {args.model}...")
        profile = create_profile(args.model, device=raw_device)
    else:
        profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}")

    cfg = V2Config(
        hidden_dim=profile.hidden_dim, device=raw_device,
        clock=ClockConfig(slow_dt_seconds=args.slow_dt),
    )

    engine = V2Engine.from_profile(
        model=raw_model, tokenizer=raw_tokenizer, profile=profile,
        device=raw_device, config=cfg,
        enable_default_mode=not args.no_default_mode,
        enable_knowledge_sparks=not args.no_sparks,
        on_should_speak=make_autonomous_callback(),
    )
    engine.start()
    print(f"  Engine started. Regions: "
          f"{[r.name for r in engine.flow.regions]}")
    print(f"  Open http://{args.host}:{args.port}/")

    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
