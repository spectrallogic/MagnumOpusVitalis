"""
Magnum Opus Vitalis — A/B Comparison Server
=============================================
Side-by-side comparison: raw model vs engine-steered model.
Same input, same model, same parameters — only difference is the engine.

Usage:
    python compare_server.py
    python compare_server.py --model gpt2-medium --port 5001
"""

import argparse
import json
import time

import torch
from flask import Flask, jsonify, render_template, request

from magnum_opus import MagnumOpusEngine, load_model, extract_vectors, load_profile
from magnum_opus.profile import profile_exists

app = Flask(__name__)

# Globals (initialized in main)
engine: MagnumOpusEngine = None
raw_history: list = []
engine_history: list = []


@app.route("/")
def index():
    return render_template("compare.html")


@app.route("/api/compare", methods=["POST"])
def compare():
    """Send the same input to both raw model and engine, return both responses."""
    data = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    max_tokens = data.get("max_tokens", 100)

    # Build prompt with conversation context (last 3 turns for both)
    def build_prompt(history, msg):
        lines = []
        for turn in history[-3:]:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        lines.append(f"User: {msg}")
        lines.append("Assistant:")
        return "\n".join(lines)

    raw_prompt = build_prompt(raw_history, user_input)
    engine_prompt = build_prompt(engine_history, user_input)

    # Raw generation (no steering, no state changes)
    torch.manual_seed(int(time.time()) % 10000)
    raw_text = engine.generate_without_steering(raw_prompt, max_tokens=max_tokens)
    if "Assistant:" in raw_text:
        raw_response = raw_text.split("Assistant:")[-1].strip()
    else:
        raw_response = raw_text[len(raw_prompt):].strip()

    # Engine generation (full steering)
    torch.manual_seed(int(time.time()) % 10000)
    engine_response = engine.converse(user_input, max_tokens=max_tokens)

    # Track histories
    raw_history.append({"user": user_input, "assistant": raw_response})
    engine_history.append({"user": user_input, "assistant": engine_response})

    # Get engine state for display
    status = engine.status()

    return jsonify({
        "raw_response": raw_response,
        "engine_response": engine_response,
        "engine_state": {
            "emotions": status["emotional_state"]["blended"],
            "residual_norm": status["residual_norm"],
            "memory_count": status["memory"]["count"],
            "subconscious_goal": status["subconscious"]["goal_strength"],
            "step": status["step"],
            "dream_cycles": status["dream_cycles"],
        },
        "turn": len(engine_history),
    })


@app.route("/api/status")
def status():
    """Get current engine state."""
    s = engine.status()
    return jsonify({
        "emotions": s["emotional_state"]["blended"],
        "residual_norm": s["residual_norm"],
        "memory_count": s["memory"]["count"],
        "subconscious_goal": s["subconscious"]["goal_strength"],
        "step": s["step"],
        "dream_cycles": s["dream_cycles"],
        "alignment": s["alignment"],
        "turn": len(engine_history),
    })


@app.route("/api/dream", methods=["POST"])
def dream():
    """Trigger a dream cycle."""
    result = engine.dream(verbose=False)
    return jsonify({"status": "complete", "cycle": engine.status()["dream_cycles"]})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset engine and conversation histories."""
    global raw_history, engine_history
    engine.reset()
    raw_history = []
    engine_history = []
    return jsonify({"status": "reset"})


@app.route("/api/export")
def export():
    """Export full conversation data as JSON."""
    return jsonify({
        "raw_history": raw_history,
        "engine_history": engine_history,
        "engine_status": engine.status(),
        "timestamp": time.time(),
    })


def main():
    global engine

    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis A/B Comparison")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--profile", action="store_true",
                        help="Use saved profile (auto-creates if missing)")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    print("=" * 50)
    print("  Magnum Opus Vitalis — Research Compare")
    print("=" * 50)

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Create engine (from profile or fresh extraction)
    if args.profile:
        if profile_exists(args.model):
            print(f"  Loading saved profile for {args.model}...")
            profile = load_profile(args.model)
        else:
            from magnum_opus import create_profile
            print(f"  No profile found — creating one for {args.model}...")
            profile = create_profile(args.model, device=device)
        engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)
    else:
        n_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
        target_layer = n_layers // 2
        vectors = extract_vectors(model, tokenizer, target_layer=target_layer, device=device)
        engine = MagnumOpusEngine(model, tokenizer, vectors, target_layer=target_layer, device=device)

    print(f"\n  Server starting at http://{args.host}:{args.port}")
    print(f"  Model: {args.model} | Device: {device}")
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
