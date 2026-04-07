"""
Magnum Opus Vitalis - Live Dashboard Server
=============================================
Runs the engine and serves a real-time dashboard at http://localhost:5000

Usage:
    python server.py                        # GPT-2, CPU
    python server.py --model gpt2-medium    # Bigger model
    python server.py --device cuda          # Force GPU
    python server.py --port 8080            # Custom port
"""

import argparse
import json
import os
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from magnum_opus.loader import load_model
from magnum_opus.extraction import extract_vectors
from magnum_opus.engine import MagnumOpusEngine, measure_projections
from magnum_opus.config import EngineConfig

app = Flask(__name__, static_folder=".")
engine: MagnumOpusEngine = None
engine_lock = threading.Lock()
heartbeat_running = False


# ─────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/status")
def api_status():
    with engine_lock:
        st = engine.status()
        snap = engine.emotional_state.snapshot()
        data = {
            "step": st["step"],
            "emotional_state": {
                "fast": snap["fast"],
                "medium": snap["medium"],
                "slow": snap["slow"],
                "blended": snap["blended"],
            },
            "residual_norm": st["residual_norm"],
            "memories": [
                {
                    "id": m.id,
                    "importance": round(m.importance, 4),
                    "connections": m.connections,
                    "access_count": m.access_count,
                    "text_summary": m.text_summary[:60],
                    "age_seconds": round(time.time() - m.timestamp, 1),
                }
                for m in engine.memory.memories[:30]
            ],
            "subconscious": st["subconscious"],
            "alignment": {
                "score": engine.alignment.history[-1].get("alignment_score", 0.8)
                         if engine.alignment.history else 0.8,
                "mutualistic": engine.alignment.history[-1].get("mutualistic_orientation", 0)
                              if engine.alignment.history else 0,
                "stability": engine.alignment.history[-1].get("homeostatic_stability", 1.0)
                             if engine.alignment.history else 1.0,
                "drift": engine.alignment.check_drift(),
            },
            "growth": st["growth"],
            "temporal": st["temporal"],
            "dream_cycles": st["dream_cycles"],
            "conversation_turns": st["conversation_turns"],
            "communicative_drive": st.get("communicative_drive", {}),
        }
    return jsonify(data)


@app.route("/api/stimulate", methods=["POST"])
def api_stimulate():
    body = request.json or {}
    emotion = body.get("emotion", "calm")
    intensity = float(body.get("intensity", 1.0))
    with engine_lock:
        engine.stimulate({emotion: intensity})
        engine._step(dt=0.5)
        snap = engine.emotional_state.get_blended()
    return jsonify({"ok": True, "emotion": emotion, "intensity": intensity, "blended": snap})


@app.route("/api/converse", methods=["POST"])
def api_converse():
    body = request.json or {}
    message = body.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    with engine_lock:
        response = engine.converse(message)
        snap = engine.emotional_state.snapshot()
        alignment = engine.alignment.history[-1] if engine.alignment.history else {}
    return jsonify({
        "response": response,
        "emotional_state": snap["blended"],
        "residual_norm": engine.residual.norm(),
        "alignment_score": alignment.get("alignment_score", 0),
        "memories": len(engine.memory.memories),
    })


@app.route("/api/dream", methods=["POST"])
def api_dream():
    with engine_lock:
        report = engine.dream(verbose=True)
    return jsonify({
        "ok": True,
        "report": {
            "replay": report.get("phases", {}).get("replay", {}),
            "compressed": report.get("phases", {}).get("compress", {}).get("merged", 0),
            "exploration_resonance": report.get("phases", {}).get("explore", {}).get("avg_resonance", 0),
            "connections": report.get("phases", {}).get("reweight", {}).get("connections", 0),
        },
        "memories_after": len(engine.memory.memories),
        "goal_strength": engine.subconscious.goal_strength,
    })


@app.route("/api/autonomous")
def api_autonomous():
    """Poll for messages the system initiated on its own."""
    with engine_lock:
        msgs = engine.get_autonomous_messages()
        drive = engine.communicative_drive.status()
        scenarios = getattr(engine, '_scenario_log', [])[-5:]
    return jsonify({"messages": msgs, "drive": drive, "scenarios": scenarios})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the engine. Optionally keep memories."""
    body = request.json or {}
    keep_memories = body.get("keep_memories", False)
    with engine_lock:
        if keep_memories:
            saved_memories = engine.memory.memories[:]
            saved_prefs = engine.communicative_drive.user_preference
            engine.reset()
            engine.memory.memories = saved_memories
            engine.communicative_drive.user_preference = saved_prefs
        else:
            engine.reset()
    return jsonify({"ok": True, "keep_memories": keep_memories})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.json or {}
    prompt = body.get("prompt", "")
    stimulus = body.get("stimulus", None)
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    with engine_lock:
        if stimulus:
            engine.stimulate(stimulus)
        text = engine.generate_raw(prompt, max_tokens=80)
    return jsonify({"text": text, "blended": engine.emotional_state.get_blended()})


@app.route("/api/save", methods=["POST"])
def api_save():
    filepath = request.json.get("filepath", "engine_state.json") if request.json else "engine_state.json"
    with engine_lock:
        engine.save(filepath)
    return jsonify({"ok": True, "filepath": filepath})


@app.route("/api/load", methods=["POST"])
def api_load():
    filepath = request.json.get("filepath", "engine_state.json") if request.json else "engine_state.json"
    with engine_lock:
        success = engine.load(filepath)
    return jsonify({"ok": success, "filepath": filepath})


# ─────────────────────────────────────────────────────────────────────
# HEARTBEAT - the engine's pulse
# ─────────────────────────────────────────────────────────────────────

def heartbeat_loop():
    """
    Background thread that keeps the engine alive.
    Ticks every 500ms. A living system is never flat:
    - Homeostatic drift toward emotional baselines
    - Subconscious micro-stimulations (background neural noise)
    - Residual decay (temporal continuity fading)
    - Communicative drive accumulation (urge to speak)
    - Autonomous speech when pressure exceeds patience
    """
    global heartbeat_running
    heartbeat_running = True
    tick = 0
    scenario_energy = 0.0     # Accumulates from resonance, triggers imagination
    scenario_threshold = 1.5  # Fires when enough internal pressure builds

    while heartbeat_running:
        time.sleep(0.5)

        with engine_lock:
            # Homeostatic decay
            engine.emotional_state.decay_step(dt=0.5)

            # Subconscious background noise
            blended = engine.emotional_state.get_blended()
            noise = engine.subconscious.generate_noise(blended)

            # Micro-stimulations from noise projections
            # These create visible fluctuations around baselines
            for emo in engine.emotional_state.names:
                if emo in engine.emotion_vectors:
                    vec = engine.emotion_vectors[emo]
                    proj = torch.dot(noise.cpu().float(), vec.cpu().float()).item()
                    # Low threshold + meaningful influence = visible background activity
                    if abs(proj) > 0.02:
                        engine.emotional_state.stimulate(emo, proj * 0.25)

            # Direct micro-jitter: tiny random nudges to simulate neural noise
            # This ensures the bars always have slight movement
            for emo in engine.emotional_state.names:
                jitter = np.random.normal(0, 0.02)
                engine.emotional_state.fast[emo] += jitter
                engine.emotional_state.fast[emo] = float(np.clip(
                    engine.emotional_state.fast[emo], -1.0, 1.0
                ))

            # Update residual
            engine.compute_steering_vector()

            # Subconscious: evaluate resonance and update goals EVERY tick
            # Project noise against emotion vectors to get resonance
            # (cheap computation, no forward pass needed)
            tick += 1
            resonance = 0.0
            for emo_name, emo_vec in engine.subconscious.emotion_vectors.items():
                proj = torch.dot(noise.cpu().float(), emo_vec.cpu().float()).item()
                # Weight by current emotional activation (resonance is stronger
                # when noise aligns with active emotions)
                emo_activation = abs(blended.get(emo_name, 0))
                resonance += abs(proj) * (1.0 + emo_activation * 2.0)
            resonance /= max(len(engine.subconscious.emotion_vectors), 1)

            # Record resonance so the dashboard can see it
            engine.subconscious.resonance_history.append(resonance)
            if len(engine.subconscious.resonance_history) > 200:
                engine.subconscious.resonance_history = engine.subconscious.resonance_history[-200:]

            # Update goals: lower threshold so goals actually crystallize
            # during background processing, not just during conversation
            engine.subconscious.update_goals(noise, resonance, threshold=0.3)

            # Every ~15 seconds, do a stronger exploration burst
            # (like a random thought surfacing)
            if tick % 30 == 0:
                burst_noise = engine.subconscious.generate_noise(blended)
                burst_noise = burst_noise * 2.0  # Stronger than normal
                burst_resonance = 0.0
                for emo_name, emo_vec in engine.subconscious.emotion_vectors.items():
                    proj = torch.dot(burst_noise.cpu().float(), emo_vec.cpu().float()).item()
                    burst_resonance += abs(proj) * (1.0 + abs(blended.get(emo_name, 0)) * 3.0)
                burst_resonance /= max(len(engine.subconscious.emotion_vectors), 1)
                engine.subconscious.update_goals(burst_noise, burst_resonance, threshold=0.2)

            # ── SCENARIO GENERATION (Layer 2+3 from the blog) ──
            # Triggered by accumulated subconscious energy, NOT by a timer.
            # The system imagines a future when it has enough internal pressure,
            # not when a clock says to. More emotional activity = more frequent
            # imagination. Calm resting state = rare imagination.
            scenario_energy += resonance * 0.1 + engine.subconscious.goal_strength * 0.05
            scenario_energy *= 0.98  # Natural decay prevents runaway

            if scenario_energy > scenario_threshold:
                scenario_energy = 0.0  # Reset after firing
                try:
                    # Build a scenario steering vector from noise + current goals
                    scenario_steering = noise * 0.5
                    if engine.subconscious.goal_strength > 0.01:
                        goal_dir = engine.subconscious.goal_vector / max(
                            engine.subconscious.goal_vector.norm().item(), 1e-8
                        )
                        scenario_steering = scenario_steering + goal_dir * 0.5

                    # Pick a seed based on what the system is feeling
                    dominant_emo = max(blended.items(), key=lambda x: abs(x[1]))
                    if abs(dominant_emo[1]) > 0.1:
                        seed = f"Imagining what might happen next, considering {dominant_emo[0]}:"
                    else:
                        seed = "Thinking about what could happen next:"

                    # Run a SHORT generation with the scenario steering
                    # This is the model imagining a possible future
                    scenario_input = engine.tokenizer(
                        seed, return_tensors="pt"
                    ).to(engine.device)

                    engine.hook.set_steering(
                        scenario_steering.to(engine.device)
                    )
                    engine.hook.clear()

                    with torch.no_grad():
                        scenario_ids = engine.model.generate(
                            scenario_input["input_ids"],
                            max_new_tokens=20,
                            temperature=0.9,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=engine.tokenizer.eos_token_id,
                        )

                    engine.hook.set_steering(None)
                    scenario_states = engine.hook.captured_states

                    # Evaluate the imagined scenario emotionally
                    if scenario_states:
                        last = scenario_states[-1].mean(dim=1).squeeze(0).float()
                        positive_resonance = 0.0
                        negative_resonance = 0.0

                        for emo_name, emo_vec in engine.subconscious.emotion_vectors.items():
                            proj = torch.dot(
                                last.to(emo_vec.device), emo_vec.float()
                            ).item()

                            # Positive emotions = desirable future
                            if emo_name in ("joy", "calm", "curious", "trust"):
                                positive_resonance += max(0, proj)
                            # Negative emotions = undesirable future
                            elif emo_name in ("desperate", "anger", "fear", "disgust"):
                                negative_resonance += max(0, proj)

                        # If the imagined future is emotionally positive,
                        # reinforce the direction that led to it
                        net_valence = positive_resonance - negative_resonance * 0.5
                        if net_valence > 0:
                            engine.subconscious.update_goals(
                                scenario_steering, abs(net_valence) * 0.5,
                                threshold=0.0  # Always reinforce positive futures
                            )
                        else:
                            # Suppress directions that lead to negative futures
                            engine.subconscious.goal_vector *= 0.9

                        # Log the scenario for monitoring
                        scenario_text = engine.tokenizer.decode(
                            scenario_ids[0], skip_special_tokens=True
                        )
                        if not hasattr(engine, '_scenario_log'):
                            engine._scenario_log = []
                        engine._scenario_log.append({
                            "timestamp": time.time(),
                            "seed": seed,
                            "scenario": scenario_text[len(seed):].strip()[:100],
                            "positive": round(positive_resonance, 3),
                            "negative": round(negative_resonance, 3),
                            "net_valence": round(net_valence, 3),
                            "reinforced": net_valence > 0,
                        })
                        if len(engine._scenario_log) > 20:
                            engine._scenario_log = engine._scenario_log[-20:]

                except Exception as e:
                    pass  # Don't crash the heartbeat on scenario errors

            # Tick communicative drive
            engine.communicative_drive.tick(
                blended,
                engine.subconscious.goal_strength,
                engine.residual.norm(),
                dt=0.5,
            )

            # Autonomous speech: system initiates when pressure demands it
            if engine.communicative_drive.should_speak():
                try:
                    msg = engine.speak_autonomously()
                    if msg:
                        print(f"  [AUTONOMOUS] {msg[:100]}")
                except Exception as e:
                    print(f"  [AUTONOMOUS ERROR] {e}")

            # Auto-dream after 5 minutes idle with enough memories
            if (engine.idle_seconds() > 300
                    and len(engine.memory.memories) >= 3
                    and not engine.is_dreaming()):
                engine.dream_async()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    global engine, heartbeat_running

    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis - Dashboard Server")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--steering-strength", type=float, default=4.0)
    parser.add_argument("--load-state", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  MAGNUM OPUS VITALIS - LIVE DASHBOARD")
    print("=" * 60)

    model, tokenizer, device = load_model(args.model, args.device)

    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    else:
        n_layers = model.config.num_hidden_layers
    target_layer = args.layer if args.layer is not None else n_layers // 2

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n  Extracting emotion vectors...")
    vectors = extract_vectors(model, tokenizer, target_layer, device)

    config = EngineConfig(steering_strength=args.steering_strength)
    engine = MagnumOpusEngine(
        model, tokenizer, vectors,
        target_layer=target_layer, config=config, device=device,
    )

    if args.load_state:
        engine.load(args.load_state)

    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  Model: {args.model} | Device: {device} | Layer: {target_layer}")
    print("  All systems active. Engine heartbeat running.")
    print("  The system is alive. Open the URL in your browser.\n")

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        heartbeat_running = False


if __name__ == "__main__":
    main()
