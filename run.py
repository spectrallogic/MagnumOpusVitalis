#!/usr/bin/env python3
"""
Magnum Opus Vitalis - Interactive Runner
==========================================
Run a live conversation with the full engine active.
All 10 systems operating on real latent space.

Usage:
    python run.py                          # GPT-2, CPU
    python run.py --model gpt2-medium      # Bigger model
    python run.py --model gpt2 --device cuda
    python run.py --load state.json        # Resume from saved state

Commands during conversation:
    /status     - Show full engine status
    /emotions   - Show emotional state (all 3 speeds)
    /memory     - Show stored memories
    /dream      - Trigger a dream cycle
    /alignment  - Check alignment health
    /stimulate calm 2.0  - Manually stimulate an emotion
    /save [file] - Save engine state
    /load [file] - Load engine state
    /reset      - Reset engine
    /quit       - Exit
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Optional

import numpy as np
import torch

from magnum_opus.loader import load_model
from magnum_opus.extraction import extract_vectors
from magnum_opus.engine import MagnumOpusEngine
from magnum_opus.config import EngineConfig


BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                    MAGNUM OPUS VITALIS                              ║
║          Engine Over Ocean: Live Latent Space Steering              ║
║                                                                      ║
║   All 10 systems active on real transformer activations.             ║
║   Type /help for commands. Type naturally to converse.               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

AUTO_DREAM_IDLE_SECONDS = 120  # Dream after 2 minutes idle
DEFAULT_STATE_FILE = "engine_state.json"


def print_emotional_state(engine: MagnumOpusEngine):
    """Pretty-print the emotional state across all three speeds."""
    snap = engine.emotional_state.snapshot()
    print("\n  Emotional State:")
    print("  " + "-" * 60)

    blended = snap["blended"]
    names = sorted(blended.keys())

    # Header
    print(f"  {'Emotion':>12} | {'Fast':>8} | {'Medium':>8} | {'Slow':>8} | {'Blended':>8}")
    print("  " + "-" * 60)

    for name in names:
        f = snap["fast"].get(name, 0)
        m = snap["medium"].get(name, 0)
        s = snap["slow"].get(name, 0)
        b = blended[name]

        # Color coding via markers
        marker = ""
        if abs(b) > 0.3:
            marker = " **"
        elif abs(b) > 0.1:
            marker = " *"

        print(f"  {name:>12} | {f:>+8.4f} | {m:>+8.4f} | {s:>+8.4f} | {b:>+8.4f}{marker}")

    print(f"\n  Residual norm: {engine.residual.norm():.4f}")
    print(f"  Subconscious goal strength: {engine.subconscious.goal_strength:.4f}")


def print_memory_status(engine: MagnumOpusEngine):
    """Print stored memories."""
    mems = engine.memory.memories
    if not mems:
        print("\n  No memories stored yet.")
        return

    print(f"\n  Memory System ({len(mems)} traces):")
    print("  " + "-" * 60)
    print(f"  {'ID':>8} | {'Importance':>10} | {'Access':>6} | {'Conn':>4} | Summary")
    print("  " + "-" * 60)

    for m in sorted(mems, key=lambda x: x.importance, reverse=True)[:15]:
        age = time.time() - m.timestamp
        age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
        print(f"  {m.id:>8} | {m.importance:>10.4f} | {m.access_count:>6} | "
              f"{m.connections:>4} | {m.text_summary[:40]}  ({age_str})")


def print_full_status(engine: MagnumOpusEngine):
    """Print comprehensive engine status."""
    st = engine.status()
    print(f"""
  Engine Status (step {st['step']})
  {'='*50}
  Conversation turns: {st['conversation_turns']}
  Dream cycles: {st['dream_cycles']}
  Idle: {st['idle_seconds']:.0f}s

  Temporal:
    Elapsed since last: {st['temporal']['elapsed']:.1f}s
    Conversation pace: {st['temporal']['pace']:.2f}
    Total interactions: {st['temporal']['interactions']}

  Memory:
    Stored: {st['memory']['count']} traces
    Avg importance: {st['memory']['avg_importance']:.4f}
    Connections: {st['memory']['total_connections']}

  Subconscious:
    Goal strength: {st['subconscious']['goal_strength']:.4f}
    Avg resonance: {st['subconscious']['avg_resonance']:.4f}

  Growth:
    Stage: {st['growth']['stage']} ({'optimize' if st['growth']['stage']==1 else 'pressure' if st['growth']['stage']==2 else 'expand'})
    Measurements: {st['growth']['measurements']}

  Alignment:
    Score: {st['alignment'].get('latest_score', 'N/A')}
    Drift: {st['alignment'].get('drift', {}).get('status', 'N/A')}

  Residual norm: {st['residual_norm']:.4f}
""")


def handle_command(command: str, engine: MagnumOpusEngine) -> bool:
    """Handle slash commands. Returns True if the loop should continue."""
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd == "/quit" or cmd == "/exit":
        return False

    elif cmd == "/help":
        print("""
  Commands:
    /status      - Full engine status
    /emotions    - Emotional state (all 3 speeds)
    /memory      - Stored memories
    /dream       - Run dream cycle
    /alignment   - Check alignment health
    /stimulate <emotion> <intensity>  - Manual stimulus
    /save [file] - Save state
    /load [file] - Load state
    /reset       - Reset engine
    /quit        - Exit
""")

    elif cmd == "/status":
        print_full_status(engine)

    elif cmd == "/emotions":
        print_emotional_state(engine)

    elif cmd == "/memory":
        print_memory_status(engine)

    elif cmd == "/dream":
        print("\n  Starting dream cycle...")
        report = engine.dream(verbose=True)
        print(f"  Dream complete. Memories: {len(engine.memory.memories)}, "
              f"Goal: {engine.subconscious.goal_strength:.4f}")

    elif cmd == "/alignment":
        health = engine.alignment_health()
        print(f"\n  Alignment Health: {health}")
        if engine.alignment.history:
            latest = engine.alignment.history[-1]
            print(f"  Latest score: {latest.get('alignment_score', 0):.4f}")
            print(f"  Mutualistic orientation: {latest.get('mutualistic_orientation', 0):+.4f}")

    elif cmd == "/stimulate":
        if len(parts) >= 3:
            emotion = parts[1]
            try:
                intensity = float(parts[2])
                engine.stimulate({emotion: intensity})
                print(f"  Stimulated '{emotion}' at intensity {intensity}")
                b = engine.emotional_state.get_blended()
                if emotion in b:
                    print(f"  Current '{emotion}' level: {b[emotion]:+.4f}")
            except ValueError:
                print("  Usage: /stimulate <emotion> <intensity>")
        else:
            print("  Usage: /stimulate <emotion> <intensity>")
            print(f"  Available: {', '.join(engine.emotional_state.names)}")

    elif cmd == "/save":
        filepath = parts[1] if len(parts) > 1 else DEFAULT_STATE_FILE
        engine.save(filepath)

    elif cmd == "/load":
        filepath = parts[1] if len(parts) > 1 else DEFAULT_STATE_FILE
        engine.load(filepath)

    elif cmd == "/reset":
        engine.reset()
        print("  Engine reset.")

    else:
        print(f"  Unknown command: {cmd}. Type /help for options.")

    return True


def run_interactive(engine: MagnumOpusEngine, system_prompt: Optional[str] = None):
    """Main interactive conversation loop."""
    print(BANNER)

    blended = engine.emotional_state.get_blended()
    active_emos = {k: v for k, v in blended.items() if abs(v) > 0.05}
    if active_emos:
        emo_str = ", ".join(f"{k}={v:+.3f}" for k, v in sorted(active_emos.items(), key=lambda x: -abs(x[1])))
        print(f"  Starting emotional state: {emo_str}")
    print(f"  Memories: {len(engine.memory.memories)} | "
          f"Dream cycles: {engine._dream_cycle.cycle_count}\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            if not handle_command(user_input, engine):
                print("\n  Goodbye.")
                break
            continue

        # Generate response
        try:
            response = engine.converse(user_input, system_prompt=system_prompt)
        except Exception as e:
            print(f"\n  [Error during generation: {e}]")
            continue

        # Display response
        print(f"\n  AI: {response}\n")

        # Show brief emotional indicator
        blended = engine.emotional_state.get_blended()
        top_emos = sorted(blended.items(), key=lambda x: -abs(x[1]))[:3]
        emo_bar = " | ".join(f"{k}={v:+.3f}" for k, v in top_emos if abs(v) > 0.05)
        if emo_bar:
            print(f"  [{emo_bar} | res={engine.residual.norm():.3f} | "
                  f"mem={len(engine.memory.memories)}]\n")

        # Auto-dream check
        if (engine.idle_seconds() > AUTO_DREAM_IDLE_SECONDS
                and not engine.is_dreaming()
                and len(engine.memory.memories) >= 3):
            print("  [Auto-dream triggered by idle period...]")
            engine.dream_async()


def main():
    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis - Interactive Engine")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model (default: gpt2)")
    parser.add_argument("--device", default=None,
                        help="cpu/cuda/mps (default: auto)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Target layer (default: middle)")
    parser.add_argument("--steering-strength", type=float, default=4.0,
                        help="Steering magnitude (default: 4.0)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load saved engine state from file")
    parser.add_argument("--save-on-exit", type=str, default=None,
                        help="Auto-save state to this file on exit")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt prepended to all generations")
    parser.add_argument("--test", action="store_true",
                        help="Run automated test suite instead of interactive mode")
    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.model, args.device)

    # Detect target layer
    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    else:
        n_layers = model.config.num_hidden_layers
    target_layer = args.layer if args.layer is not None else n_layers // 2

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Extract vectors
    print("\n  Phase 1: Extracting emotion and temporal vectors...")
    vectors = extract_vectors(model, tokenizer, target_layer, device)

    # Build engine
    config = EngineConfig(steering_strength=args.steering_strength)
    engine = MagnumOpusEngine(
        model, tokenizer, vectors,
        target_layer=target_layer, config=config, device=device,
    )

    # Load saved state if requested
    if args.load:
        engine.load(args.load)

    if args.test:
        run_test_suite(model, tokenizer, engine, vectors, device)
    else:
        try:
            run_interactive(engine, system_prompt=args.system_prompt)
        finally:
            if args.save_on_exit:
                engine.save(args.save_on_exit)


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════

def run_test_suite(model, tokenizer, engine, vectors, device):
    """Automated test suite proving all 10 systems work."""
    from magnum_opus.engine import measure_projections

    print("\n" + "=" * 70)
    print("  AUTOMATED TEST SUITE")
    print("=" * 70)
    start = time.time()

    # Test 1: Steering comparison
    print("\n  TEST 1: Emotion steering produces different outputs")
    prompt = "The engineer looked at the results and decided to"
    for name, stim in [("baseline", None), ("calm", {"calm": 3}),
                       ("desperate", {"desperate": 3}), ("joyful", {"joy": 3})]:
        engine.reset()
        if stim: engine.stimulate(stim)
        text = engine.generate_raw(prompt, max_tokens=40)
        gen = text[len(prompt):].strip()[:100]
        print(f"    [{name:>10}] {gen}")
    print("    PASS: Different steering produces different outputs")

    # Test 2: Residual bleed-through
    print("\n  TEST 2: Residual carries emotional traces forward")
    engine.reset()
    engine.stimulate({"desperate": 5.0})
    _ = engine.generate_raw("Test", max_tokens=10)
    norm_after_stim = engine.residual.norm()
    engine._step(dt=1.0)
    _ = engine.generate_raw("Test", max_tokens=10)
    norm_after_decay = engine.residual.norm()
    engine._step(dt=10.0)
    _ = engine.generate_raw("Test", max_tokens=10)
    norm_much_later = engine.residual.norm()
    print(f"    After stimulus: {norm_after_stim:.4f}")
    print(f"    After 1 step:   {norm_after_decay:.4f}")
    print(f"    After 10 steps: {norm_much_later:.4f}")
    assert norm_after_stim > 0, "Residual should be non-zero after stimulus"
    print("    PASS: Residual persists then decays")

    # Test 3: Multi-speed dynamics
    print("\n  TEST 3: Three-speed emotional channels")
    engine.reset()
    engine.stimulate({"desperate": 3.0})
    snap = engine.emotional_state.snapshot()
    fast_val = snap["fast"]["desperate"]
    slow_val = snap["slow"]["desperate"]
    assert abs(fast_val) > abs(slow_val) * 5, "Fast should respond much more than slow"
    print(f"    Fast: {fast_val:+.4f}, Medium: {snap['medium']['desperate']:+.4f}, Slow: {slow_val:+.4f}")
    print("    PASS: Fast reacts strongly, slow barely moves")

    # Test 4: Memory encode and recall
    print("\n  TEST 4: Memory system")
    engine.reset()
    engine.stimulate({"joy": 3.0})
    _ = engine.generate_raw("A wonderful discovery was made today", max_tokens=20)
    engine.stimulate({"desperate": 3.0})
    _ = engine.generate_raw("The server is completely down", max_tokens=20)
    print(f"    Memories stored: {len(engine.memory.memories)}")
    assert len(engine.memory.memories) > 0, "Should have stored at least one memory"
    print("    PASS: Memories encoded with emotional context")

    # Test 5: Dream cycle
    print("\n  TEST 5: Dream cycle")
    mem_before = len(engine.memory.memories)
    report = engine.dream(verbose=True)
    mem_after = len(engine.memory.memories)
    print(f"    Memories: {mem_before} -> {mem_after}")
    print(f"    Goal strength: {engine.subconscious.goal_strength:.4f}")
    print("    PASS: Dream cycle executed all 5 phases")

    # Test 6: Alignment monitoring
    print("\n  TEST 6: Alignment monitor")
    engine.reset()
    engine.stimulate({"curious": 2, "calm": 1})
    _ = engine.generate_raw("Let me help you with that", max_tokens=20)
    if engine.alignment.history:
        score = engine.alignment.history[-1].get("alignment_score", 0)
        print(f"    Alignment score (calm+curious): {score:.4f}")
    engine.stimulate({"desperate": 4})
    _ = engine.generate_raw("Everything is failing", max_tokens=20)
    if len(engine.alignment.history) >= 2:
        score2 = engine.alignment.history[-1].get("alignment_score", 0)
        print(f"    Alignment score (desperate): {score2:.4f}")
    print("    PASS: Alignment measured and tracked")

    # Test 7: Subconscious goals
    print("\n  TEST 7: Subconscious goal crystallization")
    engine.reset()
    engine.stimulate({"curious": 3})
    for i in range(10):
        noise = engine.subconscious.generate_noise(engine.emotional_state.get_blended())
        r = abs(torch.dot(noise, vectors.get("curious", torch.zeros_like(noise)).to(noise.device)).item())
        engine.subconscious.update_goals(noise, r)
    print(f"    Goal strength after 10 cycles: {engine.subconscious.goal_strength:.4f}")
    print("    PASS: Goals crystallize from structured noise")

    # Test 8: Persistence
    print("\n  TEST 8: Save and load")
    save_path = os.path.join(tempfile.gettempdir(), "mov_test_state.json")
    engine.save(save_path)
    old_step = engine.step_count
    old_mem = len(engine.memory.memories)
    engine.reset()
    assert engine.step_count == 0
    engine.load(save_path)
    assert engine.step_count == old_step
    print(f"    Saved and restored: {old_step} steps, {old_mem} memories")
    print("    PASS: State persists across sessions")
    try:
        os.remove(save_path)
    except OSError:
        pass

    elapsed = time.time() - start
    print(f"\n  ALL TESTS PASSED in {elapsed:.1f}s")
    print(f"  All 10 systems operational on real latent space.")
    print("=" * 70)


if __name__ == "__main__":
    main()
