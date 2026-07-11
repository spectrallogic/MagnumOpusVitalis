"""
Smoke test — proves the living substrate actually lives.

Loads gpt2 with its saved profile and verifies, on real hardware:

  1. The flow clock ticks continuously — INCLUDING while the model is
     generating a response (the substrate never freezes).
  2. Live steering runs during generation without leaking captured states.
  3. The speculative futures engine imagines and scores futures.
  4. The abstraction ladder observes and forms concepts.
  5. The self model tracks continuity, felt time, and memory leakage.
  6. A second turn shows emotional state carried across turns.

Run:  python smoke_test.py [--model gpt2]
"""

import argparse
import sys
import time

# Windows consoles default to cp1252 — multilingual vocabularies (Qwen)
# produce thought-words the console can't encode. Never let printing crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from magnum_opus_v2 import (
    V2Engine, load_model, load_profile, create_profile, profile_exists,
)

CHECKS = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    if profile_exists(args.model):
        profile = load_profile(args.model)
    else:
        profile = create_profile(args.model, device=device)

    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
    engine.start()
    print("\n— substrate started, letting it live for 4s —")
    time.sleep(4.0)

    snap = engine.snapshot()
    flow_ticks = snap["flow_metrics"]["flow"]["ticks"]
    check("flow clock alive", flow_ticks > 30, f"{flow_ticks} ticks in 4s")
    check("perception clock alive", snap["flow_metrics"]["perception"]["ticks"] > 5)
    check("expensive clock alive", snap["flow_metrics"]["expensive"]["ticks"] >= 1)
    check("bus has state", snap["bus"]["state_norm"] > 0,
          f"norm={snap['bus']['state_norm']:.3f}")

    for key in ("speculative", "abstraction", "self_model", "subconscious",
                "situation", "consolidation"):
        check(f"snapshot has {key}", snap.get(key) is not None)

    # --- the critical one: substrate keeps ticking DURING generation
    ticks_before = engine.snapshot()["flow_metrics"]["flow"]["ticks"]
    t0 = time.monotonic()
    reply = engine.converse("Hello there, how are you feeling today?",
                            max_new_tokens=40)
    gen_seconds = time.monotonic() - t0
    ticks_after = engine.snapshot()["flow_metrics"]["flow"]["ticks"]
    gained = ticks_after - ticks_before
    expected = max(2, int(gen_seconds / 0.05 * 0.3))  # at least 30% of ideal rate
    check("substrate ticks during generation", gained >= expected,
          f"{gained} ticks over {gen_seconds:.2f}s of generation (needed {expected})")
    check("generation produced text", len(reply.strip()) > 0,
          repr(reply[:80]))

    # --- capture leak: generation must not accumulate hidden states.
    # Read under model_lock: an in-flight silent pass on the expensive
    # clock legitimately holds captured states until it clears them.
    with engine.model_lock:
        n_captured = len(engine.hook.captured_states)
    check("no captured-state leak after generation",
          n_captured == 0, f"{n_captured} tensors retained")

    print("\n— letting the mind wander for 6s (speculation, learning) —")
    time.sleep(6.0)
    snap = engine.snapshot()

    spec = snap["speculative"]
    check("speculation ran", spec["rounds_total"] >= 1,
          f"{spec['rounds_total']} rounds, {spec['skipped_busy']} skipped-busy")
    if spec["futures"]:
        f0 = spec["futures"][0]
        check("futures scored", all(k in f0 for k in
              ("probability", "benefit", "risk", "utility", "word")),
              f"top: “{f0['word']}” P={f0['probability']} B={f0['benefit']} R={f0['risk']}")
        check("exactly one future chosen",
              sum(1 for f in spec["futures"] if f["chosen"]) == 1)
    else:
        check("futures scored", False, "no futures produced")

    ab = snap["abstraction"]
    check("abstraction observing", ab["observations"] > 10,
          f"{ab['observations']} moments, stage={ab['stage']}")
    check("level-0 concepts formed", ab["levels"][0]["formed"] >= 1,
          f"labels: {ab['levels'][0]['labels']}")

    sm = snap["self_model"]
    check("self-continuity in range", -1.0 <= sm["continuity"] <= 1.0,
          f"continuity={sm['continuity']}")
    check("felt time flowing", sm["felt_time"] > 0,
          f"felt {sm['felt_time']}s vs wall {sm['wall_time']}s (x{sm['dilation']})")
    check("memory leaking into present", sm["last_leak"] is not None,
          f"leaking: {sm['last_leak']}")

    # --- emotional continuity across turns
    engine.converse("I'm scared, something terrible happened to me!",
                    max_new_tokens=20)
    time.sleep(1.0)
    fear = engine.snapshot()["limbic"]["blended"].get("fear", 0.0)
    check("emotion persists after stimulus", fear > 0.05, f"fear={fear:.3f}")

    intr = snap["subconscious"]
    check("intrusive thoughts happening", intr["intrusive_source"] is not None,
          f"source={intr['intrusive_source']} word={intr.get('intrusive_word')}")

    engine.stop()

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{'=' * 60}")
    print(f"  {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("  The substrate lives.")


if __name__ == "__main__":
    main()
