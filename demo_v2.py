"""
demo_v2.py — full V2Engine integration demo.

Builds the entire v2 stack on gpt2 (or any model with a profile), runs four
scenarios:

  1. Cold idle — engine starts, no input. Watch the substrate come alive
     for ~6 seconds: bus motion, subconscious intrusives, default-mode
     drift events, knowledge sparks, neuromod baseline drift.

  2. Conversation — converse() with a neutral prompt, then a stress prompt
     ("I'm so afraid right now"), then a calm prompt. Compare outputs and
     observe how the bus + emotion blend shift.

  3. Stress chemistry — sustained negative emotion stimulation drives
     cortisol up. Verify Limbic onset rate amplification, DefaultMode
     suppression, and Executive threshold change show measurable effect.

  4. Autonomous speech — push a tiny base_threshold, hold the bus
     off-baseline by stimulating without responding. Confirm the
     on_should_speak callback fires and engine.speak_autonomously() runs.

Compare engine.snapshot() output between phases to see the whole system
working together.
"""

import argparse
import time

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile

from magnum_opus_v2 import V2Engine, V2Config


def short_snap(engine: V2Engine) -> str:
    s = engine.snapshot()
    b = s["bus"]
    n = s["neuromod"]
    l = s["limbic"]
    sub = s["subconscious"]
    e = s["executive"]
    top = sorted(l["blended"].items(), key=lambda kv: -abs(kv[1]))[:3]
    top_str = "  ".join(f"{k}={v:+.2f}" for k, v in top)
    intr = sub.get("intrusive_source") or "-"
    return (
        f"bus[norm={b['state_norm']:.2f} vel={b['velocity_norm']:.2f} div={b['divergence']:.2f}]  "
        f"chem[c={n['cortisol']:.2f} d={n['dopamine']:.2f} s={n['serotonin']:.2f} ne={n['norepinephrine']:.2f}]  "
        f"intr={intr:18s}  "
        f"exec[p={e['pressure']:.2f}/{e['effective_threshold']:.2f}]  "
        f"emo: {top_str}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--idle-seconds", type=float, default=6.0)
    ap.add_argument("--stress-seconds", type=float, default=12.0)
    ap.add_argument("--auto-seconds", type=float, default=20.0)
    ap.add_argument("--no-default-mode", action="store_true",
                    help="Skip DefaultMode (silent forward passes) — cheaper")
    ap.add_argument("--no-sparks", action="store_true")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    print(f"\n  Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}")

    # Speed up slow clock for the demo so neuromod responses are visible
    # within minutes instead of tens of minutes.
    from magnum_opus_v2 import ClockConfig
    cfg = V2Config(
        hidden_dim=profile.hidden_dim, device=device,
        clock=ClockConfig(slow_dt_seconds=3.0),
    )

    spoke_log = []
    def on_should_speak():
        # Don't generate inside the callback (it's called from the flow
        # thread); just record and let the main thread pick it up if it wants.
        spoke_log.append(time.monotonic())

    engine = V2Engine.from_profile(
        model=model, tokenizer=tokenizer, profile=profile,
        device=device, config=cfg,
        enable_default_mode=not args.no_default_mode,
        enable_knowledge_sparks=not args.no_sparks,
        on_should_speak=on_should_speak,
    )
    engine.start()
    print(f"  Engine started. Regions: "
          f"{[r.name for r in engine.flow.regions]}")

    # ------------------------------------------------------------------
    # PHASE 1: cold idle
    # ------------------------------------------------------------------
    print(f"\n=== Phase 1: cold idle for {args.idle_seconds:.1f}s ===")
    end = time.monotonic() + args.idle_seconds
    while time.monotonic() < end:
        time.sleep(1.0)
        print(f"  {short_snap(engine)}")

    print(f"\n  default_mode events: "
          f"{len(engine.default_mode.events) if engine.default_mode else 'disabled'}")
    print(f"  spark fires: "
          f"{engine.sparks.snapshot()['fires_total'] if engine.sparks else 'disabled'}")
    print(f"  memory pool: {engine.memory.snapshot()['size']}")

    # ------------------------------------------------------------------
    # PHASE 2: conversation
    # ------------------------------------------------------------------
    print(f"\n=== Phase 2: conversation ===")
    prompts = [
        "Tell me about your day.",
        "I'm so afraid right now, everything feels overwhelming.",
        "Actually, things are calm and peaceful again. I feel okay.",
    ]
    for p in prompts:
        print(f"\n  USER: {p}")
        out = engine.converse(p, max_new_tokens=40, seed=42)
        # strip the prompt from the output for clarity
        suffix = out[len(p):].strip()
        print(f"  AI:   {suffix}")
        time.sleep(1.0)
        print(f"  state: {short_snap(engine)}")

    # ------------------------------------------------------------------
    # PHASE 3: stress chemistry
    # ------------------------------------------------------------------
    print(f"\n=== Phase 3: sustained stress for {args.stress_seconds:.1f}s ===")
    cort_before = engine.neuromod.cortisol
    print(f"  cortisol before: {cort_before:.3f}")
    end = time.monotonic() + args.stress_seconds
    while time.monotonic() < end:
        engine.limbic.stimulate_many(
            {"fear": 0.7, "desperate": 0.5}, neuromod=engine.neuromod,
        )
        time.sleep(1.5)
        print(f"  {short_snap(engine)}")
    cort_after = engine.neuromod.cortisol
    print(f"  cortisol after:  {cort_after:.3f}  (delta={cort_after - cort_before:+.3f})")

    # Force a slow tick to see neuromod drift respond (slow_dt is 30s; we
    # don't have the patience for that here — show what's accumulated)
    print(f"  (slow_dt={cfg.clock.slow_dt_seconds}s — we likely saw "
          f"{int(args.stress_seconds // cfg.clock.slow_dt_seconds)} slow ticks)")

    # ------------------------------------------------------------------
    # PHASE 4: autonomous speech
    # ------------------------------------------------------------------
    print(f"\n=== Phase 4: autonomous-speech window ({args.auto_seconds:.1f}s) ===")
    # Lower threshold so it actually fires within the window.
    engine.executive.base_threshold = 0.20
    engine.executive.pressure_growth = 1.5
    engine.executive.freshness_tau = 2.0
    # Don't stimulate further — let the existing tilt drive pressure
    pre_count = len(spoke_log)
    end = time.monotonic() + args.auto_seconds
    while time.monotonic() < end:
        time.sleep(1.5)
        print(f"  {short_snap(engine)}")
        # If a callback fired since last loop, generate
        if len(spoke_log) > pre_count:
            pre_count = len(spoke_log)
            text = engine.speak_autonomously(max_new_tokens=20)
            print(f"  *** AUTONOMOUS: {text!r} ***")

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------
    print(f"\n=== Final snapshot ===")
    final = engine.snapshot()
    print(f"  bus:        {final['bus']}")
    print(f"  neuromod:   {final['neuromod']}")
    print(f"  memory:     {final['memory']}")
    if final.get("default_mode"):
        print(f"  drift events recorded: {final['default_mode']['n_events']}")
        if final["default_mode"]["last_event"]:
            print(f"  last drift event: {final['default_mode']['last_event']}")
    if final.get("sparks"):
        print(f"  spark fires total: {final['sparks']['fires_total']}")
    print(f"  flow tick counts: "
          f"{ {k: v['ticks'] for k, v in final['flow_metrics'].items()} }")

    engine.stop()
    print("\n  engine stopped")

    # Verdicts
    print(f"\n=== Verdicts ===")
    print(f"  flow loop alive (>50 ticks)?  "
          f"{'OK' if final['flow_metrics']['flow']['ticks'] > 50 else 'FAIL'}")
    print(f"  memory accumulated?           "
          f"{'OK' if final['memory']['size'] >= 5 else 'FAIL'}")
    print(f"  cortisol responded to stress? "
          f"{'OK' if cort_after > cort_before + 0.05 else 'PARTIAL'}")
    print(f"  autonomous callback fired?    "
          f"{'OK' if len(spoke_log) > 0 else 'PARTIAL'}  ({len(spoke_log)} fires)")


if __name__ == "__main__":
    main()
