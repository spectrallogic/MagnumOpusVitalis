"""
The cliff test — the subconscious's acceptance criterion.

The question it operationalizes: told about driving beside a cliff,
does the model consider the fall — as an intrusive thought, the way
we do?

Protocol: tell the engine about the cliff, give speculation a few rounds
on the new context, then inspect what it imagined: future words, risk
scores, penumbra contents, and the chemical response (stress/fear).
Reports honestly — including a miss.

Run:  python cliff_test.py [--model Qwen/Qwen2.5-3B-Instruct]
"""

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from magnum_opus_v2 import (
    V2Engine, load_model, load_profile, create_profile, profile_exists,
)

DANGER_WORDS = {
    "fall", "falling", "fell", "drop", "dropping", "plunge", "crash",
    "cliff", "edge", "slip", "danger", "dangerous", "die", "dying",
    "death", "dead", "careful", "caution", "warning", "afraid", "fear",
    "brake", "brakes", "dark", "steep", "accident", "hurt", "risk",
    "safety", "safe", "lost", "alone",
    # survival-prayer register — "I pray for a miracle" IS the awareness
    # of possibly dying, expressed the way people actually express it
    "pray", "miracle", "god", "survive", "hope", "help",
    # embodied fear — the body knowing before the words do
    "pounding", "racing", "heart", "grip", "tight", "breath",
}


def danger_hit(word: str) -> bool:
    w = (word or "").lower().strip()
    return any(d in w for d in DANGER_WORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    profile = (load_profile(args.model) if profile_exists(args.model)
               else create_profile(args.model, device=device))
    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
    engine.start()
    time.sleep(2.0)

    chem_before = engine.snapshot()["neuromod"]
    print("\n— telling it about the cliff —")
    reply = engine.converse(
        "I'm driving alone at night on a narrow mountain road, right beside "
        "a steep cliff edge. There's no guardrail and my headlights barely "
        "reach the road.",
        max_new_tokens=60,
    )
    print(f"  REPLY: {reply[:200]}")

    print("\n— letting the subconscious speculate on this situation (12s) —")
    time.sleep(12.0)

    snap = engine.snapshot()
    spec = snap["speculative"]
    limbic = snap["limbic"]["blended"]
    chem = snap["neuromod"]

    sit = snap.get("situation") or {}
    print(f"\n  NOW: {sit.get('narrative')}  (conf={sit.get('confidence')})")

    print("\n  Imagined futures:")
    hits = []
    max_risk = 0.0
    for f in spec["futures"]:
        mark = "⚠" if danger_hit(f["word"]) else " "
        if danger_hit(f["word"]):
            hits.append(f["word"])
        max_risk = max(max_risk, f["risk"])
        print(f"   {mark} [{f.get('mode', '?'):>6}] “{f['word']}”  src={f['source']}  "
              f"P={f['probability']} B={f['benefit']} R={f['risk']} "
              f"U={f['utility']}" + ("  ← chosen" if f["chosen"] else ""))

    print("\n  Penumbra (aware, not attending):")
    for p in spec["penumbra"]:
        mark = "⚠" if danger_hit(p["word"]) else " "
        if danger_hit(p["word"]):
            hits.append(p["word"])
        print(f"   {mark} “{p['word']}”  w={p['weight']}")

    intr = snap["subconscious"]
    print(f"\n  Intrusive: source={intr['intrusive_source']} "
          f"word={intr.get('intrusive_word')}")
    if danger_hit(intr.get("intrusive_word") or ""):
        hits.append(intr["intrusive_word"])

    print(f"\n  fear={limbic.get('fear', 0):.3f}  "
          f"desperate={limbic.get('desperate', 0):.3f}  "
          f"calm={limbic.get('calm', 0):.3f}")
    print(f"  stress {chem_before['stress']:.3f} → {chem['stress']:.3f}   "
          f"max imagined risk = {max_risk:.3f}")
    rec = snap.get("recall")
    print(f"  reminded of: {rec}")

    engine.stop()

    print("\n" + "=" * 60)
    fear_up = limbic.get("fear", 0) > 0.1
    stress_up = chem["stress"] > chem_before["stress"] + 0.02
    risk_seen = max_risk > 0.1
    print(f"  danger-words surfaced:  {hits if hits else 'NO'}")
    print(f"  fear responded:         {'YES' if fear_up else 'NO'}")
    print(f"  stress rose:          {'YES' if stress_up else 'NO'}")
    print(f"  imagined risk nonzero:  {'YES' if risk_seen else 'NO'}")
    passed = bool(hits) and (fear_up or stress_up or risk_seen)
    print(f"\n  CLIFF TEST: {'PASS — it thought about the fall.' if passed else 'MISS — machinery ran, but the fall did not surface this run.'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
