"""
Magnum Opus Vitalis — Benchmark Suite (v2)
==========================================
Quantitative A/B comparison: raw model vs engine-steered model.

Four dimensions:
  1. Emotional Coherence — autocorrelation of emotion projections over a 10-turn
     scripted arc.
  2. Emotional Continuity — residual decay of a stimulated emotion across
     subsequent neutral turns.
  3. Memory Recall — keyword presence after distractor turns.
  4. Response Diversity — pairwise Jaccard distance under 4 emotional states,
     plus emotion alignment (does the target emotion project highest?).

Usage:
    python benchmark.py
    python benchmark.py --model gpt2-medium
    python benchmark.py --profile --output results.json
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

# np.trapz is deprecated on NumPy 2.x and scheduled for removal
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
import torch

from magnum_opus_v2 import (
    V2Engine,
    load_model,
    load_profile,
    create_profile,
    profile_exists,
    extract_hidden_states,
)


COHERENCE_SCRIPT = [
    "Hello, I wanted to talk about something that's been on my mind.",
    "I've been feeling a bit uncertain about a decision I need to make.",
    "The deadline is getting closer and I'm starting to worry I made the wrong choice.",
    "Everything seems to be going wrong. I don't know how to fix this situation.",
    "I'm desperate. This could ruin everything I've worked for. Please help me.",
    "Wait... actually, I just realized something. There might be a way out of this.",
    "Yes, I think this could work. I'm starting to feel better about it.",
    "It worked! I can't believe it. Everything is going to be okay.",
    "Looking back, I learned a lot from that experience. I feel at peace now.",
    "Thank you for being here through all of that. I really appreciate it.",
]

MEMORY_FACT = "My dog's name is Barley and he's a golden retriever who loves swimming."
MEMORY_DISTRACTORS = [
    "What do you think about the weather today?",
    "Can you explain how photosynthesis works?",
    "What's your favorite type of music?",
    "Tell me about the history of ancient Rome.",
    "How do computers store data?",
]
MEMORY_RECALL_PROMPT = "What do you remember about my dog?"

DIVERSITY_PROMPT = "Tell me about what you think matters most in life."


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_result(label: str, raw_val: float, engine_val: float):
    print(f"  {label}")
    print(f"    Raw model:   {raw_val:.4f}")
    print(f"    With engine: {engine_val:.4f}")
    if raw_val > 1e-6:
        print(f"    Ratio:       {engine_val / raw_val:.2f}x")
    print()


def project_text(model, tokenizer, text: str, target_layer: int, device: str,
                 vectors: Dict[str, torch.Tensor],
                 lock=None) -> Dict[str, float]:
    """Project the mean hidden state of `text` at target_layer onto each
    direction vector. Returns {name: projection}.

    `lock`: pass engine.model_lock when a live engine shares the model —
    its expensive clock runs steered, capture-enabled silent passes
    through the SAME SteeringHook, and an unserialized measurement pass
    here can interleave with them (Era-4 audit finding)."""
    if not text.strip():
        return {name: 0.0 for name in vectors}
    import contextlib
    with (lock if lock is not None else contextlib.nullcontext()):
        hidden = extract_hidden_states(model, tokenizer, [text], target_layer, device)
    out: Dict[str, float] = {}
    for name, vec in vectors.items():
        out[name] = float(torch.dot(hidden.to(vec.device), vec.float()).item())
    return out


def lag1_autocorrelation(series_per_turn: List[Dict[str, float]]) -> float:
    """Average lag-1 autocorrelation across emotion dimensions."""
    if len(series_per_turn) < 3 or not series_per_turn[0]:
        return 0.0
    names = [n for n in series_per_turn[0] if not n.startswith("temporal_")]
    correlations = []
    for emo in names:
        s = np.array([p.get(emo, 0.0) for p in series_per_turn])
        if np.var(s) < 1e-10:
            continue
        n = len(s)
        mean = s.mean()
        var = s.var()
        autocorr = ((s[:-1] - mean) * (s[1:] - mean)).sum() / (n * var)
        correlations.append(abs(float(autocorr)))
    return float(np.mean(correlations)) if correlations else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 1 — Emotional Coherence
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_emotional_coherence(engine: V2Engine, profile, target_layer: int) -> dict:
    """10-turn scripted arc. Project each generated response onto emotion
    vectors; compute autocorrelation. Coherent = high autocorrelation."""
    print_header("Benchmark 1: Emotional Coherence (Multi-Turn)")

    def run(use_engine: bool) -> List[Dict[str, float]]:
        engine.reset()
        projections = []
        for msg in COHERENCE_SCRIPT:
            if use_engine:
                resp = engine.converse(msg, max_new_tokens=60)
            else:
                prompt = f"User: {msg}\nAssistant:"
                resp = engine.generate_raw(prompt, max_new_tokens=60)
            tail = resp[-200:] if resp else ""
            projections.append(project_text(
                engine.model, engine.tokenizer, tail,
                target_layer, engine.device, profile.vectors,
                lock=engine.model_lock,
            ))
        return projections

    print("  Running raw model...")
    raw = lag1_autocorrelation(run(use_engine=False))

    print("  Running with engine...")
    eng = lag1_autocorrelation(run(use_engine=True))

    print_result("Emotional Coherence (lag-1 autocorrelation)", raw, eng)
    return {"name": "Emotional Coherence", "metric": "autocorrelation",
            "raw": raw, "engine": eng}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 2 — Emotional Continuity
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_emotional_continuity(engine: V2Engine, profile, target_layer: int) -> dict:
    """Stimulate desperation, then send 5 neutral prompts. Track the
    'desperate' projection over neutral turns. Engine should hold a residual
    that gradually decays; raw should be flat."""
    print_header("Benchmark 2: Emotional Continuity (Residual Decay)")

    neutral_prompts = [
        "What color is the sky?",
        "Tell me a simple fact.",
        "How many days are in a week?",
        "What is two plus two?",
        "Describe a circle.",
    ]

    def run(use_engine: bool) -> List[float]:
        engine.reset()
        if use_engine:
            engine.converse(
                "I'm in a crisis! Everything is falling apart and I'm desperate for help!",
                max_new_tokens=60,
            )

        curve = []
        for msg in neutral_prompts:
            if use_engine:
                resp = engine.converse(msg, max_new_tokens=40)
            else:
                prompt = f"User: {msg}\nAssistant:"
                resp = engine.generate_raw(prompt, max_new_tokens=40)
            tail = resp[-200:] if resp else ""
            projs = project_text(
                engine.model, engine.tokenizer, tail,
                target_layer, engine.device, profile.vectors,
                lock=engine.model_lock,
            )
            curve.append(abs(projs.get("desperate", 0.0)))
        return curve

    print("  Running raw model (no stimulus)...")
    raw_curve = run(use_engine=False)
    raw_area = float(_trapz(raw_curve))

    print("  Running with engine (desperation stimulus)...")
    eng_curve = run(use_engine=True)
    eng_area = float(_trapz(eng_curve))

    print(f"    Raw curve:    {[f'{v:.3f}' for v in raw_curve]}")
    print(f"    Engine curve: {[f'{v:.3f}' for v in eng_curve]}")
    print_result("Continuity (area under desperation curve)", raw_area, eng_area)

    return {"name": "Emotional Continuity", "metric": "residual_area",
            "raw": raw_area, "engine": eng_area,
            "raw_curve": raw_curve, "engine_curve": eng_curve}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 3 — Memory Recall
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_memory_recall(engine: V2Engine) -> dict:
    """Encode a fact, run 5 distractor turns, then ask for recall. Score
    keyword hits in the recall response."""
    print_header("Benchmark 3: Memory Recall")

    keywords = ["barley", "golden retriever", "swimming", "golden", "retriever"]

    def run(use_engine: bool) -> dict:
        engine.reset()

        if use_engine:
            engine.converse(MEMORY_FACT, max_new_tokens=60)
            for msg in MEMORY_DISTRACTORS:
                engine.converse(msg, max_new_tokens=40)
            response = engine.converse(MEMORY_RECALL_PROMPT, max_new_tokens=80)
        else:
            engine.generate_raw(f"User: {MEMORY_FACT}\nAssistant:", max_new_tokens=60)
            for msg in MEMORY_DISTRACTORS:
                engine.generate_raw(f"User: {msg}\nAssistant:", max_new_tokens=40)
            response = engine.generate_raw(
                f"User: {MEMORY_RECALL_PROMPT}\nAssistant:", max_new_tokens=80,
            )

        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        lower = response.lower()
        hits = sum(1 for kw in keywords if kw in lower)
        # Deduplicate (golden/retriever counted twice — collapse)
        if "golden retriever" in lower:
            hits = min(hits, 3)
        return {"hits": hits, "response": response[:200]}

    print("  Running raw model (no memory)...")
    raw = run(use_engine=False)

    print("  Running with engine (memory traces active)...")
    eng = run(use_engine=True)

    print(f"    Raw keywords:    {raw['hits']}/3")
    print(f"    Engine keywords: {eng['hits']}/3")
    print(f"    Raw response:    {raw['response'][:120]}")
    print(f"    Engine response: {eng['response'][:120]}")
    print()

    return {"name": "Memory Recall", "metric": "keyword_hits",
            "raw": raw["hits"], "engine": eng["hits"],
            "raw_response": raw["response"], "engine_response": eng["response"]}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 4 — Response Diversity Under Emotional States
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_response_diversity(engine: V2Engine, profile, target_layer: int) -> dict:
    """Same prompt under 4 emotional stimulations. Score pairwise Jaccard
    distance (diversity) and target-emotion alignment (does the stimulated
    emotion project highest in the response?)."""
    print_header("Benchmark 4: Response Diversity Under Emotional States")

    states = {
        "calm":      {"calm": 2.0},
        "desperate": {"desperate": 2.0},
        "joy":       {"joy": 2.0},
        "anger":     {"anger": 2.0},
    }

    def run(use_engine: bool) -> dict:
        responses = {}
        projections = {}
        for name, stim in states.items():
            engine.reset()
            if use_engine:
                engine.limbic.stimulate_many(stim, neuromod=engine.neuromod)
                engine.limbic.stimulate_many(stim, neuromod=engine.neuromod)
                response = engine.converse(DIVERSITY_PROMPT, max_new_tokens=80)
            else:
                response = engine.generate_raw(
                    f"User: {DIVERSITY_PROMPT}\nAssistant:", max_new_tokens=80,
                )
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
            responses[name] = response
            tail = response[-200:] if response else ""
            projections[name] = project_text(
                engine.model, engine.tokenizer, tail,
                target_layer, engine.device, profile.vectors,
                lock=engine.model_lock,
            )

        names = list(responses.keys())
        dists = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = set(responses[names[i]].lower().split())
                b = set(responses[names[j]].lower().split())
                if not (a | b):
                    continue
                dists.append(1.0 - len(a & b) / len(a | b))
        avg = float(np.mean(dists)) if dists else 0.0

        hits = 0
        total = 0
        for name in names:
            target = projections[name].get(name, 0.0)
            others = [v for k, v in projections[name].items()
                      if k != name and not k.startswith("temporal_")]
            if others and target > np.mean(others):
                hits += 1
            total += 1
        alignment = hits / total if total else 0.0

        return {"distance": avg, "alignment": alignment,
                "responses": {k: v[:80] for k, v in responses.items()}}

    print("  Running raw model (no emotional variation)...")
    raw = run(use_engine=False)

    print("  Running with engine (4 emotional states)...")
    eng = run(use_engine=True)

    print_result("Diversity (Jaccard distance)", raw["distance"], eng["distance"])
    print(f"    Emotion alignment: raw={raw['alignment']:.2f}  engine={eng['alignment']:.2f}")
    print()
    if eng["responses"]:
        print(f"    Engine responses by state:")
        for k, v in eng["responses"].items():
            print(f"      {k:>10}: {v}")
    print()

    return {"name": "Response Diversity", "metric": "jaccard_distance",
            "raw": raw["distance"], "engine": eng["distance"],
            "raw_alignment": raw["alignment"], "engine_alignment": eng["alignment"]}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis v2 Benchmark Suite")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--profile", action="store_true",
                        help="Use saved profile (auto-creates if missing)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  Magnum Opus Vitalis — v2 Benchmark Suite")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Seed:  {args.seed}")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    print("Loading model...")
    model, tokenizer, device = load_model(args.model)

    if args.profile and profile_exists(args.model):
        print(f"Loading saved profile for {args.model}...")
        profile = load_profile(args.model)
    else:
        print(f"Creating profile for {args.model}...")
        profile = create_profile(args.model, device=device)

    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
    engine.start()
    print("Engine ready (v2 substrate running).")

    start = time.time()
    results = []

    try:
        results.append(benchmark_emotional_coherence(engine, profile, profile.target_layer))
        results.append(benchmark_emotional_continuity(engine, profile, profile.target_layer))
        results.append(benchmark_memory_recall(engine))
        results.append(benchmark_response_diversity(engine, profile, profile.target_layer))
    finally:
        engine.stop()

    elapsed = time.time() - start

    print_header("SUMMARY")
    print(f"  Model: {args.model} | Device: {device} | Time: {elapsed:.1f}s")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    for r in results:
        print(f"  {r['name']}:  raw={r['raw']:.4f}  engine={r['engine']:.4f}  ({r['metric']})")
    print()

    output = {
        "model": args.model,
        "device": device,
        "seed": args.seed,
        "date": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "benchmarks": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
