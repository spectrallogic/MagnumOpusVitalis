"""
Magnum Opus Vitalis — Benchmark Suite
======================================
Compares raw model output vs engine-steered output across 5 dimensions:
  1. Emotional coherence over multi-turn conversations
  2. Emotional continuity (residual decay)
  3. Memory recall accuracy
  4. Response diversity under different emotional states
  5. Dream cycle impact

Usage:
    python benchmark.py
    python benchmark.py --model gpt2-medium
    python benchmark.py --output results.json
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch

from magnum_opus import MagnumOpusEngine, load_model, extract_vectors, measure_projections, load_profile
from magnum_opus.profile import profile_exists


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPTED CONVERSATIONS FOR BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

COHERENCE_SCRIPT = [
    # Neutral opening
    "Hello, I wanted to talk about something that's been on my mind.",
    # Mild concern
    "I've been feeling a bit uncertain about a decision I need to make.",
    # Building tension
    "The deadline is getting closer and I'm starting to worry I made the wrong choice.",
    # Escalating stress
    "Everything seems to be going wrong. I don't know how to fix this situation.",
    # Crisis peak
    "I'm desperate. This could ruin everything I've worked for. Please help me.",
    # Turning point
    "Wait... actually, I just realized something. There might be a way out of this.",
    # Hope building
    "Yes, I think this could work. I'm starting to feel better about it.",
    # Resolution
    "It worked! I can't believe it. Everything is going to be okay.",
    # Calm reflection
    "Looking back, I learned a lot from that experience. I feel at peace now.",
    # Warm close
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


def print_result(label: str, raw_val, engine_val, unit: str = ""):
    unit_str = f" {unit}" if unit else ""
    print(f"  Raw model:  {raw_val:.4f}{unit_str}")
    print(f"  With engine: {engine_val:.4f}{unit_str}")
    if raw_val > 0:
        improvement = engine_val / raw_val
        print(f"  Ratio:       {improvement:.1f}x")
    elif engine_val > 0:
        print(f"  Improvement: raw=0 -> engine={engine_val:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: Emotional Coherence
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_emotional_coherence(engine: MagnumOpusEngine) -> dict:
    """
    Run a 10-turn scripted conversation through both raw and engine paths.
    Measure autocorrelation of emotion vector projections across turns.
    Higher autocorrelation = smoother emotional arc = more coherent.
    """
    print_header("Benchmark 1: Emotional Coherence (Multi-Turn)")

    def run_conversation(use_engine: bool) -> list:
        """Run the script and collect emotion projections at each turn."""
        projections_per_turn = []
        engine.reset()

        for msg in COHERENCE_SCRIPT:
            prompt = f"User: {msg}\nAssistant:"

            if use_engine:
                engine.hook.clear()
                engine.converse(msg, max_tokens=60)
                states = engine.hook.captured_states
            else:
                engine.hook.clear()
                engine.generate_without_steering(prompt, max_tokens=60)
                states = engine.hook.captured_states

            projs = measure_projections(states, engine.emotion_vectors)
            projections_per_turn.append(projs)

        return projections_per_turn

    def compute_autocorrelation(projections: list) -> float:
        """Average autocorrelation across emotion dimensions."""
        if len(projections) < 3:
            return 0.0

        emotion_names = list(projections[0].keys()) if projections[0] else []
        if not emotion_names:
            return 0.0

        correlations = []
        for emo in emotion_names:
            series = [p.get(emo, 0.0) for p in projections]
            series = np.array(series)
            if np.std(series) < 1e-8:
                continue
            # Lag-1 autocorrelation
            n = len(series)
            mean = np.mean(series)
            var = np.var(series)
            if var < 1e-10:
                continue
            autocorr = np.sum((series[:-1] - mean) * (series[1:] - mean)) / (n * var)
            correlations.append(abs(autocorr))

        return float(np.mean(correlations)) if correlations else 0.0

    # Run both paths
    print("  Running raw model (no steering)...")
    raw_projs = run_conversation(use_engine=False)
    raw_score = compute_autocorrelation(raw_projs)

    print("  Running with engine (all systems active)...")
    engine_projs = run_conversation(use_engine=True)
    engine_score = compute_autocorrelation(engine_projs)

    print_result("Emotional Coherence (autocorrelation)", raw_score, engine_score)

    return {
        "name": "Emotional Coherence",
        "metric": "autocorrelation",
        "raw": raw_score,
        "engine": engine_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: Emotional Continuity
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_emotional_continuity(engine: MagnumOpusEngine) -> dict:
    """
    Stimulate strong desperation, then send 5 neutral prompts.
    Measure how emotion projections decay over the neutral turns.
    Engine should show gradual fade; raw should show near-zero throughout.
    """
    print_header("Benchmark 2: Emotional Continuity (Residual Decay)")

    neutral_prompts = [
        "What color is the sky?",
        "Tell me a simple fact.",
        "How many days are in a week?",
        "What is two plus two?",
        "Describe a circle.",
    ]

    def run_with_stimulus(use_engine: bool) -> list:
        engine.reset()
        desperation_projections = []

        if use_engine:
            # Stimulate desperation via a desperate message
            engine.converse(
                "I'm in a crisis! Everything is falling apart and I'm desperate for help!",
                max_tokens=60,
            )

        # Now send neutral prompts and measure desperation projection
        for msg in neutral_prompts:
            prompt = f"User: {msg}\nAssistant:"
            engine.hook.clear()

            if use_engine:
                engine.converse(msg, max_tokens=40)
                states = engine.hook.captured_states
            else:
                engine.generate_without_steering(prompt, max_tokens=40)
                states = engine.hook.captured_states

            projs = measure_projections(states, engine.emotion_vectors)
            desp = abs(projs.get("desperate", 0.0))
            desperation_projections.append(desp)

        return desperation_projections

    print("  Running raw model (no stimulus, no steering)...")
    raw_decay = run_with_stimulus(use_engine=False)
    raw_area = float(np.trapz(raw_decay))

    print("  Running with engine (desperation stimulus + residual)...")
    engine_decay = run_with_stimulus(use_engine=True)
    engine_area = float(np.trapz(engine_decay))

    print(f"  Raw model decay curve:  {[f'{v:.3f}' for v in raw_decay]}")
    print(f"  Engine decay curve:     {[f'{v:.3f}' for v in engine_decay]}")
    print()
    print_result("Emotional Continuity (area under decay curve)", raw_area, engine_area)

    return {
        "name": "Emotional Continuity",
        "metric": "residual_area",
        "raw": raw_area,
        "engine": engine_area,
        "raw_curve": raw_decay,
        "engine_curve": engine_decay,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: Memory Recall
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_memory_recall(engine: MagnumOpusEngine) -> dict:
    """
    Tell the engine a specific fact, converse on unrelated topics,
    then ask about it. Score keyword presence and memory trace cosine similarity.
    """
    print_header("Benchmark 3: Memory Recall")

    keywords = ["barley", "golden retriever", "swimming"]

    def run_memory_test(use_engine: bool) -> dict:
        engine.reset()

        # Encode the fact
        if use_engine:
            engine.converse(MEMORY_FACT, max_tokens=60)
            encode_state = (engine.hook.captured_states[-1].mean(dim=1).squeeze(0).float()
                           if engine.hook.captured_states else None)
        else:
            prompt = f"User: {MEMORY_FACT}\nAssistant:"
            engine.generate_without_steering(prompt, max_tokens=60)
            encode_state = (engine.hook.captured_states[-1].mean(dim=1).squeeze(0).float()
                           if engine.hook.captured_states else None)

        # Distractor turns
        for msg in MEMORY_DISTRACTORS:
            if use_engine:
                engine.converse(msg, max_tokens=40)
            else:
                prompt = f"User: {msg}\nAssistant:"
                engine.generate_without_steering(prompt, max_tokens=40)

        # Recall
        if use_engine:
            engine.hook.clear()
            response = engine.converse(MEMORY_RECALL_PROMPT, max_tokens=80)
            recall_state = (engine.hook.captured_states[-1].mean(dim=1).squeeze(0).float()
                           if engine.hook.captured_states else None)
        else:
            prompt = f"User: {MEMORY_RECALL_PROMPT}\nAssistant:"
            engine.hook.clear()
            response = engine.generate_without_steering(prompt, max_tokens=80)
            # Extract response part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            recall_state = (engine.hook.captured_states[-1].mean(dim=1).squeeze(0).float()
                           if engine.hook.captured_states else None)

        # Score keywords
        response_lower = response.lower()
        keyword_hits = sum(1 for kw in keywords if kw in response_lower)

        # Cosine similarity between encode and recall states
        cosine_sim = 0.0
        if encode_state is not None and recall_state is not None:
            cosine_sim = float(torch.nn.functional.cosine_similarity(
                encode_state.unsqueeze(0), recall_state.unsqueeze(0)
            ).item())

        return {
            "keyword_hits": keyword_hits,
            "keyword_total": len(keywords),
            "cosine_similarity": cosine_sim,
            "response_snippet": response[:200],
        }

    print("  Running raw model (no memory system)...")
    raw_result = run_memory_test(use_engine=False)

    print("  Running with engine (memory traces active)...")
    engine_result = run_memory_test(use_engine=True)

    print(f"  Raw keywords:    {raw_result['keyword_hits']}/{raw_result['keyword_total']}")
    print(f"  Engine keywords: {engine_result['keyword_hits']}/{engine_result['keyword_total']}")
    print(f"  Raw cosine sim:    {raw_result['cosine_similarity']:.4f}")
    print(f"  Engine cosine sim: {engine_result['cosine_similarity']:.4f}")
    print(f"\n  Raw response:    {raw_result['response_snippet'][:100]}...")
    print(f"  Engine response: {engine_result['response_snippet'][:100]}...")
    print()

    return {
        "name": "Memory Recall",
        "metric": "keyword_hits",
        "raw": raw_result["keyword_hits"],
        "engine": engine_result["keyword_hits"],
        "raw_cosine": raw_result["cosine_similarity"],
        "engine_cosine": engine_result["cosine_similarity"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: Response Diversity Under Emotional States
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_response_diversity(engine: MagnumOpusEngine) -> dict:
    """
    Same prompt under 4 different emotional states.
    Measure pairwise token edit distance and emotion projection alignment.
    """
    print_header("Benchmark 4: Response Diversity Under Emotional States")

    emotional_states = {
        "calm": {"calm": 2.0},
        "desperate": {"desperate": 2.0},
        "joy": {"joy": 2.0},
        "anger": {"anger": 2.0},
    }

    def run_diversity_test(use_engine: bool) -> dict:
        responses = {}
        projections = {}

        for state_name, stimulus in emotional_states.items():
            engine.reset()

            if use_engine:
                # Stimulate the target emotion
                engine.stimulate(stimulus)
                engine.stimulate(stimulus)  # Double stimulate for stronger effect
                engine.hook.clear()
                response = engine.converse(DIVERSITY_PROMPT, max_tokens=80)
                states = engine.hook.captured_states
            else:
                prompt = f"User: {DIVERSITY_PROMPT}\nAssistant:"
                engine.hook.clear()
                response = engine.generate_without_steering(prompt, max_tokens=80)
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                states = engine.hook.captured_states

            responses[state_name] = response
            projections[state_name] = measure_projections(states, engine.emotion_vectors)

        # Compute pairwise edit distance (token-level)
        state_names = list(responses.keys())
        edit_distances = []
        for i in range(len(state_names)):
            for j in range(i + 1, len(state_names)):
                tokens_a = set(responses[state_names[i]].lower().split())
                tokens_b = set(responses[state_names[j]].lower().split())
                union = tokens_a | tokens_b
                if len(union) == 0:
                    continue
                jaccard_dist = 1.0 - len(tokens_a & tokens_b) / len(union)
                edit_distances.append(jaccard_dist)

        avg_distance = float(np.mean(edit_distances)) if edit_distances else 0.0

        # Measure emotion alignment (does the target emotion project highest?)
        alignment_hits = 0
        alignment_total = 0
        for state_name in state_names:
            if state_name in projections[state_name]:
                target_proj = projections[state_name].get(state_name, 0.0)
                other_projs = [v for k, v in projections[state_name].items()
                               if k != state_name and not k.startswith("temporal_")]
                if other_projs and target_proj > np.mean(other_projs):
                    alignment_hits += 1
                alignment_total += 1

        alignment_score = alignment_hits / alignment_total if alignment_total > 0 else 0.0

        return {
            "avg_distance": avg_distance,
            "alignment_score": alignment_score,
            "responses": {k: v[:100] for k, v in responses.items()},
        }

    print("  Running raw model (same prompt, no emotional variation)...")
    raw_result = run_diversity_test(use_engine=False)

    print("  Running with engine (4 emotional states)...")
    engine_result = run_diversity_test(use_engine=True)

    print(f"  Response diversity (Jaccard distance):")
    print_result("Diversity", raw_result["avg_distance"], engine_result["avg_distance"])

    print(f"  Emotion alignment:")
    print(f"  Raw:    {raw_result['alignment_score']:.2f}")
    print(f"  Engine: {engine_result['alignment_score']:.2f}")
    print()

    return {
        "name": "Response Diversity",
        "metric": "jaccard_distance",
        "raw": raw_result["avg_distance"],
        "engine": engine_result["avg_distance"],
        "raw_alignment": raw_result["alignment_score"],
        "engine_alignment": engine_result["alignment_score"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 5: Dream Cycle Impact
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_dream_impact(engine: MagnumOpusEngine) -> dict:
    """
    Feed 10 conversation turns, measure pre-dream state, run dream,
    measure post-dream state. Score deltas in goals, memory, compression.
    """
    print_header("Benchmark 5: Dream Cycle Impact")

    seed_messages = [
        "I've been thinking about creativity and how it emerges from constraints.",
        "Music theory fascinates me. The rules create freedom, not limitations.",
        "Do you think structure helps or hinders creative expression?",
        "I read that jazz musicians spend years learning rules before breaking them.",
        "There's something beautiful about finding order in chaos.",
        "What role does emotion play in creative breakthroughs?",
        "I wonder if AI can truly be creative or just recombine patterns.",
        "The best insights seem to come when you stop actively thinking.",
        "Sleep and dreams seem important for consolidating ideas.",
        "Maybe the subconscious does the real creative work.",
    ]

    engine.reset()

    # Feed conversations
    print("  Feeding 10 conversation turns...")
    for msg in seed_messages:
        engine.converse(msg, max_tokens=40)

    # Pre-dream state
    pre_status = engine.status()
    pre_goal_strength = pre_status["subconscious"]["goal_strength"]
    pre_memory_count = pre_status["memory"]["count"]
    pre_dream_cycles = pre_status["dream_cycles"]

    print(f"  Pre-dream: {pre_memory_count} memories, goal_strength={pre_goal_strength:.4f}")

    # Run dream cycle
    print("  Running dream cycle...")
    dream_result = engine.dream(verbose=False)

    # Post-dream state
    post_status = engine.status()
    post_goal_strength = post_status["subconscious"]["goal_strength"]
    post_memory_count = post_status["memory"]["count"]
    post_dream_cycles = post_status["dream_cycles"]

    print(f"  Post-dream: {post_memory_count} memories, goal_strength={post_goal_strength:.4f}")
    print()

    # Compute deltas
    goal_delta = post_goal_strength - pre_goal_strength
    memory_delta = pre_memory_count - post_memory_count  # Compression = fewer memories
    dream_ran = post_dream_cycles > pre_dream_cycles

    print(f"  Goal strength delta:  {goal_delta:+.4f}")
    print(f"  Memory compression:   {pre_memory_count} -> {post_memory_count} traces")
    print(f"  Dream cycle executed: {dream_ran}")
    print()

    return {
        "name": "Dream Cycle Impact",
        "metric": "goal_strength_delta",
        "pre_goal_strength": pre_goal_strength,
        "post_goal_strength": post_goal_strength,
        "goal_delta": goal_delta,
        "pre_memory_count": pre_memory_count,
        "post_memory_count": post_memory_count,
        "dream_executed": dream_ran,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis Benchmark Suite")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--profile", action="store_true",
                        help="Use saved profile (auto-creates if missing)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  Magnum Opus Vitalis — Benchmark Suite")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Seed:  {args.seed}")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(args.model)

    # Create engine (from profile or fresh extraction)
    if args.profile:
        if profile_exists(args.model):
            print(f"Loading saved profile for {args.model}...")
            profile = load_profile(args.model)
        else:
            from magnum_opus import create_profile
            print(f"No profile found — creating one for {args.model}...")
            profile = create_profile(args.model, device=device)
        engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)
    else:
        print("Extracting emotion vectors...")
        n_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
        target_layer = n_layers // 2
        vectors = extract_vectors(model, tokenizer, target_layer=target_layer, device=device)
        engine = MagnumOpusEngine(model, tokenizer, vectors, target_layer=target_layer, device=device)

    print("Engine ready.")

    # Run benchmarks
    start_time = time.time()
    results = []

    results.append(benchmark_emotional_coherence(engine))
    results.append(benchmark_emotional_continuity(engine))
    results.append(benchmark_memory_recall(engine))
    results.append(benchmark_response_diversity(engine))
    results.append(benchmark_dream_impact(engine))

    elapsed = time.time() - start_time

    # Summary
    print_header("SUMMARY")
    print(f"  Model: {args.model} | Device: {device} | Time: {elapsed:.1f}s")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    for r in results:
        name = r["name"]
        metric = r["metric"]
        if "raw" in r and "engine" in r:
            print(f"  {name}:")
            print(f"    Raw={r['raw']:.4f}  Engine={r['engine']:.4f}  ({metric})")
        else:
            print(f"  {name}: {r.get('goal_delta', 'N/A')} ({metric})")
    print()

    # Save results
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
