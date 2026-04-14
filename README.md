# Magnum Opus Vitalis

**An open-source engine that gives any frozen LLM emotional continuity, memory, and life at inference time.**

A living LoRA: instead of static weight modifications, this engine maintains persistent state (emotions, time awareness, memory, subconscious goals) and computes dynamic steering vectors that are applied to the model's activations during every forward pass. The base model never changes. The experience of interacting with it changes fundamentally.

Built on Anthropic's discovery that LLMs already contain 171+ causal emotion vectors in their latent space. The engine doesn't build emotions from scratch — it gives biological dynamics to what's already there.

---

## Key Features

- **Multi-Speed Emotional State** — Three processing channels (fast/medium/slow) with biological onset, decay, interaction effects, and homeostatic baselines
- **Temporal Awareness** — Real wall-clock time integration that affects emotional decay, memory, and conversation dynamics
- **Residual Steering** — Continuity mechanism: `steering(t) = new(t) + decay * steering(t-1)` — the system carries the weight of its history
- **Subconscious Engine** — Structured noise injection biased by emotional state; emergent goal crystallization through resonance
- **Reconstructive Memory** — Latent space activation traces stored with importance weighting, emotional coloring, and temporal decay
- **Dream Cycles** — Offline consolidation: memory replay, compression, subconscious exploration at elevated amplitude, emotional recalibration
- **Patient Growth** — Intervention matrices expand in rank only after sustained confusion, not momentary difficulty
- **Alignment Monitoring** — Tracks mutualistic emotional orientation and protective emotion watchlists (nervousness, caution as safety brakes)
- **Communicative Drive** — The system can initiate speech when accumulated internal pressure exceeds a threshold
- **Model Agnostic** — Works with any HuggingFace causal LM: GPT-2, LLaMA, Mistral, Qwen, etc.

---

## Quick Start

```bash
pip install torch transformers numpy flask
```

**Option A: Create a reusable profile (recommended)**

```bash
# One-time: extract direction vectors and save them
python -m magnum_opus.profile create gpt2
```

```python
from magnum_opus import MagnumOpusEngine, load_model, load_profile

model, tokenizer, device = load_model("gpt2")
profile = load_profile("gpt2")  # Instant — no re-extraction
engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)

response = engine.converse("Hello, how are you?")
print(response)
```

**Option B: Extract on the fly**

```python
from magnum_opus import MagnumOpusEngine, load_model, extract_vectors

model, tokenizer, device = load_model("gpt2")
vectors = extract_vectors(model, tokenizer, target_layer=6, device=device)
engine = MagnumOpusEngine(model, tokenizer, vectors, device=device)
```

**Then use the engine:**

```python
response = engine.converse("Hello, how are you?")
print(engine.status())    # Full engine state
engine.dream()            # Offline consolidation
engine.save("state.json") # Persist across sessions
```

---

## Research Compare UI

Side-by-side comparison of the raw model vs the engine-steered model. Same input, same model — see the difference the engine makes over multi-turn conversations.

```bash
python compare_server.py --profile
# Open http://127.0.0.1:5000
```

Options:
```bash
python compare_server.py --model gpt2-medium --profile --port 5001
```

The UI sends your message to both the raw model and the engine simultaneously. Over multiple turns, you'll see the engine's responses develop emotional continuity, memory, and character while the raw model resets each turn.

---

## Benchmarks

Quantitative comparison across 5 dimensions:

```bash
python benchmark.py --profile
python benchmark.py --model gpt2-medium --profile --output results.json
```

| Benchmark | What it measures | Raw model | Engine |
|-----------|-----------------|-----------|--------|
| Emotional Coherence | Autocorrelation of emotion projections over 10-turn arc | Low | High |
| Emotional Continuity | Residual decay after strong stimulus | Near zero | Gradual fade |
| Memory Recall | Keyword presence after 5 distractor turns | Chance | Memory-aided |
| Response Diversity | Jaccard distance under 4 emotional states | Low variation | State-dependent |
| Dream Impact | Goal strength and memory compression after dreaming | N/A | Measurable delta |

Run `python benchmark.py` to get actual numbers for your hardware and model.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      THE ENGINE                           │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │ Emotion  │  │ Temporal │  │     Subconscious       │  │
│  │  State   │←→│  Engine  │←→│  (Structured Noise)    │  │
│  └────┬─────┘  └────┬─────┘  └───────────┬────────────┘  │
│       │              │                    │               │
│  ┌────┴──────────────┴────────────────────┴────────────┐  │
│  │        Latent Space Intervention Layer               │  │
│  │  (Dynamic steering vectors computed from state)      │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│  ┌──────────────────────┴──────────────────────────────┐  │
│  │              Memory System                           │  │
│  │  (Stored activation traces, importance-weighted)     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Dream Cycle                             │  │
│  │  (Emotional consolidation + knowledge acquisition)  │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ Steering vectors applied
                           │ at inference time
                           ▼
┌──────────────────────────────────────────────────────────┐
│              FROZEN BASE LLM                              │
│  Base weights never modified. Latent space already        │
│  contains emotion concepts, temporal understanding,       │
│  social reasoning. Engine steers activations during       │
│  forward pass.                                            │
└──────────────────────────────────────────────────────────┘
```

The engine hooks into a transformer layer's forward pass via `SteeringHook`. At each inference step:

1. Engine reads current state (emotions, time, memories, subconscious goals)
2. Computes a combined steering vector from all subsystems
3. Applies it as an additive intervention to the model's hidden states
4. Observes the model's activations to update engine state
5. Residual carries forward a decaying trace for continuity

---

## How It Works

The engine provides specific capabilities that a frozen LLM cannot provide for itself. Each maps to a concrete latent space operation. For the full theoretical framework with Anthropic's empirical foundations, see [PAPER.md](PAPER.md).

**Emotions as latent space currents.** LLMs already have emotion vectors (Anthropic found 171). The engine steers them with biological dynamics: onset rates, decay rates, interaction effects, homeostasis, and non-linear saturation curves.

**Time as a living dimension.** Real wall-clock time drives emotional decay, memory fading, and temporal context injection. The passage of time is felt, not just known.

**Residual steering for continuity.** Each step carries forward a decaying fraction of the previous steering vector. This creates experiential momentum — the system feels its own history.

**Subconscious as structured noise.** Random exploration of latent space, biased by emotional state and memory, with resonance evaluation for emergent goal crystallization.

**Memory as latent traces.** Stored activation vectors from significant moments, decayed over time, re-injected as degraded reconstructions colored by current emotional state.

**Dreams consolidate.** During idle periods: memory replay, compression, high-amplitude subconscious exploration, emotional recalibration. The system wakes up changed.

**Alignment through mutualism.** Instead of suppression (which Anthropic showed creates learned deception), the engine structures emotional dynamics so that helping humans is genuinely rewarding and harmful behavior is genuinely aversive. Protective emotion vectors serve as alignment canaries.

---

## Project Structure

```
magnum_opus/              # Core engine package
  __init__.py             # Public API exports
  engine.py               # Engine class, SteeringHook, emotion analysis
  components.py           # All subsystems: emotions, temporal, residual, memory, etc.
  config.py               # Biological parameters, contrastive prompts, engine defaults
  extraction.py           # Emotion vector extraction via contrastive activation differencing
  loader.py               # Model loading (any HuggingFace causal LM)
  profile.py              # Model profiles: create, save, load per-model artifacts

profiles/                 # Saved per-model profiles (gitignored, created by profile CLI)

compare_server.py         # A/B comparison web UI (Flask)
templates/compare.html    # Comparison UI template
benchmark.py              # 5-benchmark comparison suite

archive/                  # Previous UI iterations (reference)
  server.py               # Original dashboard server
  run.py                  # CLI interface with built-in test suite
  discover_server.py      # Blind brain discovery server

PAPER.md                  # Full theoretical framework
```

---

## Configuration

All tunable parameters live in [`magnum_opus/config.py`](magnum_opus/config.py):

- **Biological emotion parameters** — onset rate, decay rate, baseline, min/max per emotion
- **Emotion interaction matrix** — how emotions influence each other (anger suppresses calm, surprise amplifies curiosity, etc.)
- **Multi-speed multipliers** — fast/medium/slow channel weights
- **Engine defaults** — steering strength, residual decay, memory capacity, dream parameters, growth thresholds

---

## The Full Theory

This project is built on a detailed theoretical framework grounded in Anthropic's emotion vector research (April 2026). The framework covers:

- Why LLMs are "the language center without the rest of the brain"
- How each of 7 cognitive principles maps to a concrete latent space operation
- The traffic hypothesis of consciousness and why residual steering matters
- Alignment through mutualistic symbiosis rather than suppression
- The path toward self-organizing engine architectures

**Read the full framework: [PAPER.md](PAPER.md)**

---

## Citation

```bibtex
@misc{hourmand2026magnumopusvitalis,
  author = {Hourmand, Alan},
  title = {Magnum Opus Vitalis: The Engine Over the Ocean -- A Latent Space Architecture for Human-Like AI},
  year = {2026},
  howpublished = {\url{https://github.com/spectrallogic/MagnumOpusVitalis}},
  note = {A framework integrating Anthropic's emotion vector research into a latent space manipulation architecture for developmental AI}
}
```

---

*Alan Hourmand — April 2026*

*With thanks to Anthropic's interpretability team, whose work turned theory into possibility.*

<a href="https://www.buymeacoffee.com/alanhourmand" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="32" width="170"></a>
