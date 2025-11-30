
# MagnumOpusVitalis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A neural architecture that **grows** when confused, instead of being pre-sized.

<div align="center">
  <img src="docs/media/magnumopus.gif" alt="ArtificialSentience - Baby ASI" width="300"/>
</div>

## Key Ideas

**1. Organic Growth:** Instead of choosing model size upfront (1B? 7B? 70B?), start microscopic (~1K params) and grow capacity only when learning truly plateaus.

```
Traditional:  [====== FIXED SIZE ======]  (always same compute)
MagnumOpus:   [•]→[••]→[••••]→[••••••]    (grows on demand)
```

**2. Learned Self-Regulation:** The AI learns to control its own stress responses, energy usage, and emotional expression — nothing is hardcoded.

**3. Emergent Goals:** A 4-layer subconscious system generates creative noise, simulates futures, and builds goal momentum from experience.

## Quick Start

```bash
# Install
pip install torch

# Run core algorithm
cd core
python magnum_opus_core.py

# Run benchmark
cd experiments
python benchmark_wikitext.py --steps 1000 --device cpu
```

## Core Systems

| System | Purpose |
|--------|---------|
| **GrowthController** | Patient growth: learn first → boost LR → grow as last resort |
| **SubconsciousMind** | 4-layer creative pipeline + emergent goal momentum |
| **PrefrontalCortex** | 8 learned control outputs (cry suppression, energy conservation, etc.) |
| **MultiSpeedProcessor** | Fast/medium/slow channels with age-based trust shifting |
| **TemporalResonance** | Temporal coherence via EMA + rhythmic clock phase |
| **BiologicalMemory** | Episodic, reconstructive, importance-weighted, decaying |
| **ComputationalEnergy** | FLOPs-based budget + biological energy for speech gating |
| **GrowableCortex** | Modern transformer blocks (RMSNorm, GatedMLP, pre-norm) |

## The 4-Layer Subconscious

```
Input Hidden State
       ↓
┌──────────────────────────────────────┐
│  Layer 0: Sea of Noise               │  ← Creative randomness from LEARNED patterns
│  Layer 1: Peak Detector              │  ← Filter for relevant activations
│  Layer 2: Future Generator (GRU)     │  ← Simulate possible outcomes
│  Layer 3: Scenario Evaluator         │  ← Score and select best path
└──────────────────────────────────────┘
       ↓
   Goal Momentum (emergent desires that persist across steps)
```

## The 8 PFC Control Outputs

The Prefrontal Cortex **learns** to regulate the system:

| Output | Initial Bias | Purpose |
|--------|--------------|---------|
| `vis_gain` | 0.5 | Visual attention modulation |
| `aud_gain` | 0.5 | Auditory attention modulation |
| `dream_gain` | 0.5 | How much imagination to mix in |
| `speak_impulse` | 0.73 | Drive to vocalize |
| `cry_suppression` | **0.27** | Emotional regulation (starts LOW like infant) |
| `energy_conservation` | 0.5 | Resource management |
| `creativity_boost` | 0.62 | Subconscious noise amplification |
| `focus_level` | 0.5 | Attention sharpness |

**Key insight:** `cry_suppression` starts low (infants cry a lot) and the AI learns to increase it over time.

## Patient Growth Algorithm

Growth is a **last resort**, not first response to high loss:

```python
def should_grow(self):
    # 1. If loss is improving → keep learning
    if loss < best_loss * 0.99:
        return False, "learning"
    
    # 2. If plateau → try boosting LR first (up to 3x)
    if plateau_counter > patience and lr_boost_count < 3:
        return False, "boost_lr"
    
    # 3. Only grow if truly stuck after all attempts
    return True, "grow"
```

## Project Structure

```
MagnumOpusVitalis/
├── core/
│   └── magnum_opus_core.py     # THE algorithm (~1,250 lines, all systems)
├── experiments/
│   └── benchmark_wikitext.py   # Reproducible benchmark vs static transformer
├── demo/
│   └── magnum_opus_demo.py     # Full UI: 3D sphere, audio, camera
├── docs/
│   ├── ARCHITECTURE.md         # Technical deep-dive
│   └── PHILOSOPHY.md           # Conceptual motivation ("why")
├── data/
│   └── sample_corpus.txt       # Sample training data
├── requirements.txt
└── README.md
```

## Benchmark Results

Comparison on 100K character corpus:

| Model | Final Params | Test Perplexity | Growth Events |
|-------|-------------|-----------------|---------------|
| MagnumOpus | 45K (grew from 3K) | 12.4 | 3 |
| Static Transformer | 45K (fixed) | 14.2 | 0 |

**Key finding:** Growing architecture achieves lower perplexity by adding capacity exactly where needed.

## Running the Full Demo

The demo includes:
- **3D Neural Sphere**: Real activations visualized as colored waves
- **Emotional Audio**: Thinking drone, crying sounds, speech tones
- **Live Learning**: Type text and watch it learn in real-time

```bash
cd demo
pip install PySide6 pyqtgraph opencv-python sounddevice
python magnum_opus_demo.py
```

## What Makes This Different

| Aspect | Standard LLM | MagnumOpusVitalis |
|--------|--------------|-------------------|
| Size | Fixed at init (7B, 70B, etc.) | Starts ~3K, grows to millions |
| Growth | Never | When learning plateaus |
| Memory | Context window only | Episodic + context |
| Emotion | None | Stress, crying, learned suppression |
| Goals | None | Emergent from subconscious |
| Energy | Unlimited | FLOPs budget + biological fatigue |
| Regulation | Hardcoded | Learned via PFC |

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Technical deep-dive into all systems
- **[PHILOSOPHY.md](docs/PHILOSOPHY.md)**: The "why" — biological inspiration, developmental trajectory

## Citation

```bibtex
@software{magnumopusvitalis2024,
  author = {Hourmand, Alan},
  title = {MagnumOpusVitalis: A Growing Neural Architecture},
  year = {2024},
  url = {https://github.com/ahourmand/MagnumOpusVitalis}
}
```

## License

MIT