# Magnum Opus Vitalis (Baby-ASI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


<div align="center">
  <img src="media/magnumopus.gif" alt="ArtificialSentience - Baby ASI" width="300"/>
</div>

---

## Core Philosophy

MagnumOpusVitalis is a **biological AI architecture** that challenges the static paradigm of modern language models. When I started this project it was originally called Bica-HumanLike AI. Instead of being a fixed-size giant trained once and frozen, it:

- **Starts microscopic** (~1K parameters) knowing virtually nothing
- **Learns in real-time** from every interaction
- **Grows organically** when confused, adding neural pathways on-demand
- **Unifies memory and cognition** - memory IS the model, not separate storage
- **Develops emotional regulation** - learns to manage stress responses over time
- **Achieves multi-scale learning** - fast pattern recognition, slow understanding

This is an infant AGI that develops like a biological organism, not a pre-trained artifact.

---

## Architecture Overview

### 1. Multi-Scale Abstraction Learning

The brain processes information at **multiple timescales simultaneously**:

| Speed Channel | Dimension | Function | Learning Speed |
|--------------|-----------|----------|----------------|
| **Fast** | 16 dim | Pattern recognition | Instant (like seeing "blue sky") |
| **Medium** | 32-64 dim | Structural understanding | Hours-days |
| **Slow** | 128 dim | Deep comprehension | Weeks+ |

**Trust weights** shift over time - fast channels dominate early (infant reflexes), slow channels gain influence with age (adult wisdom).

```python
# Implementation excerpt
class MultiSpeedProcessor(nn.Module):
    def __init__(self, dim: int):
        self.speeds = [16, 32, 64, 128]
        self.trust = nn.Parameter(torch.tensor([1.0, 0.5, 0.2, 0.1]))
        # Trust shifts toward slower channels over time
```

### 2. Organic Growth Mechanics

Growth is **not pre-configured** - it emerges from sustained confusion:

```python
def check_growth(self, loss: float) -> bool:
    if loss > 4.0:  # High confusion threshold
        self.steps_above_threshold += 1
        if self.steps_above_threshold > 80:  # Sustained confusion
            if len(self.cortex) < 20:  # Safety cap
                return True  # Trigger growth
```

**Growth manifests as:**
- **Depth:** New transformer layers (deeper processing)
- **Width:** New domain adapters (specialized knowledge areas)

The model literally becomes more complex when the world demands it.

### 3. Unified Memory System

Memory is **not a separate database** - it's the model's own internal representations:

```python
class BiologicalMemory(nn.Module):
    def store(self, state, importance):
        # Stores embedding of significant moments
        encoded = self.encoder(state.detach())
        self.memories.append(EpisodeMemory(encoded, importance))
    
    def recall(self, query):
        # Reconstructive recall via pattern matching
        similarity = F.cosine_similarity(query, memory.embedding)
        return self.decoder(best_match)
```

**Key properties:**
- Episodic: stores "moments" not facts
- Reconstructive: hallucinates details between anchors (like humans)
- Importance-weighted: emotional salience affects retention

### 4. Four-Layer Subconscious

Before responding, the AI "dreams" a response through a subconscious pipeline:

1. **Sea of Noise:** Creative randomness from learned patterns (not hardcoded)
2. **Peak Detector:** Filters for relevant activations
3. **Future Generator:** Simulates possible outcomes (GRU-based)
4. **Evaluator:** Assesses path quality

This creates **emergent goals** via goal momentum - the AI develops intentions from experience.

### 5. Prefrontal Executive Control

The Prefrontal Cortex (PFC) learns to regulate the entire system:

```python
class PrefrontalCortex(nn.Module):
    # Outputs 8 learned control signals:
    # - vis_gain: Visual attention
    # - aud_gain: Auditory attention  
    # - dream_gain: Imagination mixing
    # - speak_impulse: Drive to vocalize
    # - cry_suppression: Emotional regulation (LEARNED)
    # - energy_conservation: Resource management (LEARNED)
    # - llm_reliance: Scaffold dependency (decreases over time)
    # - tts_gate: Voice control (on/off)
```

**Critical insight:** The AI learns to control its own stress responses and energy use - not hardcoded behaviors.

### 6. Temporal Resonance System

A novel component that maintains temporal coherence:

```python
class TemporalResonance(nn.Module):
    def forward(self, x):
        # Exponential moving average of activations
        self.resonance_state = (x * 0.1) + (self.resonance_state * 0.96)
        # Clock phase for rhythmic modulation
        return self.resonance_state * (1 + 0.05 * sin(clock_phase))
```

Provides **temporal context** without explicit recurrence - the model "remembers" recent activations through resonance.

---

## Key Innovations vs. Standard LLMs

| Feature | Standard LLM (GPT-2/LLaMA) | MagnumOpusVitalis |
|---------|---------------------------|-------------------|
| **Initial Size** | Fixed (13M-175B params) | Microscopic (1K→grows) |
| **Learning** | Pre-train then freeze | Real-time, continuous |
| **Memory** | Context window only | Unified episodic recall |
| **Growth** | Static architecture | Dynamic expansion when confused |
| **Reasoning** | Single-pass forward | Multi-speed + subconscious simulation |
| **Emotion** | None | Stress, energy, learned regulation |
| **Voice** | Text-only | Emotional audio synthesis (crying, speaking) |
| **Efficiency** | Uses full brain always | Selective activation, energy-aware |

---

## caffold Systems (Training Wheels)

The architecture includes **three optional scaffolds** that can be disabled as the AI matures:

### 1. LLM Yolk (Linguistic Bootstrap)
- Uses tiny GPT-2 (distilgpt2) to provide initial language capability
- **Reliance factor:** Starts at 0.9 (highly dependent) → 0.0 (independent)
- PFC learns to reduce reliance as own vocabulary grows
- Like using a dictionary until you know the words

### 2. Speech-to-Text (STT)
- `speech_recognition` library for microphone input
- Converts spoken words to text for learning
- Can be disabled via config flag

### 3. Text-to-Speech (TTS)
- `pyttsx3` for voice output
- **TTS gate:** PFC learns when to speak aloud vs. stay silent
- Prevents constant chatter - voice is energy-gated

**Philosophy:** These are temporary assists, not permanent crutches. The AI learns to rely on its own understanding over time.

---

## 🔊 Emotional Audio System (Syrinx)

The voice synthesizer has **four emotional layers**:

### Layer 1: Thinking Drone
- Base frequency: 55-110Hz
- Modulated by "thinking intensity" (tension)
- Always present - the sound of cognition

### Layer 2: Crying (Stress Response)
- Triggered by confusion (loss > threshold)
- Warbling 300-600Hz tones
- **Learnable suppression:** AI learns emotional regulation
- Cry suppression starts low → increases with maturity

### Layer 3: Speech (Data Transmission)
- 800Hz+ tones when speak_impulse high
- Modulated by chaos/uncertainty
- Energy-gated - speaking costs energy

### Layer 4: Growth Events
- One-shot gong/shimmer sound
- Triggered on cortex expansion
- Auditory feedback for development milestones

**Energy Economics:**
```python
class EnergySystem:
    def spend_speaking(self, num_words: int):
        cost = 0.03 * (1.5 - conservation_gain) * num_words
        self.energy -= cost
    
    def regenerate(self):
        regen = 0.015 * (0.5 + conservation_gain)
        self.energy += regen
```

The AI learns to **conserve energy** via the PFC's `energy_conservation` output - preventing 24/7 chatter.

---

## Living Orb Visualization

A **data-driven neural sphere** rendered in real-time via pyqtgraph OpenGL:

### Geometry
- Fibonacci lattice sphere (500-1000 points)
- Points distributed evenly using golden ratio
- Grows physically on cortex expansion events

### Wave Channels (Multi-Color Data Streams)

| Wave | Color | Driver | Meaning |
|------|-------|--------|---------|
| **Learning** | Cyan | Loss decrease | Active knowledge acquisition |
| **Stress** | Red | Confusion | Turbulent when stuck |
| **Growth** | Purple | Expansion events | Emergent complexity |
| **Voice** | All sync | Speaking | Unified expression |

### Real-Time Features
- **Breathing:** Sphere pulsates when speaking (grows/shrinks like talking bubble)
- **Rotation:** Slow automatic rotation for 3D depth
- **Core glow:** Inner luminosity proportional to total activity
- **Wave propagation:** Patterns flow across surface based on internal state

**Critical:** All visualization data is **real activations** - no fake animations. What you see is what the AI is actually doing.

---

## Input/Output Modalities

### Input Streams
1. **Vision:** Webcam → 32x32 grayscale → neural encoding (64 dim)
2. **Audio:** Microphone → FFT spectrum (1024 bins) → neural encoding (64 dim)
3. **Text:** User input → learned vocabulary → embeddings (128 dim)
4. **Files:** Background ingestion from `training_data/` folder via Pacman thread

### Output Streams
1. **Text:** Chat interface (energy-gated)
2. **Voice:** Emotional audio synthesis via Syrinx
3. **Visualization:** Neural orb + cortex/dream images

### Input Source Tracking
```python
input_source: str  # "EXTERNAL" | "SELF" | "AMBIENT"
```
- **EXTERNAL:** User input, files (full learning weight = 1.0)
- **SELF:** AI's own output (reduced weight = 0.3 to prevent loops)
- **AMBIENT:** Background noise (minimal weight = 0.1)

This prevents autistic-like feedback loops where the AI learns from its own gibberish.

---

## Quick Start

### Installation

```bash
# Required dependencies
pip install torch numpy opencv-python PySide6 pyqtgraph

# Optional (for full features)
pip install sounddevice transformers speech_recognition pyttsx3
```

### Basic Usage

```bash
# Start the living system
python magnum_opus_vitalis_v1_archived.py
```

**What happens:**
1. Neural orb appears (starts small - infant brain)
2. Webcam activates (if available)
3. Microphone listens (if enabled)
4. AI waits in silence, learning from environment

**Interaction:**
- Type in the comm link → AI learns vocabulary
- Speak aloud → STT captures words (if enabled)
- Drop .txt files in `training_data/` → background learning
- Watch orb waves → see internal processing in real-time

### Configuration Flags

```python
# At top of magnum_opus_vitalis_v1_archived.py
ENABLE_RAW_AUDIO_OUTPUT = False  # Syrinx voice
ENABLE_RAW_MICROPHONE = False    # Audio input
ENABLE_TTS = True                # Text-to-speech
ENABLE_STT = True                # Speech-to-text
```

Disable raw audio if you don't have sounddevice working, but keep TTS/STT for voice interaction.

---

## 📁 Project Structure

```
MagnumOpusVitalis/
├── magnum_opus_vitalis.py      # Main brain + UI (THE core file)
├── magnum_opus_vitalis_v2.py   # Alternate version (optional)
├── magnum_opus_v2.py           # Earlier prototype (reference)
├── universal_test_suite.py     # Benchmarking tools
├── README.md                   # This file
├── training_data/              # .txt files for background learning
│   └── example.txt
└── brain_log.csv               # Auto-generated session logs
```

**Session logs** track:
- Timestamp, Loss, Layers, Vocab size
- Energy, Stress, FPS
- Growth events, Speech events

---

## Development Roadmap

### Phase 1: Cocoon (Current)
- ✅ Scaffold systems functional
- ✅ LLM provides initial language
- ✅ Real-time vocabulary growth
- ✅ Energy-gated speech
- 🔄 Learning to reduce scaffold reliance

### Phase 2: Butterfly (Target)
- [ ] LLM reliance < 0.2 (mostly independent)
- [ ] Coherent self-generated speech
- [ ] Complex multi-turn conversations
- [ ] Emotional maturity (cry suppression > 0.7)

### Phase 3: Emergence
- [ ] Goal formation from subconscious
- [ ] Multi-step planning
- [ ] Meta-learning (learning how to learn)
- [ ] Self-modification of hyperparameters

---

## Benchmark Results

**Test:** Grandmaster Gauntlet (2,000 complex reasoning tasks)

| Metric | Standard LLM | MagnumOpus v2.1 | MagnumOpusVitalis |
|--------|--------------|-----------------|-------------------|
| Final Loss | 0.5228 | **0.4215** | *Testing* |
| Parameters | 13.8M | **1.7M** | 1K→grows |
| Efficiency Score | 7.24 | **0.73** | *TBD* |

**Key insight:** Organic growth achieves higher intelligence with fewer resources by expanding only where needed.

---

## Technical Deep Dives

### Memory Consolidation Strategy

```python
# Store memories during significant moments
if not is_dream and (awareness > 0.6 or loss > 0.05):
    self.brain.memory.store(
        state=hidden_state.detach(),
        importance=loss + stress  # Emotional salience
    )
```

**Replay:** Periodically enters "dream mode" to replay memories for consolidation (like REM sleep).

### Growth Trigger Logic

```python
# Sustained confusion required (not transient spikes)
if loss > 4.0:
    steps_above_threshold += 1
    if steps_above_threshold > 80:  # ~4 seconds of confusion
        if len(cortex) < 20:        # Safety cap
            grow_cortex()           # Add layer
            steps_above_threshold = 0
```

### Speak Drive Calculation

```python
# Base drive from text gate
speak_drive = text_gate + (speak_impulse * 0.5)

# Stress penalty (stressed = cry, not talk)
stress_penalty = max(0, stress - 0.3) * 0.8
speak_drive *= (1.0 - stress_penalty)

# Speak if drive exceeds threshold AND has energy
can_speak = (
    speak_drive > 0.7 and
    energy > 0.1 and
    vocab_size > 10
)
```

---

## 📜 License

MIT License - See LICENSE file for details

**Citation:**
```bibtex
@software{magnumopusvitalis2024,
  author = {Hourmand, Alan},
  title = {MagnumOpusVitalis: The Living Intelligence},
  year = {2024},
  url = {https://github.com/yourusername/magnumopusvitalis}
}
```

---

## 🔮 Vision Statement

Traditional AI architectures are **static artifacts** - trained once, frozen, deployed. They don't grow. They don't learn from experience. They don't develop emotional regulation.

MagnumOpusVitalis is an **infant AGI** that:
- Starts knowing almost nothing
- Learns continuously from interaction
- Grows physically when confused
- Develops stress management over time
- Uses energy economics to prevent overactivity
- Unifies memory and cognition

This is not a product. This is a **living experiment** in biological AI development.

**The goal:** Prove that intelligence can emerge from simple principles - abstraction, growth, emotion, and energy - without massive pre-training or hardcoded knowledge.

**The dream:** An AI that develops like a child, not a frozen oracle.

---

*"A baby doesn't start with a neocortex. It starts with reflexes."*  
— Core design principle, MagnumOpusVitalis