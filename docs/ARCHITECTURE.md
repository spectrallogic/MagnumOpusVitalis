# MagnumOpusVitalis Architecture

*Technical deep-dive into all systems*

---

## Overview

MagnumOpusVitalis is a **growing neural architecture** with biologically-inspired subsystems. Unlike traditional transformers that are fixed at initialization, this architecture:

1. **Grows** when learning plateaus (not at first sign of confusion)
2. **Regulates itself** via learned PFC outputs
3. **Dreams** via a 4-layer subconscious pipeline
4. **Remembers** via episodic memory unified with the model
5. **Fatigues** via computational energy budgets

```
┌─────────────────────────────────────────────────────────────────┐
│                    MagnumOpusVitalis Brain                      │
├─────────────────────────────────────────────────────────────────┤
│  Input → Embed → MultiSpeed → PFC Control → Memory Recall       │
│                                    ↓                            │
│                           Subconscious (4 layers)               │
│                                    ↓                            │
│                         Temporal Resonance                      │
│                                    ↓                            │
│                    Cortex Layers (energy-gated)                 │
│                                    ↓                            │
│                              Output                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Growth Controller

**Purpose:** Decide WHEN to grow the network.

**Philosophy:** *"Learn first, grow last"*

### The Problem with Naive Growth

```python
# BAD: Grows too eagerly
if loss > 3.5:
    grow()  # Triggers constantly during early training!
```

### Patient Growth Strategy

```python
class GrowthController:
    def should_grow(self) -> (bool, reason):
        # Safety: Don't grow if too young
        if step < min_age_for_growth:
            return False, "too_young"
        
        # Safety: Don't grow if loss is already good
        if best_loss < 0.5:
            return False, "loss_ok"
        
        # Check if improving
        if loss < best_loss * 0.99:  # 1% improvement
            reset_patience()
            return False, "learning"
        
        # On plateau: Try LR boost FIRST
        if plateau_counter > patience:
            if lr_boost_count < 3:
                lr_boost_count += 1
                return False, "boost_lr"  # Try 2x, 4x, 8x LR
            
            # All boosts exhausted → finally grow
            return True, "grow"
        
        return False, "learning"
```

### Growth Types

| Type | What Changes | Cost | When to Use |
|------|--------------|------|-------------|
| **Depth** | Add new cortex layer | Low | First choice |
| **Width** | Increase hidden dim by 16 | High (rebuilds all layers) | When depth maxed |

### Near-Identity Initialization

New layers start with `residual_scale = 0.1` to minimize disruption:

```python
def grow_depth(self):
    new_layer = GrowableCortexLayer(dim)
    new_layer.residual_scale = 0.1  # Almost identity at first
    self.cortex.append(new_layer)
```

---

## 2. The 4-Layer Subconscious

**Purpose:** Generate creativity, simulate futures, form emergent goals.

```
Input (hidden state from cortex)
            ↓
┌───────────────────────────────────────┐
│  LAYER 0: Sea of Noise                │
│  - Learned basis vectors (32 × dim)   │
│  - NOT random noise, but LEARNED      │
│  - Scaled by stress (more creative    │
│    when confused)                     │
└───────────────────────────────────────┘
            ↓
┌───────────────────────────────────────┐
│  LAYER 1: Peak Detector               │
│  - Sigmoid gating on activations      │
│  - Learns what's "salient"            │
│  - Filters noise to signal            │
└───────────────────────────────────────┘
            ↓
┌───────────────────────────────────────┐
│  LAYER 2: Future Generator            │
│  - GRU cell for temporal simulation   │
│  - Maintains momentum buffer          │
│  - "What might happen next?"          │
└───────────────────────────────────────┘
            ↓
┌───────────────────────────────────────┐
│  LAYER 3: Scenario Evaluator          │
│  - Scores simulated futures (0-1)     │
│  - Weights output by quality          │
│  - "Is this a good path?"             │
└───────────────────────────────────────┘
            ↓
      Goal Momentum (persistent)
```

### Goal Momentum

The subconscious maintains a **goal momentum** buffer that accumulates across steps:

```python
# In SubconsciousMind.forward():
self.goal_momentum = 0.95 * self.goal_momentum + 0.05 * evaluated_output
```

This creates **emergent desires** — the AI develops persistent intentions from experience, not hardcoded objectives.

### Stress-Modulated Creativity

```python
noise_scale = 1.0 + stress * 0.5  # More creative when confused
s0 = self.sea(x, noise_scale)
```

When the AI is stressed (high loss), the Sea of Noise amplifies, generating more creative possibilities.

---

## 3. Prefrontal Cortex (Executive Control)

**Purpose:** Learn to regulate the entire system.

### The 8 Control Outputs

```python
class PrefrontalCortex(nn.Module):
    # GRU for temporal decision-making
    self.planner = nn.GRUCell(dim + 2, dim)  # +2 for stress, energy
    
    # Policy outputs 8 sigmoid values
    self.policy = nn.Linear(dim, 8)
```

| Index | Name | Initial | Purpose |
|-------|------|---------|---------|
| 0 | `vis_gain` | 0.5 | Modulate visual attention |
| 1 | `aud_gain` | 0.5 | Modulate auditory attention |
| 2 | `dream_gain` | 0.5 | How much subconscious to mix in |
| 3 | `speak_impulse` | **0.73** | Drive to vocalize |
| 4 | `cry_suppression` | **0.27** | Emotional regulation |
| 5 | `energy_conservation` | 0.5 | Resource management |
| 6 | `creativity_boost` | 0.62 | Amplify subconscious noise |
| 7 | `focus_level` | 0.5 | Attention sharpness |

### Developmental Trajectory

**Critical insight:** `cry_suppression` starts LOW (0.27).

```python
# In __init__:
self.policy[0].bias[4] = -1.0  # sigmoid(-1) ≈ 0.27
```

This means:
- **Early life:** AI "cries" a lot (low suppression)
- **Over training:** PFC learns to increase suppression
- **Maturity:** AI regulates emotions (high suppression)

This is **learned behavior**, not hardcoded — the gradient flows through and the AI discovers that suppressing crying (in certain contexts) leads to better outcomes.

### How Controls Are Used

```python
# In forward():

# Energy conservation affects the energy system
self.energy.set_conservation(energy_conservation)

# Dream gain controls subconscious mixing
x = x + subconscious_output * dream_gain * creativity_boost

# Focus level affects memory recall
memory_weight = 0.1 * (1 - focus_level)  # Less memory when focused
```

---

## 4. Multi-Speed Processor

**Purpose:** Process information at multiple timescales, like biological brains.

### The Three Channels

| Channel | Dimension | Analogy | Example |
|---------|-----------|---------|---------|
| **Fast** | 16 | Reflexes | "That's blue" |
| **Medium** | 32 | Recognition | "That's the sky" |
| **Slow** | 64 | Understanding | "Rayleigh scattering causes blue sky" |

### Trust Weights

```python
class MultiSpeedProcessor(nn.Module):
    def __init__(self):
        self.trust = nn.Parameter(torch.tensor([1.0, 0.3, 0.1]))
        #                                       fast  med   slow
```

**Age-based shifting:**

```python
def age_step(self):
    self.age += 1
    if self.age % 1000 == 0:
        self.trust[0] *= 0.99  # Fast trust decreases
        self.trust[2] *= 1.01  # Slow trust increases
```

Over time, the AI shifts from **reactive** (fast-dominated) to **reflective** (slow-influenced).

### Forward Pass

```python
def forward(self, x):
    fast = tanh(self.fast(x))    # [batch, seq, 16]
    medium = tanh(self.medium(x)) # [batch, seq, 32]
    slow = tanh(self.slow(x))     # [batch, seq, 64]
    
    weights = softmax(self.trust)  # Normalized
    
    combined = concat([
        fast * weights[0],
        medium * weights[1],
        slow * weights[2]
    ])
    
    return self.combine(combined)  # Back to output_dim
```

---

## 5. Temporal Resonance

**Purpose:** Maintain temporal coherence without explicit recurrence in the main path.

### Exponential Moving Average

```python
class TemporalResonance(nn.Module):
    def __init__(self, dim, decay=0.96):
        self.resonance_state = zeros(1, dim)  # EMA buffer
        self.clock_phase = 0.0                # Rhythmic oscillator
```

### The Resonance Mechanism

```python
def forward(self, x):
    # Update EMA (temporal memory)
    self.resonance_state = x.mean() * 0.04 + self.resonance_state * 0.96
    
    # Update clock (rhythmic modulation)
    self.clock_phase += 0.1  # radians per step
    
    # Apply modulation
    clock_mod = 1.0 + 0.05 * sin(clock_phase)
    
    return x + self.resonance_state * 0.1 * clock_mod
```

### Why This Matters

- **Temporal continuity:** The model "remembers" recent activations
- **Rhythmic processing:** Natural oscillations emerge (like brain waves)
- **No explicit RNN:** Adds temporal context without recurrence overhead

---

## 6. Biological Memory

**Purpose:** Episodic memory that's **part of the model**, not a separate database.

### Key Properties

| Property | Description |
|----------|-------------|
| **Episodic** | Stores "moments" (embeddings), not facts |
| **Reconstructive** | Recall by pattern matching, not lookup |
| **Importance-weighted** | Emotional salience affects retention |
| **Decaying** | Memories fade unless reinforced |

### Storage

```python
def store(self, state, importance):
    if importance < 0.3:
        return  # Not significant enough
    
    encoded = self.encoder(state)
    
    # Circular buffer
    self.memory_bank[write_head] = encoded
    self.importance[write_head] = importance
    self.timestamps[write_head] = time.time()
    
    write_head = (write_head + 1) % capacity
```

### Recall

```python
def recall(self, query, top_k=3):
    # Cosine similarity with all memories
    similarities = cosine_sim(query, memory_bank)
    
    # Weight by importance AND recency
    time_decay = exp(-0.001 * (now - timestamps))
    weighted = similarities * importance * time_decay
    
    # Get top-k
    indices = topk(weighted, k=top_k)
    
    # Only recall if similar enough
    if weighted[indices[0]] < 0.5:
        return None
    
    # Reconstruct
    return self.decoder(memory_bank[indices].mean())
```

### Significance Triggers

Memories are stored when:
- **High loss** (confusion is memorable)
- **High stress** (emotional salience)

```python
significance = loss + stress * 0.5
if significance > 0.5:
    self.memory.store(hidden_state, importance=significance)
```

---

## 7. Computational Energy

**Purpose:** Ground "energy" in actual compute, not arbitrary numbers.

### Two Energy Systems

| System | Units | Purpose |
|--------|-------|---------|
| **Computational** | FLOPs | Gates expensive operations |
| **Biological** | 0-1 scale | Gates speech, enables fatigue |

### FLOPs Estimation

```python
def estimate_attention_flops(seq_len, dim, heads, batch):
    # O(seq^2 * dim) for attention
    return batch * heads * seq_len * seq_len * (dim // heads) * 4

def estimate_linear_flops(in_dim, out_dim, batch):
    return 2 * in_dim * out_dim * batch
```

### Energy-Gated Processing

```python
for layer in self.cortex:
    flops = estimate_attention_flops(...)
    
    if self.energy.can_afford(flops):
        x = layer(x)
        self.energy.spend(flops)
    else:
        pass  # Skip layer if out of budget!
```

### Biological Energy

```python
def can_speak(self):
    return self.biological_energy > 0.2  # Fatigue threshold

def spend_speaking(self, num_words):
    cost = 0.03 * (1.5 - conservation) * num_words
    self.biological_energy -= cost

def regenerate(self):
    regen = 0.015 * (0.5 + conservation)
    self.biological_energy += regen
```

**Key insight:** The PFC's `energy_conservation` output modulates both cost and regeneration. The AI **learns** to conserve energy.

---

## 8. Growable Cortex Layers

**Purpose:** Modern transformer blocks that can be added dynamically.

### Architecture

```python
class GrowableCortexLayer(nn.Module):
    def __init__(self, dim, num_heads=4):
        # Pre-norm (more stable than post-norm)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Self-attention
        self.attn = MultiheadAttention(dim, num_heads)
        
        # Gated MLP (SwiGLU-style)
        self.mlp = GatedMLP(dim)
        
        # Learnable residual scale
        self.residual_scale = Parameter(ones(1))
```

### Why These Choices?

| Component | Why |
|-----------|-----|
| **RMSNorm** | Simpler than LayerNorm, equally effective |
| **Pre-norm** | More stable training, better gradients |
| **GatedMLP** | More expressive than standard FFN (SwiGLU) |
| **Learnable residual** | New layers can start near-identity |

### GatedMLP (SwiGLU)

```python
class GatedMLP(nn.Module):
    def forward(self, x):
        gate = silu(self.gate(x))  # Gating path
        up = self.up(x)            # Value path
        return self.down(gate * up)
```

---

## 9. Complete Forward Pass

```python
def forward(self, input_ids, targets=None):
    # 1. EMBEDDING
    x = self.embed(input_ids)
    
    # 2. MULTI-SPEED PROCESSING
    x = self.multi_speed(x)
    
    # 3. PREFRONTAL CORTEX
    actions, self.pfc_state = self.pfc(x, stress, energy, self.pfc_state)
    # Extract: dream_gain, speak_impulse, cry_suppression, etc.
    
    # 4. MEMORY RECALL
    memory = self.memory.recall(x)
    if memory is not None:
        x = x + memory * 0.1 * (1 - focus_level)
    
    # 5. SUBCONSCIOUS
    sub_out = self.subconscious(x, stress)
    x = x + sub_out['output'] * dream_gain * creativity_boost
    
    # 6. TEMPORAL RESONANCE
    x = self.temporal(x)
    
    # 7. CORTEX LAYERS (energy-gated)
    for layer in self.cortex:
        if self.energy.can_afford(layer_flops):
            x = layer(x, causal_mask)
            self.energy.spend(layer_flops)
    
    # 8. OUTPUT
    logits = self.output_proj(self.output_norm(x))
    
    # 9. MEMORY STORAGE (if training)
    if loss is not None:
        significance = loss + stress * 0.5
        if significance > 0.5:
            self.memory.store(x, significance)
    
    return {'logits': logits, 'loss': loss, ...}
```

---

## 10. Configuration

```python
@dataclass
class GrowthConfig:
    # Dimensions
    initial_dim: int = 64      # Starting hidden size
    max_dim: int = 512         # Maximum hidden size
    max_layers: int = 12       # Maximum cortex depth
    
    # Patient growth
    plateau_patience: int = 200      # Steps before considering growth
    lr_boost_attempts: int = 3       # Try LR boost before growing
    improvement_threshold: float = 0.99  # 1% improvement resets patience
    
    # Safety
    min_loss_for_growth: float = 0.5    # Don't grow if loss is good
    min_age_for_growth: int = 500       # Minimum steps before first growth
```

---

## Summary

| System | Key Insight |
|--------|-------------|
| **GrowthController** | Learn first, boost LR, grow as last resort |
| **SubconsciousMind** | Emergent goals from 4-layer creative pipeline |
| **PrefrontalCortex** | Learned regulation (cry suppression, energy conservation) |
| **MultiSpeedProcessor** | Fast→slow trust shift with age |
| **TemporalResonance** | EMA + clock creates temporal context |
| **BiologicalMemory** | Memory IS the model, not separate |
| **ComputationalEnergy** | FLOPs-grounded, enables learned conservation |
| **GrowableCortex** | Modern transformers, near-identity init |

The architecture is designed so that **nothing is hardcoded** — all regulation, goals, and growth emerge from training.