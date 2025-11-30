"""
MagnumOpusVitalis Core Algorithm (COMPLETE)
============================================
A neural architecture that GROWS when confused, instead of being pre-sized.

Key Innovation: Instead of choosing model size upfront (1B? 7B? 70B?),
start microscopic (~1K params) and grow capacity only when learning plateaus.

CORE SYSTEMS:
1. GrowthController - Patient growth (learn first, grow last)
2. ComputationalEnergy - FLOPs-based budget + biological energy
3. SubconsciousMind - 4-layer creative pipeline + goal momentum
4. PrefrontalCortex - 8 learned control outputs
5. MultiSpeedProcessor - Fast/medium/slow channels with trust shifting
6. TemporalResonance - Temporal coherence without explicit recurrence
7. BiologicalMemory - Episodic, reconstructive, importance-weighted
8. GrowableCortexLayers - Modern transformer blocks

This file contains the COMPLETE algorithm - no UI, audio, or visualization.
For the full experience, see demo/magnum_opus_demo.py

Author: Alan Hourmand
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GrowthConfig:
    """Configuration for organic growth behavior"""
    initial_dim: int = 64                    # Starting hidden dimension
    max_dim: int = 512                       # Maximum dimension (safety cap)
    max_layers: int = 12                     # Maximum depth (safety cap)

    # Patient growth parameters (researcher feedback: "learn first, grow last")
    plateau_patience: int = 200              # Steps before considering growth
    lr_boost_attempts: int = 3               # Try boosting LR this many times first
    improvement_threshold: float = 0.99      # Loss must improve by 1% to reset patience

    # Growth triggers
    min_loss_for_growth: float = 0.5         # Don't grow if loss is already low
    min_age_for_growth: int = 500            # Minimum steps before first growth


@dataclass
class TrainingState:
    """Tracks training progress for growth decisions"""
    step: int = 0
    best_loss: float = float('inf')
    plateau_counter: int = 0
    lr_boost_count: int = 0
    last_growth_step: int = 0
    total_growths: int = 0
    loss_history: List[float] = field(default_factory=list)


# ============================================================================
# COMPUTATIONAL ENERGY SYSTEM
# ============================================================================

class ComputationalEnergy:
    """
    Energy based on actual compute budget, not arbitrary numbers.

    Researcher feedback: "Energy needs to be actual energy in some way"

    This tracks estimated FLOPs and gates expensive operations.
    Also includes biological-style regeneration and conservation learning.
    """

    def __init__(self, budget_per_step: int = 1_000_000):
        self.budget = budget_per_step        # FLOPs allowed per step
        self.spent_this_step = 0
        self.total_spent = 0
        self.recovery_rate = 0.8             # Fraction of budget recovered per step

        # Biological energy (0-1 scale for speech gating, etc.)
        self.biological_energy = 1.0
        self.fatigue_threshold = 0.2         # Below this = too tired to speak

        # Learned conservation factor (0 = wasteful, 1 = very conservative)
        # This gets SET by PFC output
        self.conservation = 0.5

    def estimate_linear_flops(self, in_features: int, out_features: int, batch: int = 1) -> int:
        """Estimate FLOPs for a linear layer: 2 * in * out * batch"""
        return 2 * in_features * out_features * batch

    def estimate_attention_flops(self, seq_len: int, dim: int, heads: int, batch: int = 1) -> int:
        """Estimate FLOPs for self-attention: O(seq^2 * dim)"""
        return batch * heads * seq_len * seq_len * (dim // heads) * 4

    def can_afford(self, estimated_flops: int) -> bool:
        """Check if we have budget for this operation"""
        effective_budget = self.budget * (0.5 + 0.5 * self.conservation)
        return (effective_budget - self.spent_this_step) >= estimated_flops

    def can_speak(self) -> bool:
        """Check if we have enough biological energy to speak"""
        return self.biological_energy > self.fatigue_threshold

    def spend(self, flops: int):
        """Record compute expenditure"""
        self.spent_this_step += flops
        self.total_spent += flops

    def spend_speaking(self, num_words: int = 1):
        """Speaking costs biological energy"""
        cost = 0.03 * (1.5 - self.conservation) * num_words
        self.biological_energy = max(0, self.biological_energy - cost)

    def step_reset(self):
        """Called at end of each step - partial recovery"""
        recovered = int(self.spent_this_step * self.recovery_rate)
        self.spent_this_step = max(0, self.spent_this_step - recovered)

        # Biological energy regeneration
        regen = 0.015 * (0.5 + self.conservation)
        self.biological_energy = min(1.0, self.biological_energy + regen)

    def set_conservation(self, value: float):
        """PFC learns to set this"""
        self.conservation = max(0.0, min(1.0, value))

    @property
    def utilization(self) -> float:
        """Current budget utilization (0-1)"""
        return min(1.0, self.spent_this_step / self.budget)


# ============================================================================
# GROWTH CONTROLLER
# ============================================================================

class GrowthController:
    """
    Decides WHEN to grow the network.

    Researcher feedback: "Should try learning first, grow as last resort"

    Strategy:
    1. If loss is improving -> keep learning
    2. If loss plateaus -> try boosting learning rate
    3. If still stuck after multiple LR boosts -> grow
    """

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.state = TrainingState()

    def record_loss(self, loss: float):
        """Record a loss value and update state"""
        self.state.step += 1
        self.state.loss_history.append(loss)

        # Keep only recent history
        if len(self.state.loss_history) > 1000:
            self.state.loss_history = self.state.loss_history[-500:]

        # Check if improving
        if loss < self.state.best_loss * self.config.improvement_threshold:
            # Significant improvement - reset patience
            self.state.best_loss = loss
            self.state.plateau_counter = 0
            self.state.lr_boost_count = 0
        else:
            # Not improving - increment plateau counter
            self.state.plateau_counter += 1

    def should_grow(self) -> Tuple[bool, str]:
        """
        Returns (should_grow, reason)

        Reason can be: "learning", "boost_lr", "grow", "too_young", "loss_ok"
        """
        # Safety checks
        if self.state.step < self.config.min_age_for_growth:
            return False, "too_young"

        if self.state.best_loss < self.config.min_loss_for_growth:
            return False, "loss_ok"

        # Check if we're on a plateau
        if self.state.plateau_counter < self.config.plateau_patience:
            return False, "learning"

        # We're on a plateau - try LR boost first
        if self.state.lr_boost_count < self.config.lr_boost_attempts:
            self.state.lr_boost_count += 1
            self.state.plateau_counter = 0  # Reset to give LR boost a chance
            return False, "boost_lr"

        # All LR boosts exhausted - time to grow
        self.state.plateau_counter = 0
        self.state.lr_boost_count = 0
        self.state.last_growth_step = self.state.step
        self.state.total_growths += 1
        return True, "grow"

    def get_recommended_lr_multiplier(self) -> float:
        """Get LR multiplier based on current state"""
        if self.state.lr_boost_count > 0:
            # Progressively larger boosts: 2x, 4x, 8x
            return 2.0 ** self.state.lr_boost_count
        return 1.0


# ============================================================================
# MODERN TRANSFORMER COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class GatedMLP(nn.Module):
    """
    Gated MLP (like LLaMA's SwiGLU).
    More expressive than standard FFN.
    """

    def __init__(self, dim: int, hidden_mult: float = 2.67):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GrowableCortexLayer(nn.Module):
    """
    A single transformer-like layer that can grow.

    Uses modern practices:
    - Pre-norm (more stable training)
    - RMSNorm (simpler, works well)
    - Gated MLP (more expressive)
    - Residual connections with learnable scale
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Pre-norm
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)

        # Gated MLP
        self.mlp = GatedMLP(dim)

        # Learnable residual scale (starts at 1, can adjust)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h * self.residual_scale

        # Pre-norm MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h * self.residual_scale

        return x


# ============================================================================
# 4-LAYER SUBCONSCIOUS SYSTEM
# ============================================================================

class SeaOfNoise(nn.Module):
    """
    Layer 0: Creative randomness from LEARNED patterns.
    Uses learned basis vectors, not hardcoded noise.
    """

    def __init__(self, dim: int, num_basis: int = 32):
        super().__init__()
        # Learned basis vectors for creative noise
        self.noise_basis = nn.Parameter(torch.randn(num_basis, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        # Sample from learned noise basis
        batch = x.shape[0] if x.dim() > 1 else 1
        weights = torch.randn(batch, self.noise_basis.shape[0], device=x.device) * 0.15 * noise_scale
        creative_noise = weights @ self.noise_basis

        if x.dim() == 1:
            creative_noise = creative_noise.squeeze(0)

        return self.proj(x + creative_noise)


class PeakDetector(nn.Module):
    """
    Layer 1: Filter for relevant activations.
    Gates input based on learned salience.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.scorer(x))
        return x * gates


class FutureGenerator(nn.Module):
    """
    Layer 2: Simulate possible outcomes.
    Uses GRU for temporal simulation with momentum.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.register_buffer('momentum', torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Resize momentum if needed
        if self.momentum.shape[1] != x.shape[-1]:
            self.momentum = torch.zeros(1, x.shape[-1], device=x.device)

        # Handle batch dimension
        if x.dim() == 3:
            # [batch, seq, dim] -> use last token
            x_flat = x[:, -1, :]
        else:
            x_flat = x

        # Expand momentum for batch
        momentum = self.momentum.expand(x_flat.shape[0], -1)

        future = self.gru(x_flat, momentum)

        # Update momentum (detached to prevent gradient flow)
        self.momentum = (0.95 * self.momentum + 0.05 * future.mean(0, keepdim=True)).detach()

        return future, x_flat


class ScenarioEvaluator(nn.Module):
    """
    Layer 3: Assess path quality.
    Scores and weights the simulated future.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.judge = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.judge(x)
        return x * score, score


class SubconsciousMind(nn.Module):
    """
    4-Layer Subconscious Pipeline:

    1. Sea of Noise - Creative randomness from LEARNED patterns
    2. Peak Detector - Filter for relevant activations
    3. Future Generator - Simulate possible outcomes
    4. Evaluator - Pick the best path

    Also maintains goal_momentum for emergent goal formation.
    The AI develops intentions from experience, not hardcoded goals.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # The 4 layers
        self.sea = SeaOfNoise(dim)
        self.peak = PeakDetector(dim)
        self.future = FutureGenerator(dim)
        self.evaluator = ScenarioEvaluator(dim)

        # Output projection
        self.output = nn.Linear(dim, dim)

        # Goal momentum - emergent attractor for desires/intentions
        self.register_buffer('goal_momentum', torch.zeros(1, dim))

    def forward(self, x: torch.Tensor, stress: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Process through subconscious pipeline.

        Args:
            x: Input hidden state
            stress: Current stress level (0-1), increases creative noise

        Returns:
            Dictionary with 'output', 'value', 'goal_momentum'
        """
        # Layer 0: Sea of Noise (more creative when stressed)
        noise_scale = 1.0 + stress * 0.5
        s0 = self.sea(x, noise_scale)

        # Layer 1: Peak Detection
        s1 = self.peak(s0)

        # Layer 2: Future Generation
        s2, filtered = self.future(s1)

        # Layer 3: Evaluation
        s3, value = self.evaluator(s2)

        # Update goal momentum (emergent goals from experience)
        if s3.dim() == 2 and s3.shape[0] > 0:
            # Average over batch, update momentum
            new_goal = s3.mean(0, keepdim=True)
            if new_goal.shape == self.goal_momentum.shape:
                self.goal_momentum = (0.95 * self.goal_momentum + 0.05 * new_goal).detach()

        output = self.output(s3)

        return {
            'output': output,
            'value': value,
            'goal_momentum': self.goal_momentum.clone()
        }


# ============================================================================
# PREFRONTAL CORTEX (EXECUTIVE CONTROL)
# ============================================================================

class PrefrontalCortex(nn.Module):
    """
    The executive controller that learns to regulate the entire system.

    Outputs 8 learned control signals:
    0. vis_gain: Visual attention modulation
    1. aud_gain: Auditory attention modulation
    2. dream_gain: How much imagination to mix in
    3. speak_impulse: Drive to vocalize
    4. cry_suppression: Learned emotional regulation (increases with maturity)
    5. energy_conservation: Learned resource management
    6. creativity_boost: Subconscious noise amplification
    7. focus_level: Attention sharpness

    Critical insight: The AI LEARNS to control its own responses.
    Nothing is hardcoded - regulation emerges from experience.
    """

    def __init__(self, dim: int):
        super().__init__()

        # GRU for temporal context in decision-making
        self.planner = nn.GRUCell(dim + 2, dim)  # +2 for stress and energy

        # Policy network outputs 8 control signals
        self.policy = nn.Sequential(
            nn.Linear(dim, 8),
            nn.Sigmoid()
        )

        # Initialize biases for developmental trajectory
        with torch.no_grad():
            # speak_impulse starts moderate
            self.policy[0].bias[3] = 1.0   # ~0.73
            # cry_suppression starts LOW (infants cry a lot)
            self.policy[0].bias[4] = -1.0  # ~0.27
            # energy_conservation starts neutral
            self.policy[0].bias[5] = 0.0   # ~0.5
            # creativity starts moderate
            self.policy[0].bias[6] = 0.5   # ~0.62

    def forward(
        self,
        h: torch.Tensor,
        stress: float,
        energy: float,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate control signals based on current state.

        Args:
            h: Current hidden state
            stress: Current stress level (0-1)
            energy: Current energy level (0-1)
            prev_state: Previous PFC state

        Returns:
            (actions, new_state) where actions is [batch, 8]
        """
        if prev_state is None:
            prev_state = torch.zeros_like(h)
        else:
            prev_state = prev_state.detach()

        # Ensure h is 2D
        if h.dim() == 3:
            h = h[:, -1, :]  # Take last token
        if h.dim() == 1:
            h = h.unsqueeze(0)

        # Include stress and energy as context
        context = torch.tensor([[stress, energy]], device=h.device).expand(h.shape[0], -1)
        inp = torch.cat([h, context], dim=1)

        # Generate new state and actions
        new_state = self.planner(inp, prev_state)
        actions = self.policy(new_state)

        return actions, new_state


# ============================================================================
# TEMPORAL RESONANCE SYSTEM
# ============================================================================

class TemporalResonance(nn.Module):
    """
    Maintains temporal coherence without explicit recurrence.

    The model "remembers" recent activations through resonance -
    an exponential moving average that creates temporal context.

    Also includes a clock phase for rhythmic modulation,
    creating natural oscillations in processing.
    """

    def __init__(self, dim: int, decay: float = 0.96):
        super().__init__()
        self.dim = dim
        self.decay = decay

        # Resonance state (EMA of activations)
        self.register_buffer('resonance_state', torch.zeros(1, dim))

        # Clock for rhythmic modulation
        self.register_buffer('clock_phase', torch.tensor(0.0))
        self.clock_speed = 0.1  # Radians per step

        # Learnable modulation strength
        self.modulation_strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal resonance to input.

        Returns input modulated by:
        1. Exponential moving average (temporal memory)
        2. Clock phase (rhythmic processing)
        """
        # Ensure x is 2D
        if x.dim() == 3:
            x_mean = x.mean(dim=1)  # Average over sequence
        elif x.dim() == 1:
            x_mean = x.unsqueeze(0)
        else:
            x_mean = x

        # Resize resonance state if needed
        if self.resonance_state.shape[1] != x_mean.shape[-1]:
            self.resonance_state = torch.zeros(1, x_mean.shape[-1], device=x.device)

        # Update resonance state (EMA)
        self.resonance_state = (
            x_mean.mean(0, keepdim=True) * (1 - self.decay) +
            self.resonance_state * self.decay
        ).detach()

        # Update clock
        self.clock_phase = (self.clock_phase + self.clock_speed) % (2 * math.pi)

        # Apply modulation
        clock_mod = 1.0 + self.modulation_strength * torch.sin(self.clock_phase)
        resonance_contribution = self.resonance_state * 0.1

        # Add resonance to original input
        if x.dim() == 3:
            return x + resonance_contribution.unsqueeze(1) * clock_mod
        else:
            return x + resonance_contribution * clock_mod


# ============================================================================
# MULTI-SPEED PROCESSING
# ============================================================================

class MultiSpeedProcessor(nn.Module):
    """
    Processes information at multiple timescales simultaneously.

    Philosophy: A baby recognizes "blue sky" instantly but takes years
    to understand atmospheric physics.

    - Fast channel (16 dim): Pattern recognition, instant
    - Medium channel (32 dim): Structural understanding, hours-days
    - Slow channel (64 dim): Deep comprehension, weeks+

    Trust weights shift over time - fast dominates early, slow gains influence with age.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Multi-speed channels
        self.fast = nn.Linear(input_dim, 16)      # Instant reactions
        self.medium = nn.Linear(input_dim, 32)    # Developing understanding
        self.slow = nn.Linear(input_dim, 64)      # Deep knowledge

        # Combine to output dimension
        self.combine = nn.Linear(16 + 32 + 64, output_dim)

        # Trust weights (learnable, but initialized to favor fast)
        # These will naturally shift as the model learns
        self.trust = nn.Parameter(torch.tensor([1.0, 0.3, 0.1]))

        # Age counter for trust shifting
        self.register_buffer('age', torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process at each speed
        fast_out = torch.tanh(self.fast(x))
        medium_out = torch.tanh(self.medium(x))
        slow_out = torch.tanh(self.slow(x))

        # Apply trust weights (softmax to sum to 1)
        trust_weights = F.softmax(self.trust, dim=0)

        # Weighted combination
        combined = torch.cat([
            fast_out * trust_weights[0],
            medium_out * trust_weights[1],
            slow_out * trust_weights[2]
        ], dim=-1)

        return self.combine(combined)

    def age_step(self):
        """Call each training step to gradually shift trust toward slower channels"""
        self.age += 1

        # Every 1000 steps, slightly increase slow channel trust
        if self.age % 1000 == 0:
            with torch.no_grad():
                self.trust[0] *= 0.99  # Decrease fast trust
                self.trust[2] *= 1.01  # Increase slow trust


# ============================================================================
# BIOLOGICAL MEMORY (Unified with Model)
# ============================================================================

class BiologicalMemory(nn.Module):
    """
    Memory IS the model's internal state, not a separate database.

    Key properties:
    - Episodic: Stores embeddings of "significant" moments (high loss, high emotion)
    - Reconstructive: Retrieval by pattern matching, hallucinates details
    - Importance-weighted: Emotional salience affects retention
    - Decay: Memories fade over time unless reinforced
    """

    def __init__(self, dim: int, capacity: int = 500):
        super().__init__()
        self.dim = dim
        self.capacity = capacity

        # Memory bank as parameters (part of the model, not separate)
        self.register_buffer('memory_bank', torch.zeros(capacity, dim))
        self.register_buffer('importance', torch.zeros(capacity))
        self.register_buffer('timestamps', torch.zeros(capacity))
        self.register_buffer('access_counts', torch.zeros(capacity))

        self.write_head = 0
        self.num_memories = 0

        # Encoder/decoder for reconstruction
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, dim)

    def store(self, state: torch.Tensor, importance: float = 1.0):
        """Store a memory if significant enough"""
        if importance < 0.3:
            return  # Not significant enough

        with torch.no_grad():
            # Encode state
            if state.dim() == 3:
                state = state[:, -1, :]  # Last token
            if state.dim() == 2:
                state = state.mean(0)  # Average over batch

            # Resize if needed
            if state.shape[-1] != self.dim:
                return  # Dimension mismatch, skip

            encoded = self.encoder(state.detach())

            # Write to memory bank (circular buffer)
            self.memory_bank[self.write_head] = encoded
            self.importance[self.write_head] = importance
            self.timestamps[self.write_head] = time.time()
            self.access_counts[self.write_head] = 1

            self.write_head = (self.write_head + 1) % self.capacity
            self.num_memories = min(self.num_memories + 1, self.capacity)

    def recall(self, query: torch.Tensor, top_k: int = 3) -> Optional[torch.Tensor]:
        """Retrieve most similar memories via pattern matching"""
        if self.num_memories == 0:
            return None

        # Prepare query
        if query.dim() == 3:
            query = query[:, -1, :]
        if query.dim() == 2:
            query = query.mean(0)

        query_flat = query.detach().view(-1)[:self.dim]
        if query_flat.shape[0] < self.dim:
            return None  # Query too small

        # Compute similarities with all memories
        active_memories = self.memory_bank[:self.num_memories]
        similarities = F.cosine_similarity(
            query_flat.unsqueeze(0),
            active_memories,
            dim=1
        )

        # Weight by importance and recency
        time_decay = torch.exp(-0.001 * (time.time() - self.timestamps[:self.num_memories]))
        weighted_sim = similarities * self.importance[:self.num_memories] * time_decay

        # Get top-k
        top_k = min(top_k, self.num_memories)
        values, indices = torch.topk(weighted_sim, top_k)

        # Only recall if similarity is high enough
        if values[0] < 0.5:
            return None

        # Update access counts
        self.access_counts[indices] += 1

        # Reconstruct from top memories
        retrieved = active_memories[indices].mean(dim=0)
        return self.decoder(retrieved)

    def decay_memories(self, factor: float = 0.999):
        """Gradually decay importance of all memories"""
        with torch.no_grad():
            self.importance[:self.num_memories] *= factor


# ============================================================================
# THE GROWING BRAIN (COMPLETE)
# ============================================================================

class MagnumOpusCore(nn.Module):
    """
    The core growing neural architecture.

    Starts microscopic (~1K params), grows when confused.
    All knowledge is learned, nothing hardcoded.

    COMPLETE with all systems:
    - GrowthController: Patient growth (learn first, grow last)
    - ComputationalEnergy: FLOPs-based budget
    - SubconsciousMind: 4-layer creative pipeline + goal momentum
    - PrefrontalCortex: 8 learned control outputs
    - MultiSpeedProcessor: Fast/medium/slow with trust shifting
    - TemporalResonance: Temporal coherence
    - BiologicalMemory: Episodic, reconstructive
    - GrowableCortexLayers: Modern transformer blocks
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        config: Optional[GrowthConfig] = None
    ):
        super().__init__()

        self.config = config or GrowthConfig()
        self.vocab_size = vocab_size
        self.current_dim = self.config.initial_dim

        # Core controllers
        self.growth_ctrl = GrowthController(self.config)
        self.energy = ComputationalEnergy()

        # Input embedding
        self.embed = nn.Embedding(vocab_size, self.current_dim)

        # Multi-speed processor for inputs
        self.multi_speed = MultiSpeedProcessor(self.current_dim, self.current_dim)

        # Subconscious system (4-layer creative pipeline)
        self.subconscious = SubconsciousMind(self.current_dim)

        # Prefrontal cortex (executive control)
        self.pfc = PrefrontalCortex(self.current_dim)
        self.pfc_state = None

        # Temporal resonance (temporal coherence)
        self.temporal = TemporalResonance(self.current_dim)

        # Biological memory
        self.memory = BiologicalMemory(self.current_dim)

        # Cortex layers (start with just 1)
        self.cortex = nn.ModuleList([
            GrowableCortexLayer(self.current_dim, num_heads=max(1, self.current_dim // 16))
        ])

        # Output projection
        self.output_norm = RMSNorm(self.current_dim)
        self.output_proj = nn.Linear(self.current_dim, vocab_size, bias=False)

        # Tie weights
        self.output_proj.weight = self.embed.weight

        # State tracking
        self.current_stress = 0.0
        self.current_actions = None

        # Track device
        self._device = torch.device('cpu')

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = device
        return super().to(device)

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # === GROWTH OPERATIONS ===

    def grow_depth(self) -> bool:
        """Add a new cortex layer"""
        if len(self.cortex) >= self.config.max_layers:
            return False

        new_layer = GrowableCortexLayer(
            self.current_dim,
            num_heads=max(1, self.current_dim // 16)
        ).to(self.device)

        # Initialize as near-identity (minimal disruption)
        with torch.no_grad():
            new_layer.residual_scale.fill_(0.1)

        self.cortex.append(new_layer)
        return True

    def grow_width(self, increase: int = 16) -> bool:
        """Increase hidden dimension - expensive, requires rebuilding"""
        new_dim = self.current_dim + increase
        if new_dim > self.config.max_dim:
            return False

        old_dim = self.current_dim
        self.current_dim = new_dim

        # Rebuild all dimension-dependent components
        self.embed = nn.Embedding(self.vocab_size, new_dim).to(self.device)
        self.multi_speed = MultiSpeedProcessor(new_dim, new_dim).to(self.device)
        self.subconscious = SubconsciousMind(new_dim).to(self.device)
        self.pfc = PrefrontalCortex(new_dim).to(self.device)
        self.pfc_state = None
        self.temporal = TemporalResonance(new_dim).to(self.device)
        self.memory = BiologicalMemory(new_dim).to(self.device)

        # Rebuild cortex
        old_cortex = self.cortex
        self.cortex = nn.ModuleList()
        for _ in old_cortex:
            self.cortex.append(GrowableCortexLayer(
                new_dim,
                num_heads=max(1, new_dim // 16)
            ).to(self.device))

        # Rebuild output
        self.output_norm = RMSNorm(new_dim).to(self.device)
        self.output_proj = nn.Linear(new_dim, self.vocab_size, bias=False).to(self.device)
        self.output_proj.weight = self.embed.weight

        return True

    # === FORWARD PASS ===

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all systems active.

        Args:
            input_ids: [batch, seq_len] token indices
            targets: [batch, seq_len] target indices for loss
            return_details: Whether to return internal state details

        Returns:
            Dictionary with 'logits', optionally 'loss', and internal states
        """
        batch, seq_len = input_ids.shape

        # Update stress based on recent loss
        self.current_stress = min(1.0, self.growth_ctrl.state.best_loss)

        # === EMBEDDING ===
        x = self.embed(input_ids)  # [batch, seq, dim]

        # === MULTI-SPEED PROCESSING ===
        x = self.multi_speed(x)

        # === PREFRONTAL CORTEX (Executive Control) ===
        actions, self.pfc_state = self.pfc(
            x,
            self.current_stress,
            self.energy.biological_energy,
            self.pfc_state
        )
        self.current_actions = actions[0].detach().cpu().numpy()

        # Extract control signals
        vis_gain = actions[:, 0:1].unsqueeze(1)      # For future multimodal
        aud_gain = actions[:, 1:2].unsqueeze(1)      # For future multimodal
        dream_gain = actions[:, 2:3].mean().item()
        speak_impulse = actions[:, 3].mean().item()
        cry_suppression = actions[:, 4].mean().item()
        energy_conservation = actions[:, 5].mean().item()
        creativity_boost = actions[:, 6].mean().item()
        focus_level = actions[:, 7].mean().item()

        # Update energy system with learned conservation
        self.energy.set_conservation(energy_conservation)

        # === MEMORY RECALL ===
        memory_context = self.memory.recall(x)
        if memory_context is not None:
            memory_weight = 0.1 * (1 - focus_level)  # Less memory when focused
            x = x + memory_context.unsqueeze(0).unsqueeze(0) * memory_weight

        # === SUBCONSCIOUS PROCESSING ===
        # Mix in subconscious output based on dream_gain
        sub_out = self.subconscious(x[:, -1, :], self.current_stress)
        subconscious_contrib = sub_out['output'].unsqueeze(1)
        x = x + subconscious_contrib * dream_gain * creativity_boost

        # === TEMPORAL RESONANCE ===
        x = self.temporal(x)

        # === CORTEX PROCESSING ===
        # Causal attention mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        for layer in self.cortex:
            # Energy-gated processing
            flops = self.energy.estimate_attention_flops(
                seq_len, self.current_dim, layer.num_heads, batch
            )

            if self.energy.can_afford(flops):
                x = layer(x, mask=mask)
                self.energy.spend(flops)
            # If can't afford, skip layer (learned energy conservation)

        # === OUTPUT ===
        x = self.output_norm(x)
        logits = self.output_proj(x)

        result = {
            'logits': logits,
            'speak_impulse': speak_impulse,
            'cry_suppression': cry_suppression,
            'goal_momentum': sub_out['goal_momentum'],
        }

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            result['loss'] = loss

            # Store significant moments in memory
            significance = loss.item() + self.current_stress * 0.5
            if significance > 0.5:
                self.memory.store(x, importance=significance)

        if return_details:
            result['actions'] = self.current_actions
            result['stress'] = self.current_stress
            result['energy'] = self.energy.biological_energy
            result['subconscious_value'] = sub_out['value']

        # Age systems
        self.multi_speed.age_step()
        self.energy.step_reset()
        self.memory.decay_memories()

        return result

    # === TRAINING INTERFACE ===

    def training_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-3
    ) -> Dict[str, float]:
        """
        Single training step with growth logic.

        Returns metrics dict.
        """
        # Forward
        outputs = self.forward(input_ids, targets, return_details=True)
        loss = outputs['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        # Record loss and check for growth
        loss_val = loss.item()
        self.growth_ctrl.record_loss(loss_val)

        should_grow, reason = self.growth_ctrl.should_grow()

        grew = False
        if should_grow:
            # Prefer depth growth (cheaper) over width
            if len(self.cortex) < self.config.max_layers:
                grew = self.grow_depth()
            elif self.current_dim < self.config.max_dim:
                grew = self.grow_width()

        # Adjust LR if recommended
        if reason == "boost_lr":
            multiplier = self.growth_ctrl.get_recommended_lr_multiplier()
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * multiplier
        elif reason in ["grow", "learning"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr

        return {
            'loss': loss_val,
            'params': self.count_parameters(),
            'layers': len(self.cortex),
            'dim': self.current_dim,
            'grew': grew,
            'growth_reason': reason,
            'lr_multiplier': self.growth_ctrl.get_recommended_lr_multiplier(),
            'energy': self.energy.biological_energy,
            'stress': self.current_stress,
            'speak_impulse': outputs['speak_impulse'],
            'cry_suppression': outputs['cry_suppression'],
            'memories': self.memory.num_memories
        }


# ============================================================================
# SIMPLE TRAINING LOOP
# ============================================================================

def train_on_text(
    model: MagnumOpusCore,
    text: str,
    epochs: int = 1,
    seq_len: int = 64,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> List[Dict]:
    """Simple training loop for demonstration."""
    model = model.to(device)
    model.train()

    # Simple character-level tokenization
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}

    # Resize model vocab if needed
    if len(chars) != model.vocab_size:
        model.vocab_size = len(chars)
        model.embed = nn.Embedding(len(chars), model.current_dim).to(device)
        model.output_proj = nn.Linear(model.current_dim, len(chars), bias=False).to(device)
        model.output_proj.weight = model.embed.weight

    # Encode text
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        starts = list(range(0, len(data) - seq_len - 1, seq_len))

        for i, start in enumerate(starts):
            batch_starts = starts[i:i+batch_size]
            if len(batch_starts) < batch_size:
                continue

            inputs = torch.stack([data[s:s+seq_len] for s in batch_starts]).to(device)
            targets = torch.stack([data[s+1:s+seq_len+1] for s in batch_starts]).to(device)

            metrics = model.training_step(inputs, targets, optimizer, base_lr=lr)
            metrics['epoch'] = epoch
            metrics['step'] = len(history)
            history.append(metrics)

            if verbose and len(history) % 50 == 0:
                print(f"Step {len(history)}: loss={metrics['loss']:.4f}, "
                      f"params={metrics['params']:,}, layers={metrics['layers']}, "
                      f"stress={metrics['stress']:.2f}, speak={metrics['speak_impulse']:.2f}")

            if metrics['grew'] and verbose:
                print(f"  *** GROWTH EVENT: Now {metrics['layers']} layers ***")

    return history


# ============================================================================
# DEMO / MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MagnumOpusVitalis Core - COMPLETE Growing Neural Architecture")
    print("=" * 60)
    print()
    print("Systems active:")
    print("  - GrowthController (patient growth)")
    print("  - ComputationalEnergy (FLOPs budget)")
    print("  - SubconsciousMind (4-layer creative pipeline)")
    print("  - PrefrontalCortex (8 control outputs)")
    print("  - MultiSpeedProcessor (fast/medium/slow)")
    print("  - TemporalResonance (temporal coherence)")
    print("  - BiologicalMemory (episodic)")
    print("  - GrowableCortexLayers (modern transformers)")
    print()

    config = GrowthConfig(
        initial_dim=64,
        max_dim=256,
        max_layers=8,
        plateau_patience=100,
        lr_boost_attempts=2
    )

    model = MagnumOpusCore(vocab_size=256, config=config)
    print(f"Initial model: {model.count_parameters():,} parameters")
    print(f"Layers: {len(model.cortex)}, Dim: {model.current_dim}")
    print()

    # Demo text
    demo_text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    The only thing we have to fear is fear itself.
    Learning is not the accumulation of facts.
    Learning is the restructuring of understanding.
    """ * 20

    print(f"Training on {len(demo_text)} characters...")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = train_on_text(
        model,
        demo_text,
        epochs=3,
        seq_len=32,
        batch_size=4,
        lr=1e-3,
        device=device,
        verbose=True
    )

    print("-" * 60)
    print(f"\nFinal model: {model.count_parameters():,} parameters")
    print(f"Layers: {len(model.cortex)}, Dim: {model.current_dim}")
    print(f"Growth events: {model.growth_ctrl.state.total_growths}")
    print(f"Memories stored: {model.memory.num_memories}")

    if history:
        print(f"\nLoss: {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")
        print(f"Cry suppression: {history[0].get('cry_suppression', 0):.2f} → {history[-1].get('cry_suppression', 0):.2f}")