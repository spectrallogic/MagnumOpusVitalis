from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIG + TRAINING STATE
# ============================================================================

@dataclass
class GrowthConfig:
    """Configuration for organic growth behavior."""
    initial_dim: int = 64
    max_dim: int = 512
    max_layers: int = 12

    # Patient growth parameters ("learn first, grow last")
    plateau_patience: int = 200
    lr_boost_attempts: int = 3
    improvement_threshold: float = 0.99  # EMA must improve by >=1% to reset patience
    min_loss_for_growth: float = 0.5
    min_age_for_growth: int = 500

    # How long a LR boost should persist once triggered
    lr_boost_window: int = 200


@dataclass
class TrainingState:
    step: int = 0
    loss_ema: Optional[float] = None
    best_loss_ema: float = float("inf")
    plateau_counter: int = 0
    lr_boosts_used: int = 0
    lr_boost_remaining: int = 0
    last_reason: str = "learning"


class GrowthController:
    """
    Controls growth and LR boosts based on loss EMA plateaus.

    Key behaviors:
    - Track EMA of loss and detect plateaus (lack of improvement).
    - First try LR boosts for a window; only then grow.
    - Exposes recommended LR multiplier that persists while boost_remaining > 0.
    """

    def __init__(self, config: GrowthConfig, ema_alpha: float = 0.02):
        self.config = config
        self.ema_alpha = ema_alpha
        self.state = TrainingState()

    def record_loss(self, loss_val: float) -> None:
        s = self.state
        s.step += 1

        if s.loss_ema is None:
            s.loss_ema = float(loss_val)
        else:
            s.loss_ema = (1.0 - self.ema_alpha) * s.loss_ema + self.ema_alpha * float(loss_val)

        if s.loss_ema < s.best_loss_ema:
            s.best_loss_ema = s.loss_ema

        # decay boost window
        if s.lr_boost_remaining > 0:
            s.lr_boost_remaining -= 1

    def _is_improving(self) -> bool:
        s = self.state
        if s.loss_ema is None or s.best_loss_ema == float("inf"):
            return True
        # "Improving" means loss_ema <= best_loss_ema / threshold
        return s.loss_ema <= (s.best_loss_ema / self.config.improvement_threshold)

    def should_grow(self) -> Tuple[bool, str]:
        """
        Returns: (should_grow, reason)
        reason in {"learning","boost_lr","grow","too_early","already_good"}
        """
        s = self.state

        if s.step < self.config.min_age_for_growth:
            s.last_reason = "too_early"
            return False, "too_early"

        if s.loss_ema is not None and s.loss_ema < self.config.min_loss_for_growth:
            s.last_reason = "already_good"
            return False, "already_good"

        # If we're currently in a boost window, continue boosting (no growth)
        if s.lr_boost_remaining > 0:
            s.last_reason = "boost_lr"
            return False, "boost_lr"

        if self._is_improving():
            s.plateau_counter = 0
            s.last_reason = "learning"
            return False, "learning"

        # Not improving: count plateau
        s.plateau_counter += 1
        if s.plateau_counter < self.config.plateau_patience:
            s.last_reason = "learning"
            return False, "learning"

        # Plateau reached: try LR boost first, limited attempts
        if s.lr_boosts_used < self.config.lr_boost_attempts:
            s.lr_boosts_used += 1
            s.lr_boost_remaining = self.config.lr_boost_window
            s.plateau_counter = 0
            s.last_reason = "boost_lr"
            return False, "boost_lr"

        # LR boosts exhausted: grow
        s.plateau_counter = 0
        s.last_reason = "grow"
        return True, "grow"

    def get_recommended_lr_multiplier(self) -> float:
        """Persists while lr_boost_remaining > 0."""
        s = self.state
        if s.lr_boost_remaining <= 0:
            return 1.0
        # modest boost, not crazy; increases slightly with number of boosts used
        return 1.25 + 0.1 * max(0, s.lr_boosts_used - 1)


# ============================================================================
# ENERGY (COMPUTE + BIOLOGICAL)
# ============================================================================

class ComputationalEnergy:
    """
    A simple compute budget + biological energy proxy.

    - flops_budget: how much compute per step the agent can afford.
    - biological_energy: 0..1, decays with spending and recovers per step reset.
    - conservation: learned control from PFC (0..1), scales budget.

    The demo uses energy.can_afford(10000) as a cheap gating check; we keep that.
    """

    def __init__(self, flops_budget: float = 2e7):
        self.base_flops_budget = float(flops_budget)
        self.flops_remaining = float(flops_budget)
        self.biological_energy = 1.0
        self.conservation = 0.0  # 0=spend freely, 1=conserve hard

    def set_conservation(self, conservation: float) -> None:
        self.conservation = float(max(0.0, min(1.0, conservation)))

    def step_reset(self) -> None:
        # more conservation => smaller per-step budget usage, higher "energy"
        budget = self.base_flops_budget * (1.0 - 0.5 * self.conservation)
        self.flops_remaining = float(budget)
        # recover some biological energy each step
        self.biological_energy = float(min(1.0, self.biological_energy + 0.05))

    def can_afford(self, flops: float) -> bool:
        return float(flops) <= self.flops_remaining and self.biological_energy > 0.02

    def spend(self, flops: float) -> None:
        fl = float(max(0.0, flops))
        self.flops_remaining = max(0.0, self.flops_remaining - fl)
        # biological_energy decays softly with spending
        frac = 0.0 if self.base_flops_budget <= 0 else (fl / self.base_flops_budget)
        self.biological_energy = float(max(0.0, self.biological_energy - 0.15 * frac))

    @staticmethod
    def estimate_attention_flops(seq_len: int, dim: int, heads: int, batch: int) -> float:
        # very rough: QKV projections + attention matmul + output projection
        # O(B * S * D^2) dominates at small S; attention matmul O(B * H * S^2 * Dh)
        s = float(seq_len)
        d = float(dim)
        b = float(batch)
        h = float(max(1, heads))
        dh = d / h
        proj = 3.0 * b * s * d * d
        attn = b * h * (s * s * dh)
        out = b * s * d * d
        return proj + attn + out


# ============================================================================
# SMALL BUILDING BLOCKS
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.67):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SelfAttention(nn.Module):
    """
    Minimal causal self-attention (single tensor, no kv-cache).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, int(num_heads))
        assert dim % self.num_heads == 0, "dim must divide heads"
        self.head_dim = dim // self.num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x)  # [b,s,3d]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # [b,h,s,dh]
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # [b,h,s,s]
        if mask is not None:
            att = att + mask  # broadcastable
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v)  # [b,h,s,dh]
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out(out)


class GrowableCortexLayer(nn.Module):
    """
    Transformer block with residual scaling (for gentle depth growth).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, int(num_heads))

        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, self.num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = GatedMLP(dim)

        # scales new layers toward near-identity when grown
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask) * self.residual_scale
        x = x + self.mlp(self.norm2(x)) * self.residual_scale
        return x


# ============================================================================
# MULTI-SPEED PROCESSOR
# ============================================================================

class MultiSpeedProcessor(nn.Module):
    """
    Fast/medium/slow channels with learnable trust weights.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        fast = max(8, output_dim // 4)
        med = max(16, output_dim // 2)
        slow = output_dim

        self.fast = nn.Sequential(nn.Linear(input_dim, fast), nn.GELU(), nn.Linear(fast, fast))
        self.med = nn.Sequential(nn.Linear(input_dim, med), nn.GELU(), nn.Linear(med, med))
        self.slow = nn.Sequential(nn.Linear(input_dim, slow), nn.GELU(), nn.Linear(slow, slow))

        self.fast_up = nn.Linear(fast, output_dim)
        self.med_up = nn.Linear(med, output_dim)
        self.slow_up = nn.Identity()

        self.trust_logits = nn.Parameter(torch.tensor([1.5, 1.0, 0.5]))  # fast > med > slow early
        self.age = 0

        self.combine = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b,s,d]
        b, s, d = x.shape
        xf = self.fast_up(self.fast(x))
        xm = self.med_up(self.med(x))
        xs = self.slow_up(self.slow(x))

        trust = F.softmax(self.trust_logits, dim=0)  # [3]
        out = trust[0] * xf + trust[1] * xm + trust[2] * xs
        return self.combine(out)

    def age_step(self) -> None:
        self.age += 1
        if self.age % 1000 == 0:
            # gentle drift toward slower channel
            with torch.no_grad():
                self.trust_logits[0] -= 0.02
                self.trust_logits[2] += 0.02


# ============================================================================
# 4-LAYER SUBCONSCIOUS
# ============================================================================

class SeaOfNoise(nn.Module):
    def __init__(self, dim: int, num_basis: int = 32):
        super().__init__()
        self.dim = dim
        self.noise_basis = nn.Parameter(torch.randn(num_basis, dim) * 0.02)
        self.proj = nn.Linear(dim, num_basis)

    def forward(self, h: torch.Tensor, creativity: float = 1.0) -> torch.Tensor:
        # h: [b,dim]
        w = F.softmax(self.proj(h), dim=-1)  # [b,basis]
        noise = torch.matmul(w, self.noise_basis)  # [b,dim]
        return noise * float(max(0.0, creativity))


class DreamMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mix = nn.Linear(dim * 2, dim)

    def forward(self, h: torch.Tensor, noise: torch.Tensor, gain: float) -> torch.Tensor:
        x = torch.cat([h, noise], dim=-1)
        return h + self.mix(x) * float(max(0.0, gain))


class GoalMomentum(nn.Module):
    """
    A tiny goal accumulator. Not "true goals", but a learnable persistence vector.
    """
    def __init__(self, dim: int, momentum: float = 0.995):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("goal", torch.zeros(dim))

        self.update = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, stress: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [b,dim]
        g = self.goal.unsqueeze(0).expand(h.shape[0], -1)
        proposed = torch.tanh(self.update(h))
        # higher stress => more "plasticity" (updates faster)
        plastic = 0.001 + 0.02 * float(max(0.0, min(1.0, stress)))
        new_g = (1.0 - plastic) * g + plastic * proposed

        with torch.no_grad():
            # keep a single shared goal trace (mean over batch)
            self.goal.mul_(self.momentum).add_((1.0 - self.momentum) * new_g.mean(0))

        v = self.value(new_g)
        return new_g, v


class SubconsciousMind(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.noise = SeaOfNoise(dim)
        self.mixer = DreamMixer(dim)
        self.goal = GoalMomentum(dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, stress: float, creativity_boost: float = 1.0) -> Dict[str, torch.Tensor]:
        # h: [b,dim]
        noise = self.noise(h, creativity=float(creativity_boost))
        mixed = self.mixer(h, noise, gain=float(creativity_boost))
        g, v = self.goal(mixed, stress=stress)
        out = self.output(mixed)
        return {"output": out, "value": v, "goal_momentum": self.goal.goal.clone()}


# ============================================================================
# PREFRONTAL CORTEX (EXECUTIVE CONTROL)
# ============================================================================

class PrefrontalCortex(nn.Module):
    """
    Learns to output 8 control signals (sigmoid 0..1):
    0 vis_gain, 1 aud_gain, 2 dream_gain, 3 speak_impulse,
    4 cry_suppression, 5 energy_conservation, 6 creativity_boost, 7 focus_level
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.planner = nn.GRUCell(dim + 2, dim)
        self.policy = nn.Sequential(nn.Linear(dim, 8), nn.Sigmoid())

        # developmental-ish bias init
        with torch.no_grad():
            self.policy[0].bias.zero_()
            self.policy[0].bias[3] = 0.0   # speak_impulse moderate
            self.policy[0].bias[4] = 0.5   # cry_suppression higher initially
            self.policy[0].bias[5] = 0.0   # energy_conservation neutral
            self.policy[0].bias[7] = 0.0   # focus neutral

    def forward(
        self,
        h: torch.Tensor,
        stress: float,
        energy: float,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # reduce h to [b,dim]
        if h.dim() == 3:
            h2 = h[:, -1, :]
        elif h.dim() == 2:
            h2 = h
        else:
            h2 = h.unsqueeze(0)

        if prev_state is None:
            prev = torch.zeros_like(h2)
        else:
            prev = prev_state.detach()
            if prev.dim() == 3:
                prev = prev[:, -1, :]
            if prev.dim() == 1:
                prev = prev.unsqueeze(0)
            # if size mismatch (after width growth), pad or crop
            if prev.shape[-1] != h2.shape[-1]:
                prev = _pad_or_crop_last_dim(prev, h2.shape[-1])

        ctx = torch.tensor([[float(stress), float(energy)]], device=h2.device, dtype=h2.dtype).expand(h2.shape[0], -1)
        inp = torch.cat([h2, ctx], dim=-1)
        new_state = self.planner(inp, prev)
        actions = self.policy(new_state)
        return actions, new_state


# ============================================================================
# TEMPORAL RESONANCE
# ============================================================================

class TemporalResonance(nn.Module):
    """
    Simple causal temporal coherence: gated mixing with previous token state.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b,s,d]
        prev = torch.zeros_like(x)
        prev[:, 1:, :] = x[:, :-1, :]
        g = torch.sigmoid(self.gate(x))
        return x + g * prev


# ============================================================================
# BIOLOGICAL MEMORY (EPISODIC, RECONSTRUCTIVE)
# ============================================================================

class BiologicalMemory(nn.Module):
    """
    Episodic memory stored inside the model state (state_dict).
    Implemented as a non-trainable Parameter for alignment with your philosophy,
    but updated through explicit writes (no optimizer gradients).
    """

    def __init__(self, dim: int, capacity: int = 500):
        super().__init__()
        self.dim = int(dim)
        self.capacity = int(capacity)

        # Non-trainable but part of model "weights" / state_dict
        self.memory_bank = nn.Parameter(torch.zeros(capacity, dim), requires_grad=False)
        self.register_buffer("importance", torch.zeros(capacity))
        self.register_buffer("timestamps", torch.zeros(capacity))
        self.register_buffer("access_counts", torch.zeros(capacity))
        self.write_head = 0
        self.num_memories = 0

        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, dim)

    def store(self, state: torch.Tensor, importance: float = 1.0) -> None:
        if float(importance) < 0.3:
            return

        with torch.no_grad():
            s = state
            if s.dim() == 3:
                s = s[:, -1, :]
            if s.dim() == 2:
                s = s.mean(0)
            if s.shape[-1] != self.dim:
                return

            enc = self.encoder(s.detach())
            self.memory_bank[self.write_head].copy_(enc)
            self.importance[self.write_head] = float(importance)
            self.timestamps[self.write_head] = float(time.time())
            self.access_counts[self.write_head] = 1.0

            self.write_head = (self.write_head + 1) % self.capacity
            self.num_memories = min(self.num_memories + 1, self.capacity)

    def recall(self, query: torch.Tensor, top_k: int = 3) -> Optional[torch.Tensor]:
        if self.num_memories == 0:
            return None

        q = query
        if q.dim() == 3:
            q = q[:, -1, :]
        if q.dim() == 2:
            q = q.mean(0)

        q = q.detach().view(-1)[: self.dim]
        if q.shape[0] < self.dim:
            return None

        mem = self.memory_bank[: self.num_memories]
        sim = F.cosine_similarity(q.unsqueeze(0), mem, dim=1)

        # weight by importance and recency
        age = float(time.time()) - self.timestamps[: self.num_memories]
        time_decay = torch.exp(-0.001 * age)
        weighted = sim * self.importance[: self.num_memories] * time_decay

        k = min(int(top_k), int(self.num_memories))
        vals, idx = torch.topk(weighted, k=k, largest=True)
        with torch.no_grad():
            self.access_counts[idx] += 1.0

        retrieved = mem[idx].mean(dim=0)
        return self.decoder(retrieved)

    def decay_memories(self, factor: float = 0.999) -> None:
        with torch.no_grad():
            self.importance[: self.num_memories] *= float(factor)


# ============================================================================
# HELPERS FOR GROWTH (WEIGHT-PRESERVING)
# ============================================================================

def _pad_or_crop_last_dim(x: torch.Tensor, new_dim: int) -> torch.Tensor:
    d = x.shape[-1]
    if d == new_dim:
        return x
    if d > new_dim:
        return x[..., :new_dim]
    pad = new_dim - d
    return F.pad(x, (0, pad))


def _copy_overlapping(dst: torch.Tensor, src: torch.Tensor) -> None:
    """
    Copy src into dst for the overlapping slice of each dimension.
    """
    with torch.no_grad():
        slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
        dst[slices].copy_(src[slices])


def _widen_linear(old: nn.Linear, new_in: int, new_out: int, bias: bool) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=bias).to(old.weight.device, dtype=old.weight.dtype)
    _copy_overlapping(new.weight, old.weight)
    if bias and old.bias is not None and new.bias is not None:
        _copy_overlapping(new.bias, old.bias)
    return new


def _widen_rmsnorm(old: RMSNorm, new_dim: int) -> RMSNorm:
    new = RMSNorm(new_dim, eps=old.eps).to(old.weight.device, dtype=old.weight.dtype)
    _copy_overlapping(new.weight, old.weight)
    return new


def _widen_grucell(old: nn.GRUCell, new_input: int, new_hidden: int) -> nn.GRUCell:
    new = nn.GRUCell(new_input, new_hidden).to(old.weight_ih.device, dtype=old.weight_ih.dtype)
    _copy_overlapping(new.weight_ih, old.weight_ih)
    _copy_overlapping(new.weight_hh, old.weight_hh)
    _copy_overlapping(new.bias_ih, old.bias_ih)
    _copy_overlapping(new.bias_hh, old.bias_hh)
    return new


def _widen_embedding(old: nn.Embedding, new_dim: int) -> nn.Embedding:
    new = nn.Embedding(old.num_embeddings, new_dim, padding_idx=old.padding_idx).to(old.weight.device, dtype=old.weight.dtype)
    _copy_overlapping(new.weight, old.weight)
    return new


# ============================================================================
# THE GROWING BRAIN (CORE)
# ============================================================================

class MagnumOpusCore(nn.Module):
    """
    Core model that supports:
    - patient growth (depth first, width later)
    - energy gating
    - PFC executive control
    - multi-speed processing
    - subconscious creativity pipeline
    - episodic reconstructive memory
    """

    def __init__(self, vocab_size: int, config: Optional[GrowthConfig] = None, device: Optional[str] = None):
        super().__init__()
        self.config = config or GrowthConfig()
        self.vocab_size = int(vocab_size)

        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dim = int(self.config.initial_dim)

        # Growth controller + energy
        self.growth_ctrl = GrowthController(self.config)
        self.energy = ComputationalEnergy()

        # Embedding and tied output
        self.embed = nn.Embedding(self.vocab_size, self.current_dim).to(self.device_str)
        self.output_norm = RMSNorm(self.current_dim).to(self.device_str)
        self.output_proj = nn.Linear(self.current_dim, self.vocab_size, bias=False).to(self.device_str)
        self.output_proj.weight = self.embed.weight  # weight tying

        # Core systems
        self.multi_speed = MultiSpeedProcessor(self.current_dim, self.current_dim).to(self.device_str)
        self.subconscious = SubconsciousMind(self.current_dim).to(self.device_str)
        self.pfc = PrefrontalCortex(self.current_dim).to(self.device_str)
        self.pfc_state: Optional[torch.Tensor] = None
        self.temporal = TemporalResonance(self.current_dim).to(self.device_str)
        self.memory = BiologicalMemory(self.current_dim).to(self.device_str)

        # Cortex layers
        self.cortex = nn.ModuleList([
            GrowableCortexLayer(self.current_dim, num_heads=max(1, self.current_dim // 16)).to(self.device_str)
        ])

        # Runtime signals for UI/debug
        self.current_actions = None
        self.current_stress = 0.0

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def grow_depth(self) -> bool:
        if len(self.cortex) >= self.config.max_layers:
            return False
        new_layer = GrowableCortexLayer(self.current_dim, num_heads=max(1, self.current_dim // 16)).to(self.device_str)
        with torch.no_grad():
            new_layer.residual_scale.fill_(0.1)  # near-identity
        self.cortex.append(new_layer)
        return True

    def grow_width(self, increase: int = 16) -> bool:
        new_dim = int(self.current_dim + increase)
        if new_dim > self.config.max_dim:
            return False

        old_dim = self.current_dim
        self.current_dim = new_dim

        # widen embedding + output (keep weight tying)
        old_embed = self.embed
        self.embed = _widen_embedding(old_embed, new_dim).to(self.device_str)

        self.output_norm = _widen_rmsnorm(self.output_norm, new_dim).to(self.device_str)
        self.output_proj = nn.Linear(new_dim, self.vocab_size, bias=False).to(self.device_str)
        self.output_proj.weight = self.embed.weight

        # widen multi-speed
        ms_old = self.multi_speed
        self.multi_speed = MultiSpeedProcessor(new_dim, new_dim).to(self.device_str)
        self._partial_copy_module(self.multi_speed, ms_old)

        # widen subconscious + pfc + temporal + memory
        sub_old = self.subconscious
        self.subconscious = SubconsciousMind(new_dim).to(self.device_str)
        self._partial_copy_module(self.subconscious, sub_old)

        pfc_old = self.pfc
        self.pfc = PrefrontalCortex(new_dim).to(self.device_str)
        # widen GRUCell explicitly
        self.pfc.planner = _widen_grucell(pfc_old.planner, new_input=new_dim + 2, new_hidden=new_dim).to(self.device_str)
        self._partial_copy_module(self.pfc.policy, pfc_old.policy)

        temp_old = self.temporal
        self.temporal = TemporalResonance(new_dim).to(self.device_str)
        self._partial_copy_module(self.temporal, temp_old)

        mem_old = self.memory
        self.memory = BiologicalMemory(new_dim, capacity=mem_old.capacity).to(self.device_str)
        # Copy episodic bank into new dim (pad)
        with torch.no_grad():
            _copy_overlapping(self.memory.memory_bank, mem_old.memory_bank)
            _copy_overlapping(self.memory.importance, mem_old.importance)
            _copy_overlapping(self.memory.timestamps, mem_old.timestamps)
            _copy_overlapping(self.memory.access_counts, mem_old.access_counts)
            self.memory.write_head = mem_old.write_head
            self.memory.num_memories = mem_old.num_memories
        self._partial_copy_module(self.memory.encoder, mem_old.encoder)
        self._partial_copy_module(self.memory.decoder, mem_old.decoder)

        # widen cortex layers (partial copy each)
        old_cortex = list(self.cortex)
        self.cortex = nn.ModuleList()
        for layer_old in old_cortex:
            layer_new = GrowableCortexLayer(new_dim, num_heads=max(1, new_dim // 16)).to(self.device_str)
            self._partial_copy_module(layer_new, layer_old)
            # pad residual scale
            with torch.no_grad():
                layer_new.residual_scale.copy_(layer_old.residual_scale)
            self.cortex.append(layer_new)

        # widen pfc_state if it exists
        if self.pfc_state is not None:
            self.pfc_state = _pad_or_crop_last_dim(self.pfc_state, new_dim)

        return True

    @staticmethod
    def _partial_copy_module(dst: nn.Module, src: nn.Module) -> None:
        """
        Copy overlapping weights/biases for matching keys where shapes differ.
        This keeps old knowledge while allowing new capacity.
        """
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        with torch.no_grad():
            for k, v in dst_sd.items():
                if k not in src_sd:
                    continue
                sv = src_sd[k]
                if v.shape == sv.shape:
                    v.copy_(sv)
                else:
                    # partial copy for overlapping shape
                    _copy_overlapping(v, sv)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all systems active.
        Returns dict with at least: logits, speak_impulse, cry_suppression, goal_momentum.
        If targets provided, includes loss.
        """
        batch, seq_len = input_ids.shape

        # Reset per-step energy budget (no side effects like aging/memory decay here)
        self.energy.step_reset()

        # === EMBEDDING ===
        x = self.embed(input_ids)  # [b,s,d]

        # === MULTI-SPEED ===
        x = self.multi_speed(x)

        # === STRESS (inference-friendly): derived from growth controller EMA, if available
        s = self.growth_ctrl.state
        if s.loss_ema is None or s.best_loss_ema == float("inf"):
            self.current_stress = 0.2
        else:
            delta = max(0.0, float(s.loss_ema - s.best_loss_ema))
            self.current_stress = float(1.0 - math.exp(-2.0 * delta))
        self.current_stress = float(max(0.0, min(1.0, self.current_stress)))

        # === PFC CONTROL ===
        actions, self.pfc_state = self.pfc(
            x, self.current_stress, self.energy.biological_energy, self.pfc_state
        )
        self.current_actions = actions[0].detach().cpu().numpy()

        dream_gain = float(actions[:, 2:3].mean().item())
        speak_impulse = float(actions[:, 3].mean().item())
        cry_suppression = float(actions[:, 4].mean().item())
        energy_conservation = float(actions[:, 5].mean().item())
        creativity_boost = float(actions[:, 6].mean().item())
        focus_level = float(actions[:, 7].mean().item())

        self.energy.set_conservation(energy_conservation)

        # === MEMORY RECALL ===
        mem = self.memory.recall(x)
        if mem is not None:
            mem_w = 0.1 * (1.0 - focus_level)
            x = x + mem.view(1, 1, -1) * mem_w

        # === SUBCONSCIOUS ===
        sub = self.subconscious(x[:, -1, :], stress=self.current_stress, creativity_boost=creativity_boost)
        x = x + sub["output"].unsqueeze(1) * dream_gain * creativity_boost

        # === TEMPORAL ===
        x = self.temporal(x)

        # === CORTEX (energy-gated) ===
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype) * float("-inf"),
            diagonal=1
        )
        for layer in self.cortex:
            flops = self.energy.estimate_attention_flops(seq_len, self.current_dim, layer.num_heads, batch)
            if self.energy.can_afford(flops):
                x = layer(x, mask=mask)
                self.energy.spend(flops)

        # === OUTPUT ===
        x = self.output_norm(x)
        logits = self.output_proj(x)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "speak_impulse": torch.tensor(speak_impulse, device=logits.device),
            "cry_suppression": torch.tensor(cry_suppression, device=logits.device),
            "goal_momentum": sub["goal_momentum"].detach(),
        }

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
            out["loss"] = loss

            # optional: store significant episodes (only when training-like usage)
            # significance uses loss + stress (bounded)
            significance = float(loss.detach().item()) + 0.5 * self.current_stress
            if significance > 0.7:
                self.memory.store(x.detach(), importance=min(2.0, significance))

        if return_details:
            out["actions"] = actions.detach()
            out["stress"] = torch.tensor(self.current_stress, device=logits.device)
            out["energy"] = torch.tensor(self.energy.biological_energy, device=logits.device)
            out["subconscious_value"] = sub["value"].detach()

        return out

    # ------------------------------------------------------------------
    # Training step (handles growth + aging side-effects)
    # ------------------------------------------------------------------

    def training_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-3
    ) -> Dict[str, float]:
        self.train()
        outputs = self.forward(input_ids, targets=targets, return_details=True)
        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss.detach().item())
        self.growth_ctrl.record_loss(loss_val)

        should_grow, reason = self.growth_ctrl.should_grow()

        grew = False
        if should_grow:
            # depth first, then width
            if len(self.cortex) < self.config.max_layers:
                grew = self.grow_depth()
            elif self.current_dim < self.config.max_dim:
                grew = self.grow_width()

            if grew:
                # After growth, optimizer MUST see new params.
                self._refresh_optimizer_params(optimizer)

        # LR multiplier persists during boost window
        mult = self.growth_ctrl.get_recommended_lr_multiplier()
        for group in optimizer.param_groups:
            group["lr"] = float(base_lr) * float(mult)

        # aging side-effects happen per training step (not inference)
        self.multi_speed.age_step()
        self.memory.decay_memories()

        return {
            "loss": loss_val,
            "params": float(self.count_parameters()),
            "layers": float(len(self.cortex)),
            "dim": float(self.current_dim),
            "grew": float(1.0 if grew else 0.0),
            "growth_reason": reason,
            "lr_multiplier": float(mult),
            "energy": float(self.energy.biological_energy),
            "stress": float(self.current_stress),
            "speak_impulse": float(outputs["speak_impulse"].detach().item()),
            "cry_suppression": float(outputs["cry_suppression"].detach().item()),
            "memories": float(self.memory.num_memories),
        }


    def _refresh_optimizer_params(self, optimizer: torch.optim.Optimizer) -> None:  # type: ignore[override]
        params = [p for p in self.parameters() if p.requires_grad]
        for group in optimizer.param_groups:
            group["params"] = params

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Minimal self-test (optional)
# ============================================================================

def _self_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GrowthConfig(initial_dim=32, max_dim=64, max_layers=4, plateau_patience=5, min_age_for_growth=1)
    m = MagnumOpusCore(vocab_size=128, config=cfg, device=device).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    x = torch.randint(0, 128, (2, 8), device=device)
    y = torch.randint(0, 128, (2, 8), device=device)

    # forward
    o = m(x, y)
    assert "logits" in o and "loss" in o
    # train
    for _ in range(3):
        m.training_step(x, y, opt, base_lr=1e-3)

    # width growth safety
    m.grow_width(16)
    o2 = m(x, y)
    assert o2["logits"].shape[-1] == 128

if __name__ == "__main__":
    _self_test()
