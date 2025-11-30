"""
MagnumOpusVitalis: The Living Intelligence
================================================================
ARCHITECT: Alan Hourmand
VERSION: 2.4 (Genesis Unified + Synaptic Fatigue + Selective Loss + Persistent Identity)

PHILOSOPHY:
"A seed that grows, not a machine that thinks."

INTEGRATIONS:
- ElasticLowRankLayer (low-rank growth with compression)
- Exploratory Routing (temperature + epsilon-greedy)
- TemporalConsciousness (specious present)
- Biological Hormone System (Oxytocin/Stress regulation)
- Emotional Audio Synthesis (FM Synthesis Droid/Infant)
- Synaptic Fatigue (Natural repetition inhibition)
- Selective Cluster Knowledge (Islands of Confidence)
- Persistent Scaffold Identity (Short-term context window)

CORE PRINCIPLES:
1. TABULA RASA - Starts knowing nothing, learns everything
2. MULTI-SCALE ABSTRACTION - Fast pattern recognition, slow understanding
3. ORGANIC GROWTH - Expands only when confused (low-rank growth)
4. UNIFIED MEMORY - Memory IS the model, not separate
5. ENERGY ECONOMICS - Speaking costs energy, silence recovers
6. TEMPORAL CONSCIOUSNESS - Aware of past/present/future simultaneously
7. SELECTIVE MASTERY - Failure in one domain does not break confidence in another

NO HARDCODED KNOWLEDGE. NO CHEATING.
"""

import sys
import time
import queue
import random
import math
import re
import csv
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ============================================================================
# 🛡️ CONFIGURATION FLAGS
# ============================================================================

ENABLE_RAW_AUDIO_OUTPUT = True   # ENABLED: Set to True for R2-D2 sounds
ENABLE_RAW_MICROPHONE = False    # Set True if you have a mic and sounddevice
ENABLE_TTS = True                # Text-to-speech via pyttsx3
ENABLE_STT = True                # Speech-to-text via speech_recognition

# Windows COM fix for TTS
TRY_COM_FIX = False
try:
    import pythoncom
    TRY_COM_FIX = True
except ImportError:
    pass

# Optional audio
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    if ENABLE_RAW_AUDIO_OUTPUT:
        print("[SYSTEM] sounddevice not available - audio disabled despite config")

# ============================================================================
# 🥚 SCAFFOLD SYSTEMS (Training Wheels)
# ============================================================================

SCAFFOLD_LLM_AVAILABLE = False
SCAFFOLD_STT_AVAILABLE = False
SCAFFOLD_TTS_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    SCAFFOLD_LLM_AVAILABLE = True
    print("[SCAFFOLD] LLM available (transformers)")
except ImportError:
    print("[SCAFFOLD] LLM not available - install: pip install transformers")

if ENABLE_STT:
    try:
        import speech_recognition as sr
        SCAFFOLD_STT_AVAILABLE = True
        print("[SCAFFOLD] STT available (speech_recognition)")
    except ImportError:
        print("[SCAFFOLD] STT not available - install: pip install SpeechRecognition")

if ENABLE_TTS:
    try:
        import pyttsx3
        SCAFFOLD_TTS_AVAILABLE = True
        print("[SCAFFOLD] TTS available (pyttsx3)")
    except ImportError:
        print("[SCAFFOLD] TTS not available - install: pip install pyttsx3")


class ScaffoldLLM:
    """
    The YOLK - A tiny pre-trained LLM that provides initial language capability.
    Now includes CONTEXT HISTORY and PERSONA.
    """

    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'
        self.history = deque(maxlen=6) # Keep last 6 exchanges to prevent overflow
        self.system_prompt = "You are Magnum Opus Vitalis, a biological artificial intelligence that is growing and learning. Keep responses short, curious, and child-like."

        if SCAFFOLD_LLM_AVAILABLE:
            try:
                print("[LLM YOLK] Loading GPT-2...")
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.tokenizer.pad_token = self.tokenizer.eos_token

                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.to(self.device)

                self.model.eval()
                self.available = True
                print(f"[LLM YOLK] Ready on {self.device}")
            except Exception as e:
                print(f"[LLM YOLK] Failed to load: {e}")

    def chat(self, user_input: str, max_tokens: int = 40) -> str:
        """Generate a response to user input using history."""
        if not self.available:
            return "..."
        try:
            # Build context string
            context_str = self.system_prompt + "\n"
            for turn in self.history:
                context_str += turn + "\n"

            prompt = f"{context_str}User: {user_input}\nAI:"

            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            # Truncate if too long for GPT-2 (1024 tokens max)
            if inputs.shape[1] > 900:
                inputs = inputs[:, -900:]

            mask = torch.ones(inputs.shape, device=self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, attention_mask=mask, max_new_tokens=max_tokens,
                    do_sample=True, temperature=0.8, top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the new response
            response_part = full_text.split("AI:")[-1].strip()

            # Clean up response (stop at next User: prompt if generated)
            if "User:" in response_part:
                response_part = response_part.split("User:")[0].strip()

            # Update history
            self.history.append(f"User: {user_input}")
            self.history.append(f"AI: {response_part}")

            return response_part
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "..."


class ScaffoldSTT:
    """Speech-to-Text scaffold - lets the AI hear spoken words."""

    def __init__(self):
        self.available = SCAFFOLD_STT_AVAILABLE
        self.recognizer = None
        if self.available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
            except:
                self.available = False

    def listen_once(self, timeout: float = 0.5) -> Optional[str]:
        if not self.available:
            return None
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=4)
            return self.recognizer.recognize_google(audio).lower()
        except:
            return None


from PySide6.QtCore import QThread

class TTSWorker(QThread):
    """
    Dedicated thread for Text-to-Speech to prevent blocking the main brain loop.
    Fixes the 'speak once then die' bug.
    """
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.available = SCAFFOLD_TTS_AVAILABLE

    def speak(self, text):
        if self.available:
            self.queue.put(text)

    def run(self):
        if not self.available:
            return

        try:
            # Initialize engine inside the thread
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)

            while self.running:
                try:
                    text = self.queue.get(timeout=1.0)
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for s in sentences:
                        if s.strip():
                            engine.say(s)
                            engine.runAndWait()
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"[TTS ERROR] {e}")
        except Exception as e:
            print(f"[TTS INIT ERROR] {e}")


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class EpisodeMemory:
    """A single memory trace."""
    embedding: torch.Tensor
    timestamp: float
    importance: float = 1.0
    access_count: int = 0

    def decay(self, rate: float = 0.995):
        self.importance *= rate


@dataclass
class ClusterKnowledge:
    """
    Tracks mastery and confidence for a specific topic cluster.
    """
    cluster_id: int
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=20))
    total_exposure: int = 0
    mastery_level: float = 0.0 # 0.0 (Novice) -> 1.0 (Expert)

    def record_loss(self, loss: float):
        self.recent_losses.append(loss)
        self.total_exposure += 1

        # Calculate local mastery based on recent performance
        avg_loss = sum(self.recent_losses) / len(self.recent_losses)

        # Mastery increases if loss is consistently low
        current_performance = max(0.0, 1.0 - (avg_loss / 3.0))
        self.mastery_level = 0.95 * self.mastery_level + 0.05 * current_performance

    @property
    def local_stress(self) -> float:
        """How stressful is this specific topic?"""
        if not self.recent_losses:
            return 0.1
        avg_loss = sum(self.recent_losses) / len(self.recent_losses)
        # Normalize: Loss of 0.5 is calm, Loss of 5.0 is panic
        return min(1.0, avg_loss / 4.0)


class TabularRasaVocabulary:
    """Dynamic vocabulary that learns new words."""

    def __init__(self, max_vocab: int = 10000):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.max_vocab = max_vocab
        self.recent_words: deque = deque(maxlen=100)

    def learn_text(self, text: str) -> List[int]:
        if not text:
            return []
        words = re.findall(r'\b\w+\b', text.lower())
        indices = []
        for w in words:
            if w not in self.word2idx:
                if len(self.idx2word) < self.max_vocab:
                    idx = len(self.idx2word)
                    self.word2idx[w] = idx
                    self.idx2word.append(w)
            if w in self.word2idx:
                indices.append(self.word2idx[w])
                self.recent_words.append(self.word2idx[w])
        return indices

    def __len__(self):
        return len(self.idx2word)


class EnergySystem:
    """
    Energy economics - Time-Independent (Wall Clock) Implementation.
    """

    def __init__(self):
        self.energy = 0.8  # Start higher (80%)
        self.conservation_gain = 0.5
        self.last_time = time.time()
        self.metabolic_rate = 0.02  # Cost of existing per second
        self.regen_rate_base = 0.05 # Recovery per second

    def can_speak(self) -> bool:
        return self.energy > 0.15

    def spend_speaking(self, num_words: int = 1):
        # Cap max words charged per event to prevent instant death
        capped_words = min(num_words, 5)
        base_cost = 0.04
        word_cost = 0.01 * capped_words
        total_cost = (base_cost + word_cost) * (1.2 - self.conservation_gain * 0.4)
        self.energy = max(0.0, self.energy - total_cost)

    def spend_thinking(self):
        pass

    def regenerate(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 1.0: dt = 0.0

        drain = self.metabolic_rate * dt
        recovery = self.regen_rate_base * (0.8 + self.conservation_gain * 0.5) * dt

        net_change = recovery - drain
        self.energy = min(1.0, max(0.0, self.energy + net_change))


class SessionLogger:
    """Log session data to CSV."""

    def __init__(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = f"brain_log_{timestamp}.csv"
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'loss', 'layers', 'vocab_size',
            'energy', 'stress', 'fps', 'event'
        ])
        self.file.flush()

    def log(self, loss, layers, vocab_size, energy, stress, fps, event=""):
        self.writer.writerow([
            time.strftime("%H:%M:%S"), f"{loss:.4f}", layers, vocab_size,
            f"{energy:.2f}", f"{stress:.2f}", f"{fps:.1f}", event
        ])
        self.file.flush()

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()


# ============================================================================
# 🧬 SECTION 1: NEURAL COMPONENTS
# ============================================================================

class ElasticLowRankLayer(nn.Module):
    """
    Low-rank factorization layer that grows organically.
    """

    def __init__(self, n_in: int, n_out: int, rank: int = 4, num_clusters: int = 8):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rank = rank
        self.num_clusters = num_clusters

        # Core low-rank matrices (shared)
        self.U = nn.Parameter(0.02 * torch.randn(n_out, rank))
        self.V = nn.Parameter(0.02 * torch.randn(n_in, rank))

        # Per-cluster residual growth
        self.U_res = nn.ParameterList()
        self.V_res = nn.ParameterList()
        for _ in range(num_clusters):
            self.U_res.append(nn.Parameter(torch.zeros(n_out, 0), requires_grad=False))
            self.V_res.append(nn.Parameter(torch.zeros(n_in, 0), requires_grad=False))

        # Protected basis for consolidation (prevents forgetting)
        self.register_buffer("protected_basis_V", torch.empty(n_in, 0))

    def forward(self, x: torch.Tensor, active_cluster: int = 0) -> torch.Tensor:
        # Core transformation: U @ V^T @ x
        core = self.U @ (self.V.t() @ x.t())

        # Add cluster-specific residual if it exists
        if self.U_res[active_cluster].numel() > 0:
            residual = self.U_res[active_cluster] @ (self.V_res[active_cluster].t() @ x.t())
            core = core + residual

        return F.gelu(core.t())

    def grow_cluster(self, cluster_idx: int, grow_rank: int = 2):
        """Add capacity to a specific cluster."""
        device = self.U.device
        new_U = 0.01 * torch.randn(self.n_out, grow_rank, device=device)
        new_V = 0.01 * torch.randn(self.n_in, grow_rank, device=device)

        if self.U_res[cluster_idx].numel() > 0:
            self.U_res[cluster_idx] = nn.Parameter(
                torch.cat([self.U_res[cluster_idx].data, new_U], dim=1),
                requires_grad=True
            )
            self.V_res[cluster_idx] = nn.Parameter(
                torch.cat([self.V_res[cluster_idx].data, new_V], dim=1),
                requires_grad=True
            )
        else:
            self.U_res[cluster_idx] = nn.Parameter(new_U, requires_grad=True)
            self.V_res[cluster_idx] = nn.Parameter(new_V, requires_grad=True)

    def get_cluster_rank(self, cluster_idx: int) -> int:
        """Get current rank for a cluster."""
        base = self.rank
        extra = self.U_res[cluster_idx].shape[1] if self.U_res[cluster_idx].numel() > 0 else 0
        return base + extra


class NearestCentroidRouter(nn.Module):
    """
    Exploratory routing.
    """

    def __init__(self, dim: int, num_clusters: int = 8, momentum: float = 0.02):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.momentum = momentum

        # Initialize centroids
        self.centroids = nn.Parameter(
            F.normalize(torch.randn(num_clusters, dim), dim=1),
            requires_grad=False
        )

    def update_centroid(self, k: int, z: torch.Tensor):
        """Update centroid with new sample."""
        with torch.no_grad():
            z_norm = F.normalize(z.detach(), dim=0)
            self.centroids[k] = F.normalize(
                (1 - self.momentum) * self.centroids[k] + self.momentum * z_norm,
                dim=0
            )

    def forward(self, z: torch.Tensor, tau: float = 1.0, eps: float = 0.0) -> int:
        """Route to cluster with exploration."""
        z_norm = F.normalize(z, dim=-1)
        if z_norm.dim() == 2:
            z_norm = z_norm.squeeze(0)

        sims = F.cosine_similarity(z_norm.unsqueeze(0), self.centroids, dim=1)

        # Epsilon-greedy exploration
        if self.training and random.random() < eps:
            return random.randint(0, self.num_clusters - 1)

        # Temperature-based soft routing
        if self.training and tau > 0:
            probs = F.softmax(sims / tau, dim=0)
            return int(torch.multinomial(probs, 1).item())

        return int(torch.argmax(sims).item())


class TemporalConsciousness(nn.Module):
    """
    Maintains awareness across past/present/future simultaneously.
    """

    def __init__(self, model_dim: int, window_size: int = 7):
        super().__init__()
        self.model_dim = model_dim
        self.window_size = window_size
        self.past_size = window_size // 2
        self.future_size = window_size // 2

        # Temporal attention across the conscious window
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=4, batch_first=True
        )

        # Presence weights: past fades, present peaks, future anticipates
        self.presence_weights = nn.Parameter(torch.zeros(window_size))
        with torch.no_grad():
            center = window_size // 2
            for i in range(window_size):
                distance = abs(i - center)
                self.presence_weights[i] = 1.0 / (1.0 + distance * 0.3)

        # Integration layer
        self.integrate = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

        # Future anticipation
        self.anticipate = nn.GRUCell(model_dim, model_dim)

        # Conscious window buffer
        self.conscious_window = deque(maxlen=window_size)

    def initialize_window(self, initial_state: torch.Tensor):
        self.conscious_window.clear()
        for _ in range(self.window_size):
            self.conscious_window.append(initial_state.clone())

    def forward(self, z_current: torch.Tensor) -> Tuple[torch.Tensor, Dict, List]:
        device = z_current.device

        if len(self.conscious_window) == 0:
            self.initialize_window(z_current)

        # Update window
        self.conscious_window.append(z_current)

        # Generate future anticipations
        window_list = list(self.conscious_window)
        current_state = window_list[-1]

        future_states = []
        h_future = current_state
        for _ in range(self.future_size):
            h_future = self.anticipate(z_current, h_future)
            future_states.append(h_future)

        # Construct full temporal window
        past_states = window_list[:self.past_size]
        present_state = window_list[self.past_size] if len(window_list) > self.past_size else current_state

        all_states = past_states + [present_state] + future_states
        if len(all_states) < self.window_size:
            # Pad with current state
            all_states = all_states + [current_state] * (self.window_size - len(all_states))

        temporal_window = torch.stack(all_states[:self.window_size], dim=0)

        # Temporal attention
        query = present_state.unsqueeze(0).unsqueeze(0)
        kv = temporal_window.unsqueeze(0)

        attended, attention_weights = self.temporal_attention(query, kv, kv, need_weights=True)
        attended = attended.squeeze(0).squeeze(0)

        # Modulate by presence intensity
        presence_modulated = temporal_window * self.presence_weights.unsqueeze(1).to(device)
        blended = presence_modulated.mean(dim=0)

        # Integrate
        h_conscious = self.integrate(attended + blended)

        awareness_map = {
            'attention_weights': attention_weights.squeeze(0).detach(),
            'presence_weights': self.presence_weights.detach()
        }

        return h_conscious, awareness_map, future_states


class TemporalResonance(nn.Module):
    """
    Simple temporal context via exponential moving average.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('resonance_state', torch.zeros(1, dim))
        self.clock_phase = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure resonance state matches batch size
        if self.resonance_state.shape[0] != x.shape[0]:
            self.resonance_state = torch.zeros_like(x)

        # EMA update
        self.resonance_state = (x * 0.1) + (self.resonance_state * 0.9)
        self.clock_phase += 0.1

        # Rhythmic modulation
        rhythm = 1 + 0.05 * math.sin(self.clock_phase)
        return self.resonance_state * rhythm


class MultiSpeedProcessor(nn.Module):
    """
    Multi-scale abstraction: Fast pattern recognition, slow understanding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.speeds = [16, 32, 64, 128]

        # Projectors for each speed
        self.projectors = nn.ModuleList([
            nn.Linear(dim, speed) for speed in self.speeds
        ])
        self.combiners = nn.ModuleList([
            nn.Linear(speed, dim) for speed in self.speeds
        ])

        # Trust weights (learnable - shift toward slower channels over time)
        self.trust = nn.Parameter(torch.tensor([1.0, 0.5, 0.2, 0.1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)

        outputs = []
        for i, (proj, comb) in enumerate(zip(self.projectors, self.combiners)):
            speed_repr = F.gelu(proj(x))
            restored = comb(speed_repr)
            outputs.append(restored * self.trust[i])

        combined = sum(outputs) / (self.trust.sum() + 1e-6)
        return combined


# ============================================================================
# 🧠 SECTION 2: SUBCONSCIOUS SYSTEM
# ============================================================================

class SeaOfNoise(nn.Module):
    """Layer 0: Creative randomness from learned vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.noise_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.3
        return self.noise_proj(x + noise)


class PeakDetector(nn.Module):
    """Layer 1: Filter for relevant activations."""

    def __init__(self, dim: int):
        super().__init__()
        self.threshold = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.threshold(x))
        return x * gates


class FutureGenerator(nn.Module):
    """Layer 2: Simulate possible outcomes."""

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.register_buffer('momentum', torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.momentum.shape != x.shape:
            self.momentum = torch.zeros_like(x)

        future = self.gru(x, self.momentum)
        self.momentum = 0.95 * self.momentum + 0.05 * future.detach()
        return future, x


class ScenarioEvaluator(nn.Module):
    """Layer 3: Assess path quality."""

    def __init__(self, dim: int):
        super().__init__()
        self.judge = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = torch.sigmoid(self.judge(x))
        return x * score


class SubconsciousMind(nn.Module):
    """
    Four-layer subconscious pipeline.
    Creates emergent goals via goal momentum.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sea = SeaOfNoise(dim)
        self.peak = PeakDetector(dim)
        self.future = FutureGenerator(dim)
        self.evaluator = ScenarioEvaluator(dim)
        self.output = nn.Linear(dim, dim)

        # Goal momentum (emergent desires)
        self.register_buffer('goal_momentum', torch.zeros(1, dim))

    def forward(self, x: torch.Tensor, stress: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Layer 0: Creative noise (more when stressed)
        noise_scale = 1.0 + stress * 0.5
        s0 = self.sea(x) * noise_scale

        # Layer 1: Peak detection
        s1 = self.peak(s0)

        # Layer 2: Future simulation
        s2, filtered = self.future(s1)

        # Layer 3: Evaluation
        s3 = self.evaluator(s2)

        # Update goal momentum
        if s3.shape == self.goal_momentum.shape:
            self.goal_momentum = 0.95 * self.goal_momentum + 0.05 * s3.detach()

        output = self.output(s3)
        return output, s3, self.goal_momentum


# ============================================================================
# 💾 SECTION 3: BIOLOGICAL MEMORY
# ============================================================================

class BiologicalMemory(nn.Module):
    """
    Memory IS the model's internal state, not a database.
    Episodic, reconstructive, importance-weighted.
    """

    def __init__(self, dim: int, capacity: int = 2000):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.memories: List[EpisodeMemory] = []

        # Memory encoder/decoder
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, dim)

    def store(self, state: torch.Tensor, importance: float = 0.5):
        """Store a memory trace."""
        if len(self.memories) >= self.capacity:
            # Remove least important
            self.memories.sort(key=lambda m: m.importance)
            self.memories.pop(0)

        encoded = self.encoder(state.detach())
        self.memories.append(EpisodeMemory(
            embedding=encoded,
            timestamp=time.time(),
            importance=importance
        ))

    def recall(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """Reconstructive recall via pattern matching."""
        if not self.memories:
            return None

        query_enc = self.encoder(query.detach())
        if query_enc.dim() == 1:
            query_enc = query_enc.unsqueeze(0)

        best_sim = -1
        best_memory = None

        for mem in self.memories:
            mem_emb = mem.embedding
            if mem_emb.dim() == 1:
                mem_emb = mem_emb.unsqueeze(0)

            sim = F.cosine_similarity(query_enc, mem_emb, dim=-1).mean().item()
            weighted_sim = sim * mem.importance

            if weighted_sim > best_sim:
                best_sim = weighted_sim
                best_memory = mem

        if best_memory is not None:
            best_memory.access_count += 1
            return self.decoder(best_memory.embedding)

        return None

    def decay_all(self, rate: float = 0.995):
        """Decay all memory importances."""
        for mem in self.memories:
            mem.decay(rate)


# ============================================================================
# 🎨 SECTION 4: IMAGINARIUM
# ============================================================================

class Imaginarium(nn.Module):
    """Dream image generation - mixes real, memory, and subconscious."""

    def __init__(self, dim: int):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, real: torch.Tensor, memory: Optional[torch.Tensor],
                subconscious: torch.Tensor, dream_gain: float) -> Tuple[torch.Tensor, torch.Tensor]:

        if memory is None:
            memory = torch.zeros_like(real)
        if memory.shape != real.shape:
            memory = torch.zeros_like(real)
        if subconscious.shape != real.shape:
            subconscious = torch.zeros_like(real)

        combined = torch.cat([real, memory, subconscious], dim=-1)
        dream_state = torch.tanh(self.mixer(combined)) * (0.5 + dream_gain)
        dream_image = self.decoder(dream_state)

        return dream_image, dream_state


# ============================================================================
# 🧠 SECTION 5: PREFRONTAL CORTEX (Executive Control)
# ============================================================================

class PrefrontalCortex(nn.Module):
    """
    Learns to regulate the entire system.
    8 learned control outputs.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.planner = nn.GRUCell(dim + 2, dim)  # +2 for stress and energy
        self.policy = nn.Sequential(
            nn.Linear(dim, 8),
            nn.Sigmoid()
        )

        # Initialize biases for scaffold dependence
        # Higher bias = sigmoid outputs closer to 1.0
        with torch.no_grad():
            self.policy[0].bias[6] = 3.5  # llm_reliance starts VERY high (~0.97)
            self.policy[0].bias[7] = 2.5  # tts_gate starts high (~0.92)
            self.policy[0].bias[3] = 1.0  # speak_impulse moderate (~0.73)
            self.policy[0].bias[5] = 0.0  # energy_conservation neutral (~0.5)

    def forward(self, h: torch.Tensor, stress: float, energy: float,
                prev_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if prev_state is None:
            prev_state = torch.zeros_like(h)
        else:
            prev_state = prev_state.detach()

        context = torch.tensor([[stress, energy]], device=h.device)
        inp = torch.cat([h, context], dim=1)

        new_state = self.planner(inp, prev_state)
        actions = self.policy(new_state)

        # actions: [vis_gain, aud_gain, dream_gain, speak_impulse, cry_suppression,
        #           energy_conservation, llm_reliance, tts_gate]
        return actions, new_state


# ============================================================================
# 🧬 SECTION 6: THE OMNIBRAIN (Main Model)
# ============================================================================

class OmniBrain(nn.Module):
    """
    The complete living brain.
    Starts microscopic, grows organically.
    All knowledge is learned, nothing hardcoded.
    """

    def __init__(self, vocab_size: int = 10000, num_clusters: int = 8):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.base_dim = 128
        self.num_clusters = num_clusters

        # === SENSORY INPUTS ===
        self.eye = nn.Sequential(nn.Linear(32 * 32, 64), nn.Tanh())
        self.ear_external = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, self.base_dim)
        self.text_embed = nn.Embedding(vocab_size, self.base_dim)

        # === CORE SYSTEMS ===
        self.pfc = PrefrontalCortex(self.base_dim)
        self.subconscious = SubconsciousMind(self.base_dim)
        self.multi_speed = MultiSpeedProcessor(self.base_dim)
        self.memory = BiologicalMemory(self.base_dim)
        self.imaginarium = Imaginarium(self.base_dim)
        self.temporal_resonance = TemporalResonance(self.base_dim)
        self.temporal_consciousness = TemporalConsciousness(self.base_dim, window_size=7)

        # === ROUTING (Exploratory) ===
        self.router = NearestCentroidRouter(self.base_dim, num_clusters)

        # === ELASTIC LOW-RANK CORTEX ===
        self.cortex = ElasticLowRankLayer(
            self.base_dim, self.base_dim,
            rank=4, num_clusters=num_clusters
        )

        # === ADDITIONAL CORTEX LAYERS (can grow) ===
        self.cortex_layers = nn.ModuleList([nn.Linear(self.base_dim, self.base_dim)])

        # === OUTPUTS ===
        self.vis_out = nn.Linear(self.base_dim, 32 * 32)
        self.voice_ctrl = nn.Linear(self.base_dim, 3)
        self.text_out = nn.Linear(self.base_dim, vocab_size)
        self.text_gate = nn.Linear(self.base_dim, 1)

        # === STATE ===
        self.pfc_state: Optional[torch.Tensor] = None
        self.growth_pressure = 0.0
        self.growth_threshold = 3.5  # Lowered slightly for responsiveness
        self.growth_patience = 150   # Set to ~3-4 seconds to prevent panic growth
        self.steps_above_threshold = 0
        self.current_cluster = 0

        # === SYNAPTIC FATIGUE (For Repetition Inhibition) ===
        # Tracks how "tired" each word neuron is.
        self.register_buffer('synaptic_fatigue', torch.zeros(vocab_size))

        # === SELECTIVE LOSS KNOWLEDGE (The "Islands of Knowledge") ===
        # Stores mastery stats for each topic cluster independently
        self.knowledge_islands = [
            ClusterKnowledge(cluster_id=i) for i in range(num_clusters)
        ]

        # Age (for exploration annealing)
        self.age = 0

        self.to(self.device)

    def get_exploration_params(self) -> Tuple[float, float]:
        """Get temperature and epsilon based on age."""
        # Early: explore widely; Late: exploit
        progress = min(1.0, self.age / 10000)
        tau = 2.0 - 1.4 * progress      # 2.0 → 0.6
        eps = 0.2 - 0.18 * progress     # 0.2 → 0.02
        return tau, eps

    def grow_cortex_layer(self) -> int:
        """Add a new cortex layer (depth growth)."""
        new_layer = nn.Linear(self.base_dim, self.base_dim).to(self.device)
        with torch.no_grad():
            new_layer.weight.copy_(
                torch.eye(self.base_dim) + torch.randn(self.base_dim, self.base_dim) * 0.01
            )
            new_layer.bias.zero_()
        self.cortex_layers.append(new_layer)
        return len(self.cortex_layers)

    def grow_cluster(self, cluster_idx: int) -> int:
        """Grow a specific cluster's capacity."""
        self.cortex.grow_cluster(cluster_idx, grow_rank=2)
        return self.cortex.get_cluster_rank(cluster_idx)

    def check_growth(self, loss: float) -> bool:
        """Check if growth should be triggered."""
        if loss > self.growth_threshold:
            self.steps_above_threshold += 1
            if self.steps_above_threshold > self.growth_patience:
                if len(self.cortex_layers) < 15:  # Safety cap
                    self.steps_above_threshold = 0
                    return True
        else:
            self.steps_above_threshold = max(0, self.steps_above_threshold - 1)
        return False

    def check_cluster_growth(self, cluster_idx: int, loss: float) -> bool:
        """Check if a specific cluster needs growth."""
        # Uses the knowledge island data
        island = self.knowledge_islands[cluster_idx]

        # Simple heuristic: If recently stuck, grow
        if len(island.recent_losses) >= 20:
            avg_loss = sum(island.recent_losses) / 20
            # If loss is consistently high but mastery is low
            if avg_loss > 3.0 and island.mastery_level < 0.2:
                # We are struggling here. Expand capacity.
                return True
        return False

    def forward(self, v: torch.Tensor, a_ext: torch.Tensor, a_self: torch.Tensor,
                t_idx: torch.Tensor, stress: float, energy: float,
                input_source: str = "AMBIENT") -> Dict[str, torch.Tensor]:

        self.age += 1

        # === INPUT ENCODING ===
        v_enc = self.eye(v)  # [batch, 64]
        a_ext_enc = self.ear_external(a_ext)  # [batch, 64]
        a_self_enc = self.self_proj(self.ear_self(a_self))  # [batch, 128]

        # Text encoding
        if t_idx.numel() > 0 and t_idx.max() < self.vocab_size:
            t_enc = self.text_embed(t_idx).mean(dim=1)
        else:
            t_enc = torch.zeros(1, self.base_dim, device=self.device)

        # === PFC CONTROL ===
        if self.pfc_state is None:
            self.pfc_state = torch.zeros(1, self.base_dim, device=self.device)

        actions, self.pfc_state = self.pfc(self.pfc_state, stress, energy, self.pfc_state)

        vis_gain = actions[0, 0]
        aud_gain = actions[0, 1]
        dream_gain = actions[0, 2]
        speak_impulse = actions[0, 3]
        cry_suppression = actions[0, 4]
        energy_conservation = actions[0, 5]
        llm_reliance = actions[0, 6]
        tts_gate = actions[0, 7]

        # === INPUT WEIGHTING ===
        if input_source == "EXTERNAL":
            input_weight = 1.0
        elif input_source == "SELF":
            input_weight = 0.3
        else:
            input_weight = 0.1

        # === INTEGRATION ===
        combined = torch.cat([v_enc * (1 + vis_gain), a_ext_enc * (1 + aud_gain)], dim=1)
        h = combined.sum(dim=1, keepdim=True).expand(-1, self.base_dim)
        h = h + a_self_enc + (t_enc * input_weight)

        # === TEMPORAL PROCESSING ===
        h_resonance = self.temporal_resonance(h)
        h = h + h_resonance * 0.2

        h_conscious, awareness, futures = self.temporal_consciousness(h.squeeze(0) if h.dim() > 1 else h)
        if h_conscious.dim() == 1:
            h_conscious = h_conscious.unsqueeze(0)
        h = h + h_conscious * 0.3

        # === SUBCONSCIOUS ===
        sub_state, sub_value, goal = self.subconscious(h, stress)
        h = h + sub_state * 0.3

        # === MULTI-SPEED PROCESSING ===
        h = self.multi_speed(h)

        # === ROUTING (Exploratory) ===
        tau, eps = self.get_exploration_params()
        self.current_cluster = self.router(h.squeeze(0), tau=tau, eps=eps)

        # Update router centroid
        with torch.no_grad():
            self.router.update_centroid(self.current_cluster, h.squeeze(0))

        # === ELASTIC CORTEX ===
        h = self.cortex(h, active_cluster=self.current_cluster)

        # === ADDITIONAL CORTEX LAYERS ===
        for layer in self.cortex_layers:
            h = F.gelu(layer(h)) + h

        # === MEMORY ===
        memory_recall = self.memory.recall(h)

        # === IMAGINATION ===
        dream_img, dream_h = self.imaginarium(h, memory_recall, sub_state, dream_gain.item())

        # === OUTPUTS ===
        cortex_img = torch.sigmoid(self.vis_out(h))
        voice_params = torch.sigmoid(self.voice_ctrl(h))

        # === TEXT GENERATION WITH FATIGUE ===
        raw_logits = self.text_out(dream_h)
        text_logits = raw_logits - (self.synaptic_fatigue * 5.0)
        self.synaptic_fatigue *= 0.98 # Decay fatigue

        text_gate_raw = torch.sigmoid(self.text_gate(h))

        # Combine speak impulse with stress penalty
        stress_penalty = max(0, stress - 0.3) * 0.8
        final_speak_drive = (text_gate_raw + speak_impulse * 0.5) * (1.0 - stress_penalty)

        return {
            'cortex_image': cortex_img,
            'dream_image': dream_img,
            'voice_params': voice_params,
            'text_logits': text_logits,
            'speak_drive': final_speak_drive,
            'speak_impulse': speak_impulse,
            'cry_suppression': cry_suppression,
            'energy_conservation': energy_conservation,
            'llm_reliance': llm_reliance,
            'tts_gate': tts_gate,
            'input_weight': torch.tensor(input_weight),
            'hidden': h,
            'actions': actions,
            'goal_momentum': goal,
            'sub_value': sub_value,
            'cluster': self.current_cluster,
            'awareness': awareness
        }


# ============================================================================
# 🔊 SECTION 7: AUDIO ENGINE (SYRINX) - UPDATED FOR DROID + INFANT SOUNDS
# ============================================================================

class EmotionalSyrinx:
    """
    Voice synthesizer with biological & mechanical layers:
    1. Drone - Base metabolic hum (Thinking)
    2. Cry - Distress signal (Biological Infant)
    3. Droid - FM Synthesis tones (R2-D2 style language)
    4. Growth - One-shot expansion sound
    """

    def __init__(self, sample_rate: int = 16000):
        self.fs = sample_rate
        self.phase = 0.0
        self.base_freq = 55.0
        self.cry_phase = 0.0
        self.data_phase = 0.0
        self.growth_triggered = False
        self.growth_countdown = 0

    def trigger_growth_sound(self):
        self.growth_triggered = True
        self.growth_countdown = int(self.fs * 0.5)

    def generate(self, frames: int, tension: float, chaos: float,
                 speak_impulse: float, stress: float, energy: float,
                 cry_suppression: float) -> np.ndarray:

        t = np.arange(frames) / self.fs
        output = np.zeros(frames)

        # === Layer 1: Thinking Drone (The "Heartbeat") ===
        # Pitch rises with tension (55Hz -> 110Hz)
        target_freq = 55.0 + (tension * 55.0)
        self.base_freq = 0.95 * self.base_freq + 0.05 * target_freq
        drone = np.sin(2 * np.pi * self.base_freq * t + self.phase) * 0.08
        self.phase += 2 * np.pi * self.base_freq * (frames / self.fs)
        output += drone

        # === Layer 2: Biological Cry (Infant-like) ===
        # Triggered by Stress. Suppressed by Learning.
        # Math: Higher pitch (450Hz base) + Second Harmonic (nasal quality) + Faster vibrato
        cry_amount = max(0, stress - 0.4) * (1.0 - cry_suppression)

        if cry_amount > 0.05:
            # Vibrato speed increases with stress (panic)
            vibrato_speed = 0.15 + (stress * 0.1)

            # Base cry pitch 450-600Hz (Infant range)
            cry_freq = 450.0 + 150.0 * np.sin(self.cry_phase)

            # Mix fundamental + 2nd harmonic for that "nasal/baby" timbre
            cry_wave = (0.7 * np.sin(2 * np.pi * cry_freq * t) +
                        0.3 * np.sin(2 * np.pi * cry_freq * 2 * t))

            output += cry_wave * cry_amount * 0.15
            self.cry_phase += vibrato_speed

        # === Layer 3: Droid Syntax (FM Synthesis / R2-D2 Style) ===
        # Triggered by Speak Impulse. Modulated by Chaos.
        # Low Chaos = Pure Whistles (Curious/Happy)
        # High Chaos = Jittery Bleeps (Excited/Urgent)

        if speak_impulse > 0.1: # Lower threshold so it can "whisper"

            # Carrier Freq: The main pitch of the whistle
            # Glides based on chaos (creates the "sliding" whistle sound)
            carrier_freq = 600.0 + 800.0 * np.sin(self.data_phase * 0.3)

            # Modulator: Creates the texture
            # High chaos = Faster modulation (metallic bleeps)
            mod_freq = 30.0 + (chaos * 300.0)

            # Index: How "deep" the modulation is
            mod_index = 2.0 + (chaos * 8.0)

            # FM Synthesis: sin(Carrier + Index * sin(Modulator))
            # Smooth sine wave base creates pleasant "whistles" instead of harsh static
            fm_tone = np.sin(
                2 * np.pi * carrier_freq * t +
                mod_index * np.sin(2 * np.pi * mod_freq * t)
            )

            # Envelope: Breathy pulses (makes it sound like syllables)
            envelope = 0.5 + 0.5 * np.sin(self.data_phase * 8.0)

            output += fm_tone * speak_impulse * 0.3 * envelope

            # Advance phase (Speed of talking)
            self.data_phase += 0.2 + (chaos * 0.3)

        # === Layer 4: Growth Gong (Evolution Event) ===
        if self.growth_countdown > 0:
            growth_t = np.arange(min(frames, self.growth_countdown)) / self.fs
            # A rich bell tone
            gong_freq = 220 * (1 + 0.5 * np.exp(-growth_t * 5))
            gong = np.sin(2 * np.pi * gong_freq * growth_t) * np.exp(-growth_t * 3) * 0.6
            output[:len(gong)] += gong
            self.growth_countdown -= frames

        return np.clip(output, -0.9, 0.9).reshape(-1, 1).astype(np.float32)


# ============================================================================
# 📁 SECTION 8: FILE INGESTION (PACMAN)
# ============================================================================

from PySide6.QtCore import Signal, QThread

class Pacman(QThread):
    """Watches training_data folder for new .txt files."""

    sig_read = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.queue = queue.Queue()
        self.folder = "training_data"

    def run(self):
        seen = set()
        os.makedirs(self.folder, exist_ok=True)

        while self.running:
            try:
                for filepath in glob.glob(f"{self.folder}/*.txt"):
                    if filepath not in seen:
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read()
                                words = re.findall(r'\b\w+\b', text.lower())
                                for word in words:
                                    self.queue.put(word)
                            seen.add(filepath)
                            self.sig_read.emit(f"📚 Ingested: {os.path.basename(filepath)}")
                        except Exception as e:
                            self.sig_read.emit(f"⚠️ Error reading {filepath}: {e}")
            except Exception:
                pass
            time.sleep(2)


# ============================================================================
# ⚙️ SECTION 9: MAIN WORKER THREAD
# ============================================================================

class VitalisWorker(QThread):
    """Main processing loop - runs the brain continuously."""

    sig_update = Signal(dict)
    sig_growth = Signal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.text_queue = queue.Queue()

        # === CORE SYSTEMS ===
        self.vocab = TabularRasaVocabulary()
        self.energy = EnergySystem()
        self.syrinx = EmotionalSyrinx()
        self.logger = SessionLogger()
        self.pacman = Pacman()
        self.tts_worker = TTSWorker()

        # === HORMONE SYSTEM ===
        self.oxytocin = 0.0  # Range 0.0 to 1.0 (The "Love/Safety" Hormone)
        self.soothe_trigger = False

        # === LEARNING CONTROL ===
        self.learning_paused = False

        # === SCAFFOLD SYSTEMS ===
        print("[INIT] Loading scaffold systems...")
        self.scaffold_llm = ScaffoldLLM()
        self.scaffold_stt = ScaffoldSTT()
        # TTS moved to worker

        # Track scaffold usage
        self.llm_reliance = 0.9
        self.tts_gate = 1.0

        # === BRAIN ===
        self.brain = OmniBrain(self.vocab.max_vocab)
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)

        # === STATE ===
        self.last_loss = 0.5
        self.current_stress = 0.0
        self.last_hidden = None
        self.efference_copy = np.zeros(1024)
        self.current_volume = 0.0
        self.is_speaking = False
        self.last_voice = np.array([0.0, 0.0, 0.0])
        self.last_cry_suppression = 0.5
        self.last_ai_output = None

        # Audio
        self.audio_queue = queue.Queue(maxsize=10)

        # Flags
        self.use_camera = True
        self.use_microphone = ENABLE_RAW_MICROPHONE and SOUNDDEVICE_AVAILABLE
        self.use_audio_output = ENABLE_RAW_AUDIO_OUTPUT and SOUNDDEVICE_AVAILABLE

        # Start Pacman & TTS
        self.pacman.start()
        self.tts_worker.start()

        print("[INIT] Worker initialized successfully")

    def soothe(self):
        """Injects a massive dose of Oxytocin."""
        self.soothe_trigger = True

    def run(self):
        print("[WORKER] Starting main loop...")

        # Setup audio output
        stream = None
        if self.use_audio_output:
            try:
                stream = sd.OutputStream(
                    samplerate=16000, channels=1,
                    callback=self._audio_callback, blocksize=512
                )
                stream.start()
                print("[AUDIO] Output stream started")
            except Exception as e:
                print(f"[AUDIO] Output error: {e}")
                stream = None

        # Setup camera
        cap = None
        if self.use_camera:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    print("[CAMERA] Opened successfully")
                else:
                    cap = None
                    self.use_camera = False
            except:
                cap = None
                self.use_camera = False

        # Setup microphone
        mic_stream = None
        if self.use_microphone:
            try:
                mic_stream = sd.InputStream(
                    samplerate=16000, channels=1, blocksize=1024,
                    callback=self._mic_callback
                )
                mic_stream.start()
                print("[MIC] Input stream started")
            except:
                mic_stream = None

        fps = 10.0
        loop_count = 0

        print("[WORKER] Entering main loop...")

        while self.running:
            try:
                loop_start = time.time()
                loop_count += 1

                # === GATHER INPUTS ===
                text_input = ""
                input_source = "AMBIENT"
                user_msg = None
                ai_msg = None
                pac_msg = None

                # User text
                try:
                    text_input = self.text_queue.get_nowait()
                    input_source = "EXTERNAL"
                    user_msg = text_input
                except queue.Empty:
                    pass

                # STT
                if self.scaffold_stt.available and loop_count % 30 == 0:
                    try:
                        speech = self.scaffold_stt.listen_once(timeout=0.3)
                        if speech:
                            text_input = text_input + " " + speech if text_input else speech
                            input_source = "EXTERNAL"
                    except:
                        pass

                # Pacman (Check if paused)
                try:
                    if not self.learning_paused:
                        pac_word = self.pacman.queue.get_nowait()
                        text_input = text_input + " " + pac_word if text_input else pac_word
                        pac_msg = f"[LEARN] {pac_word}"
                        if input_source != "EXTERNAL":
                            input_source = "EXTERNAL"
                except queue.Empty:
                    pass

                # Learn vocabulary
                if text_input:
                    self.vocab.learn_text(text_input)

                # Convert text to indices
                text_indices = self.vocab.learn_text(text_input) if text_input else []

                # === VISION ===
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (32, 32))
                        v = torch.from_numpy(gray).float().flatten().unsqueeze(0)
                        v = v / 255.0
                    else:
                        v = torch.randn(1, 32 * 32) * 0.1
                else:
                    v = torch.randn(1, 32 * 32) * 0.1
                v = v.to(self.brain.device)

                # === AUDIO ===
                try:
                    raw_audio = self.audio_queue.get_nowait()
                    a_ext = torch.from_numpy(np.abs(np.fft.fft(raw_audio.flatten(), 1024))).float().unsqueeze(0)
                except:
                    a_ext = torch.randn(1, 1024) * 0.1
                a_ext = a_ext.to(self.brain.device)

                a_self = torch.from_numpy(self.efference_copy).float().unsqueeze(0).to(self.brain.device)

                # === TEXT TENSOR ===
                if text_indices:
                    t_idx = torch.tensor([text_indices], dtype=torch.long).to(self.brain.device)
                else:
                    t_idx = torch.zeros(1, 1, dtype=torch.long).to(self.brain.device)

                # === FORWARD PASS ===
                self.brain.train()
                outputs = self.brain(
                    v, a_ext, a_self, t_idx,
                    self.current_stress, self.energy.energy,
                    input_source
                )

                # Update tracking
                self.llm_reliance = outputs['llm_reliance'].item()
                self.tts_gate = outputs['tts_gate'].item()
                self.last_cry_suppression = outputs['cry_suppression'].item()
                self.energy.conservation_gain = outputs['energy_conservation'].item()

                # === HORMONE REGULATION & STRESS UPDATE ===

                # 1. Injection (Soothe Button)
                if self.soothe_trigger:
                    self.oxytocin = 1.0  # Full dose
                    self.energy.energy = min(1.0, self.energy.energy + 0.4) # Energy boost
                    print("[HORMONE] Oxytocin flooded. Stress inhibited.")
                    self.soothe_trigger = False

                # 2. Natural Decay (It wears off slowly)
                self.oxytocin *= 0.995

                # 3. SELECTIVE STRESS CALCULATION
                active_cluster_id = outputs['cluster']
                active_island = self.brain.knowledge_islands[active_cluster_id]
                local_stress = active_island.local_stress
                sensory_load = min(1.0, self.current_volume * 2.0)
                raw_stress = (local_stress * 0.7 + sensory_load * 0.3)

                # 4. The Oxytocin Shield
                effective_stress = raw_stress * (1.0 - self.oxytocin)
                self.current_stress = 0.8 * self.current_stress + 0.2 * effective_stress

                # === STORE MEMORY ===
                if loop_count % 10 == 0:
                    importance = self.last_loss + self.current_stress
                    if importance > 0.5:
                        self.brain.memory.store(outputs['hidden'].detach(), importance)

                # === CHECK GROWTH ===
                grew = False
                if self.brain.check_growth(self.last_loss):
                    new_layers = self.brain.grow_cortex_layer()
                    self.syrinx.trigger_growth_sound()
                    self.sig_growth.emit(new_layers)
                    self.logger.log(self.last_loss, new_layers, len(self.vocab),
                                    self.energy.energy, self.current_stress, fps, "LAYER_GROWTH")
                    print(f"[GROWTH] New cortex layer! Total: {new_layers}")
                    grew = True

                if self.brain.check_cluster_growth(outputs['cluster'], self.last_loss):
                    new_rank = self.brain.grow_cluster(outputs['cluster'])
                    self.syrinx.trigger_growth_sound()
                    print(f"[GROWTH] Cluster {outputs['cluster']} expanded to rank {new_rank}")
                    grew = True

                # === GENERATE RESPONSE ===
                speak_drive = outputs['speak_drive'].item()
                self.is_speaking = False

                # ALWAYS respond to user input (EXTERNAL)
                if input_source == "EXTERNAL" and user_msg and self.energy.can_speak():
                    print(f"[RESPONSE] User said: '{user_msg}' | LLM reliance: {self.llm_reliance:.2f} | Energy: {self.energy.energy:.2f}")

                    if self.llm_reliance > 0.2:  # Lowered threshold for COCOON mode
                        # Cocoon mode: use scaffold LLM
                        ai_msg = self.scaffold_llm.chat(user_msg)
                        if ai_msg:
                            self.is_speaking = True
                            self.energy.spend_speaking(len(ai_msg.split()))
                            self.vocab.learn_text(ai_msg)
                            print(f"[COCOON] AI responds: {ai_msg[:50]}...")
                            if self.tts_gate > 0.3:
                                self.tts_worker.speak(ai_msg)
                        else:
                            ai_msg = self._generate_own_response(outputs)
                    else:
                        # Butterfly mode: own voice
                        ai_msg = self._generate_own_response(outputs)
                        if ai_msg:
                            self.is_speaking = True
                            self.energy.spend_speaking(len(ai_msg.split()))
                            print(f"[BUTTERFLY] AI responds: {ai_msg}")

                # Spontaneous speech (not triggered by user)
                elif speak_drive > 0.7 and self.energy.can_speak() and len(self.vocab) > 20:
                    ai_msg = self._generate_own_response(outputs)
                    if ai_msg:
                        self.is_speaking = True
                        self.energy.spend_speaking(len(ai_msg.split()))
                        print(f"[SPONTANEOUS] AI says: {ai_msg}")

                # === APPLY FATIGUE FOR SPOKEN WORDS ===
                if self.is_speaking and ai_msg:
                    indices = self.vocab.learn_text(ai_msg)
                    if indices:
                        idx_tensor = torch.tensor(indices, device=self.brain.device)
                        self.brain.synaptic_fatigue.index_add_(
                            0, idx_tensor,
                            torch.ones(len(indices), device=self.brain.device) * 2.0
                        )

                # === TRAINING (With Selective Loss & Metabolic Pain) ===
                if text_indices and len(text_indices) > 1:
                    target = torch.tensor(
                        [text_indices[1:] + [text_indices[-1]]],
                        dtype=torch.long
                    ).to(self.brain.device)

                    logits = outputs['text_logits']
                    if logits.dim() == 2:
                        logits = logits.unsqueeze(1).expand(-1, target.size(1), -1)

                    task_loss = F.cross_entropy(
                        logits.view(-1, self.brain.vocab_size),
                        target.view(-1),
                        ignore_index=-1
                    ) * outputs['input_weight'].item()

                    # RECORD LOCAL LOSS
                    active_island.record_loss(task_loss.item())

                    # Metabolic Pain (Energy Penalty)
                    energy_cost = max(0, 0.8 - self.energy.energy)
                    total_loss = task_loss + (energy_cost * 2.0)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                    self.optimizer.step()

                    self.last_loss = task_loss.item()

                # Spend small energy just for thinking
                self.energy.spend_thinking()
                self.energy.regenerate()

                # Update voice params
                voice = outputs['voice_params'].detach().cpu().numpy()
                if voice.ndim > 1:
                    voice = voice[0]
                self.last_voice = voice

                # === EMIT UPDATE ===
                elapsed = time.time() - loop_start
                fps = 1.0 / max(0.001, elapsed)

                update_data = {
                    'loss': self.last_loss,
                    'stress': self.current_stress,
                    'layers': len(self.brain.cortex_layers),
                    'vocab_size': len(self.vocab),
                    'energy': self.energy.energy,
                    'volume': self.current_volume,
                    'speaking': self.is_speaking,
                    'fps': fps,
                    'input_source': input_source,
                    'llm_reliance': self.llm_reliance,
                    'tts_gate': self.tts_gate,
                    'cortex_image': outputs['cortex_image'].detach().cpu().numpy().reshape(32, 32),
                    'dream_image': outputs['dream_image'].detach().cpu().numpy().reshape(32, 32),
                    'actions': outputs['actions'].detach().cpu().numpy().flatten(),
                    'ai_msg': ai_msg,
                    'user_msg': user_msg,
                    'pac_msg': pac_msg,
                    'cry_suppression': self.last_cry_suppression,
                    'cluster': outputs['cluster'],
                    'cluster_rank': self.brain.cortex.get_cluster_rank(outputs['cluster']),
                    'grew': grew,
                    'oxytocin': self.oxytocin # Send hormone level to UI
                }

                self.sig_update.emit(update_data)

                # Logging
                if loop_count % 30 == 0:
                    self.logger.log(
                        self.last_loss, len(self.brain.cortex_layers),
                        len(self.vocab), self.energy.energy,
                        self.current_stress, fps
                    )

                # Frame timing
                sleep_time = max(0.01, 0.05 - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"[ERROR] Main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        # Cleanup
        print("[WORKER] Shutting down...")
        if cap:
            cap.release()
        if stream:
            stream.stop()
        if mic_stream:
            mic_stream.stop()
        self.pacman.running = False
        self.pacman.wait()
        self.tts_worker.running = False
        self.tts_worker.wait()
        print("[WORKER] Shutdown complete")

    def _generate_own_response(self, outputs) -> Optional[str]:
        """Generate response using the AI's own learned vocabulary."""
        try:
            if len(self.vocab) < 5:
                return None

            idx = torch.argmax(outputs['text_logits'], dim=-1)
            words = []
            for i in idx.flatten():
                if i.item() < len(self.vocab.idx2word):
                    word = self.vocab.idx2word[i.item()]
                    words.append(word)

            # Take first 10 unique-ish words
            response = " ".join(words[:10])

            if response and len(response) > 2:
                return response
            return None
        except Exception as e:
            print(f"[OWN_VOICE] Error generating response: {e}")
            return None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio output callback."""
        try:
            voice = self.last_voice
            if not isinstance(voice, np.ndarray) or len(voice) < 3:
                voice = np.array([0.0, 0.0, 0.0])

            # Get raw voice chaos from the neural network
            neural_chaos = float(voice[1])

            # Apply Oxytocin filter:
            # If Oxytocin is high, it forces Chaos down (calming the voice)
            final_chaos = neural_chaos * (1.0 - self.oxytocin)

            wave = self.syrinx.generate(
                frames,
                tension=float(voice[0]),
                chaos=final_chaos,
                speak_impulse=float(voice[2]),
                stress=float(self.current_stress),
                energy=float(self.energy.energy),
                cry_suppression=float(self.last_cry_suppression)
            )

            self.efference_copy = np.abs(np.fft.fft(wave.flatten(), 1024))[:1024]
            outdata[:] = wave

        except Exception:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)

    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone input callback."""
        try:
            vol = float(np.linalg.norm(indata))
            self.current_volume = vol
            if vol > 0.01:
                try:
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass
        except:
            pass


# ============================================================================
# 🖥️ SECTION 10: UI COMPONENTS
# ============================================================================

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QProgressBar, QFrame, QSplitter, QGridLayout,
    QPushButton, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl


class LivingOrb(gl.GLViewWidget):
    """
    Organic neural visualization with data-driven waves.
    """

    def __init__(self):
        super().__init__()
        self.opts['distance'] = 35
        self.opts['fov'] = 60
        self.setBackgroundColor('#030308')

        # Geometry
        self.n_rings = 48
        self.n_points = 100
        self.base_radius = 8.0
        self.current_radius = 8.0

        # Wave state
        self.learning_wave = np.zeros(self.n_rings)
        self.stress_wave = np.zeros(self.n_rings)
        self.growth_wave = np.zeros(self.n_rings)
        self.voice_wave = 0.0

        # Phases
        self.learning_phase = 0.0
        self.stress_phase = 0.0
        self.growth_phase = 0.0
        self.time = 0.0

        # Data tracking
        self.last_loss = 1.0
        self.loss_derivative = 0.0
        self.growth_triggered = False
        self.speak_intensity = 0.0

        # Create geometry
        self.ring_items = []
        self._create_rings()
        self._create_core()

    def _create_rings(self):
        for i in range(self.n_rings):
            lat = -np.pi / 2 + (np.pi * i / (self.n_rings - 1))
            theta = np.linspace(0, 2 * np.pi, self.n_points)
            r = self.base_radius * np.cos(lat)
            z = self.base_radius * np.sin(lat)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z_arr = np.full(self.n_points, z)
            pos = np.column_stack((x, y, z_arr))

            t = i / (self.n_rings - 1)
            base_color = np.array([0.2 + 0.3 * t, 0.8 - 0.3 * t, 0.9, 0.8])
            colors = np.tile(base_color, (self.n_points, 1))

            line = gl.GLLinePlotItem(
                pos=pos, color=colors, width=3.5,
                antialias=True, mode='line_strip'
            )
            self.addItem(line)
            self.ring_items.append({
                'item': line, 'base_lat': lat, 'base_r': r,
                'base_z': z, 'theta': theta, 'ring_idx': i
            })

    def _create_core(self):
        n = 400
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        r = 3.0
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        self.core_pos = np.column_stack((x, y, z))
        self.core_colors = np.ones((n, 4)) * np.array([0.4, 0.6, 0.9, 0.5])

        self.core = gl.GLScatterPlotItem(
            pos=self.core_pos, color=self.core_colors, size=5, pxMode=True
        )
        self.addItem(self.core)

    def set_data(self, loss: float, stress: float, is_speaking: bool,
                 growth_event: bool, voice_intensity: float = 0.0):
        """Feed actual data to drive visualization."""
        # Learning wave
        self.loss_derivative = self.last_loss - loss
        self.last_loss = loss

        if self.loss_derivative > 0:
            intensity = min(1.0, self.loss_derivative * 10)
            self.learning_wave = 0.8 * self.learning_wave + 0.2 * intensity
        else:
            self.learning_wave *= 0.95

        self.learning_phase += 0.15

        # Stress wave
        self.stress_wave = 0.7 * self.stress_wave + 0.3 * stress
        if stress > 0.5:
            self.stress_phase += 0.3 * stress
        else:
            self.stress_phase += 0.1

        # Growth wave
        if growth_event:
            self.growth_triggered = True
            self.growth_phase = 0.0
            self.base_radius += 0.5

        if self.growth_triggered:
            self.growth_wave = np.maximum(self.growth_wave, np.exp(-self.growth_phase * 0.5))
            self.growth_phase += 0.1
            if self.growth_phase > 5:
                self.growth_triggered = False
        else:
            self.growth_wave *= 0.95

        # Voice
        self.speak_intensity = voice_intensity if is_speaking else self.speak_intensity * 0.9
        if is_speaking:
            self.voice_wave = 0.5 + 0.5 * np.sin(self.time * 10)
        else:
            self.voice_wave = 0.0

        # Breathing
        if is_speaking:
            self.current_radius = self.base_radius * (1.0 + 0.18 * np.sin(self.time * 7))
        else:
            self.current_radius = 0.93 * self.current_radius + 0.07 * self.base_radius

        self.time += 0.05

    def update_visualization(self):
        """Update ring geometries."""
        try:
            for ring_data in self.ring_items:
                i = ring_data['ring_idx']
                lat = ring_data['base_lat']
                theta = ring_data['theta']

                base_r = self.current_radius * np.cos(lat)
                base_z = self.current_radius * np.sin(lat)

                # Wave deformations
                learn_disp = self.learning_wave[i] * np.sin(theta * 3 + self.learning_phase) * 1.8
                stress_disp = self.stress_wave[i] * (
                    np.sin(theta * 7 + self.stress_phase) * 1.2 +
                    np.sin(theta * 11 + self.stress_phase * 1.5) * 0.8
                )
                growth_disp = self.growth_wave[i] * np.sin(theta * 2 + self.growth_phase * 5) * 2.5

                if self.speak_intensity > 0.1:
                    voice_sync = self.speak_intensity * np.sin(self.time * 10) * 1.5
                    learn_disp += voice_sync
                    stress_disp += voice_sync * 0.5

                ambient = np.sin(theta * 4 + self.time * 2) * 0.15

                total_disp = learn_disp + stress_disp + growth_disp + ambient
                r = base_r + total_disp

                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z_arr = np.full(self.n_points, base_z)
                pos = np.column_stack((x, y, z_arr))

                # Colors
                t = i / (self.n_rings - 1)
                r_col = 0.2 + 0.3 * t + self.stress_wave[i] * 0.5 + self.speak_intensity * 0.3
                g_col = 0.8 - 0.3 * t + self.learning_wave[i] * 0.3
                b_col = 0.9 + self.growth_wave[i] * 0.1
                alpha = 0.7 + self.learning_wave[i] * 0.2 + self.speak_intensity * 0.1

                colors = np.zeros((self.n_points, 4))
                colors[:, 0] = min(1.0, r_col)
                colors[:, 1] = min(1.0, g_col)
                colors[:, 2] = min(1.0, b_col)
                colors[:, 3] = min(0.9, alpha)

                ring_data['item'].setData(pos=pos, color=colors)

            # Core
            total_activity = (
                np.mean(np.abs(self.learning_wave)) * 2 +
                np.mean(np.abs(self.stress_wave)) * 1.5 +
                np.mean(self.growth_wave) * 2 +
                self.speak_intensity * 2
            )

            core_brightness = 0.4 + min(0.6, total_activity * 0.4)

            base_r = 0.3 + self.speak_intensity * 0.6 + np.mean(np.abs(self.stress_wave)) * 0.5
            base_g = 0.5 + np.mean(self.learning_wave) * 0.5
            base_b = 0.8 + np.mean(self.growth_wave) * 0.2

            self.core_colors[:, 0] = min(1.0, base_r)
            self.core_colors[:, 1] = min(1.0, base_g)
            self.core_colors[:, 2] = min(1.0, base_b)
            self.core_colors[:, 3] = min(0.8, core_brightness)

            if self.speak_intensity > 0.1:
                pulse = 1 + 0.5 * np.sin(self.time * 10)
                self.core.setData(pos=self.core_pos * pulse, color=self.core_colors)
            else:
                self.core.setData(color=self.core_colors)

            self.opts['azimuth'] = self.opts.get('azimuth', 0) + 0.3

        except Exception as e:
            print(f"[ORB] Error: {e}")

    def trigger_growth(self):
        self.growth_triggered = True
        self.growth_phase = 0.0
        self.base_radius += 0.5


class SciFiPanel(QFrame):
    """Styled panel for UI sections."""

    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet("""
            background: rgba(0, 20, 30, 180);
            border: 1px solid #00FFFF;
            border-radius: 6px;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel(title)
        label.setStyleSheet("""
            color: #00FFFF;
            font-family: 'Consolas', monospace;
            font-weight: bold;
            font-size: 10pt;
            border: none;
            background: none;
        """)
        layout.addWidget(label)
        self.content = layout


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MagnumOpusVitalis v2.0 - Genesis Unified")
        self.resize(1700, 950)
        self.setStyleSheet("""
            background: #050505;
            color: #00FF88;
            font-family: 'Consolas', monospace;
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === LEFT: ORB + STATS ===
        left = QWidget()
        l_layout = QVBoxLayout(left)
        l_layout.setContentsMargins(0, 0, 0, 0)

        self.orb = LivingOrb()
        self.orb.setMinimumSize(500, 500)
        l_layout.addWidget(self.orb, 4)

        # Stats panel
        stats_panel = SciFiPanel("NEURAL TELEMETRY")
        self.stats_label = QLabel("INITIALIZING...")
        self.stats_label.setStyleSheet("color: #00FF88; font-size: 11pt; border: none;")
        self.stats_label.setWordWrap(True)
        stats_panel.content.addWidget(self.stats_label)
        l_layout.addWidget(stats_panel, 1)

        main_layout.addWidget(left, 2)

        # === RIGHT: MONITORS + CHAT ===
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)

        # Top: Optical array
        optical_panel = SciFiPanel("OPTICAL ARRAY")
        opt_layout = QHBoxLayout()

        # Cortex view
        cortex_frame = QFrame()
        cortex_layout = QVBoxLayout(cortex_frame)
        cortex_layout.setContentsMargins(2, 2, 2, 2)
        self.cortex_label = QLabel("CORTEX")
        self.cortex_label.setStyleSheet("color: #00FF00; font-size: 9pt; border: none;")
        self.cortex_img = QLabel()
        self.cortex_img.setFixedSize(160, 160)
        self.cortex_img.setStyleSheet("background: #000; border: 1px solid #00FF00;")
        cortex_layout.addWidget(self.cortex_label)
        cortex_layout.addWidget(self.cortex_img)
        opt_layout.addWidget(cortex_frame)

        # Dream view
        dream_frame = QFrame()
        dream_layout = QVBoxLayout(dream_frame)
        dream_layout.setContentsMargins(2, 2, 2, 2)
        self.dream_label = QLabel("DREAM")
        self.dream_label.setStyleSheet("color: #FF00FF; font-size: 9pt; border: none;")
        self.dream_img = QLabel()
        self.dream_img.setFixedSize(160, 160)
        self.dream_img.setStyleSheet("background: #000; border: 1px solid #FF00FF;")
        dream_layout.addWidget(self.dream_label)
        dream_layout.addWidget(self.dream_img)
        opt_layout.addWidget(dream_frame)

        optical_panel.content.addLayout(opt_layout)
        r_layout.addWidget(optical_panel, 1)

        # Middle: Telemetry bars
        telem_panel = SciFiPanel("PFC OUTPUTS")
        telem_grid = QGridLayout()

        self.pfc_bars = {}
        bar_names = ['VIS', 'AUD', 'DRM', 'SPK', 'CRY↓', 'NRG↓', 'LLM', 'TTS']
        for i, name in enumerate(bar_names):
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #888; font-size: 8pt; border: none;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setStyleSheet("::chunk { background: #00FFFF; }")
            telem_grid.addWidget(lbl, i // 4, (i % 4) * 2)
            telem_grid.addWidget(bar, i // 4, (i % 4) * 2 + 1)
            self.pfc_bars[name] = bar

        telem_panel.content.addLayout(telem_grid)

        # Energy bar
        energy_label = QLabel("BIO-ENERGY")
        energy_label.setStyleSheet("color: #FFD700; font-size: 9pt; border: none;")
        self.energy_bar = QProgressBar()
        self.energy_bar.setRange(0, 100)
        self.energy_bar.setStyleSheet("::chunk { background: #FFD700; }")
        telem_panel.content.addWidget(energy_label)
        telem_panel.content.addWidget(self.energy_bar)

        r_layout.addWidget(telem_panel, 1)

        # Bottom: Chat
        split = QSplitter(Qt.Vertical)

        chat_panel = SciFiPanel("COMM LINK")
        self.chat_txt = QTextEdit()
        self.chat_txt.setReadOnly(True)
        self.chat_txt.setStyleSheet("""
            background: #001122;
            color: #00FF88;
            border: none;
            font-size: 11pt;
            font-weight: bold;
        """)

        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self._send_message)
        self.input_field.setStyleSheet("""
            background: #002233;
            color: #FFF;
            border: 1px solid #005577;
            padding: 5px;
        """)
        self.input_field.setPlaceholderText("TRANSMIT TO AI...")

        # === ADD PAUSE SWITCH ===
        self.pause_learning_cb = QCheckBox("PAUSE BACKGROUND LEARNING")
        self.pause_learning_cb.setStyleSheet("""
            QCheckBox { color: #888; font-size: 9pt; }
            QCheckBox::indicator { border: 1px solid #555; width: 12px; height: 12px; }
            QCheckBox::indicator:checked { background: #00FF88; }
        """)
        self.pause_learning_cb.toggled.connect(self._toggle_learning)

        # === ADD SOOTHE BUTTON ===
        self.soothe_btn = QPushButton("TRANSMIT CALM (432Hz)")
        self.soothe_btn.setCursor(Qt.PointingHandCursor)
        self.soothe_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 255, 136, 20);
                border: 1px solid #00FF88;
                color: #00FF88;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                margin-top: 5px;
            }
            QPushButton:hover {
                background: rgba(0, 255, 136, 50);
                border: 1px solid #FFFFFF;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background: #00FF88;
                color: #000000;
            }
        """)
        self.soothe_btn.clicked.connect(self._on_soothe_clicked)

        chat_panel.content.addWidget(self.chat_txt)
        chat_panel.content.addWidget(self.pause_learning_cb)
        chat_panel.content.addWidget(self.input_field)
        chat_panel.content.addWidget(self.soothe_btn)
        split.addWidget(chat_panel)

        r_layout.addWidget(split, 2)
        main_layout.addWidget(right, 3)

        # === START WORKER ===
        self.worker = VitalisWorker()
        self.worker.sig_update.connect(self._on_update)
        self.worker.sig_growth.connect(self._on_growth)
        self.worker.pacman.sig_read.connect(self._on_pacman)
        self.worker.start()

    def _send_message(self):
        text = self.input_field.text().strip()
        if text:
            self.worker.text_queue.put(text)
            self.chat_txt.append(f"<span style='color:#FFFFFF'>YOU: {text}</span>")
            self.input_field.clear()

    def _toggle_learning(self, checked: bool):
        if hasattr(self, 'worker'):
            self.worker.learning_paused = checked
            state = "PAUSED" if checked else "RESUMED"
            color = "#FF9900" if checked else "#00FF88"
            self.chat_txt.append(f"<i style='color:{color}'>*** BACKGROUND LEARNING {state} ***</i>")

    def _on_soothe_clicked(self):
        """Send calming signal to the brain."""
        if hasattr(self, 'worker'):
            self.worker.soothe()
            self.chat_txt.append(
                "<i style='color:#00FF88'>*** TRANSMITTING CALMING FREQUENCY ***</i>"
            )

    def _on_update(self, data: dict):
        try:
            # Orb visualization
            self.orb.set_data(
                loss=data['loss'],
                stress=data['stress'],
                is_speaking=data['speaking'],
                growth_event=data.get('grew', False),
                voice_intensity=data['volume'] if data['speaking'] else 0.0
            )
            self.orb.update_visualization()

            # Stats
            mode = "COCOON" if data['llm_reliance'] > 0.3 else "BUTTERFLY"
            mode_color = "#FF9900" if mode == "COCOON" else "#00FF00"
            src_color = "#00FF00" if data['input_source'] == "EXTERNAL" else "#666666"

            self.stats_label.setText(
                f"LAYERS: {data['layers']} | "
                f"VOCAB: {data['vocab_size']} | "
                f"LOSS: {data['loss']:.3f} | "
                f"OXYTOCIN: {data.get('oxytocin', 0.0):.2f} | "
                f"<span style='color:{mode_color}'>{mode}</span> | "
                f"<span style='color:{src_color}'>IN:{data['input_source']}</span> | "
                f"FPS: {data['fps']:.1f}"
            )

            # PFC bars
            if 'actions' in data and len(data['actions']) >= 8:
                actions = data['actions']
                bar_map = ['VIS', 'AUD', 'DRM', 'SPK', 'CRY↓', 'NRG↓', 'LLM', 'TTS']
                for i, name in enumerate(bar_map):
                    self.pfc_bars[name].setValue(int(actions[i] * 100))

            # Energy
            self.energy_bar.setValue(int(data['energy'] * 100))

            # Images
            def to_pixmap(arr):
                img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                qimg = QImage(img, 32, 32, 32, QImage.Format_Grayscale8)
                return QPixmap.fromImage(qimg).scaled(160, 160)

            self.cortex_img.setPixmap(to_pixmap(data['cortex_image']))
            self.dream_img.setPixmap(to_pixmap(data['dream_image']))

            # Chat messages
            if data.get('ai_msg'):
                self.chat_txt.append(f"<span style='color:#00FF00'>AI: {data['ai_msg']}</span>")
            if data.get('pac_msg'):
                self.chat_txt.append(f"<span style='color:#555555'>{data['pac_msg']}</span>")

        except Exception as e:
            print(f"[UI] Update error: {e}")

    def _on_growth(self, layers: int):
        self.orb.trigger_growth()
        self.chat_txt.append(
            f"<b style='color:#FF3300'>*** NEURAL EXPANSION *** Layers: {layers}</b>"
        )

    def _on_pacman(self, msg: str):
        self.chat_txt.append(f"<span style='color:#555555'>{msg}</span>")

    def closeEvent(self, event):
        self.worker.running = False
        self.worker.wait()
        event.accept()


# ============================================================================
# 🚀 ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MAGNUMOPUSVITALIS v2.4 - Genesis Unified + Synaptic Fatigue + Selective Loss")
    print("  'A seed that grows, not a machine that thinks.'")
    print("=" * 70)
    print()
    print("📋 STARTUP CHECKLIST:")
    print(f"  • PyTorch: {'✅ CUDA' if torch.cuda.is_available() else '⚠️ CPU only'}")
    print(f"  • LLM Scaffold: {'✅ Available' if SCAFFOLD_LLM_AVAILABLE else '❌ Not installed'}")
    print(f"  • STT Scaffold: {'✅ Available' if SCAFFOLD_STT_AVAILABLE else '❌ Not installed'}")
    print(f"  • TTS Scaffold: {'✅ Available' if SCAFFOLD_TTS_AVAILABLE else '❌ Not installed'}")
    print(f"  • Audio Output: {'✅ Enabled' if ENABLE_RAW_AUDIO_OUTPUT else '⚪ Disabled'}")
    print(f"  • Microphone: {'✅ Enabled' if ENABLE_RAW_MICROPHONE else '⚪ Disabled'}")
    print()
    print("🎯 HOW TO INTERACT:")
    print("  • Type in the COMM LINK box and press Enter")
    print("  • Drop .txt files in 'training_data/' folder for learning")
    print("  • Press 'TRANSMIT CALM' to calm the AI if it gets stressed")
    print("  • Use the 'Pause Learning' switch to focus the conversation")
    print()
    print("🧠 MODES:")
    print("  • COCOON: AI uses GPT-2 scaffold for responses (early stage)")
    print("  • BUTTERFLY: AI uses its own learned vocabulary (mature stage)")
    print()
    print("=" * 70)
    print("Starting UI...")
    print()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())