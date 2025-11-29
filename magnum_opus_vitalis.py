"""
MagnumOpusVitalis: The Living Intelligence
================================================================
ARCHITECT: Alan Hourmand
VERSION: 1.0 (Genesis)

PHILOSOPHY:
"A seed that grows, not a machine that thinks."

CORE PRINCIPLES:
1. TABULA RASA - Starts knowing nothing, learns everything
2. MULTI-SCALE ABSTRACTION - Fast pattern recognition, slow understanding
3. ORGANIC GROWTH - Expands only when confused
4. UNIFIED MEMORY - Memory IS the model, not separate
5. ENERGY ECONOMICS - Speaking costs energy, silence recovers

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
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ============================================================================
# 🥚 SCAFFOLD SYSTEMS (Training Wheels)
# ============================================================================

# Try to import optional scaffold dependencies
SCAFFOLD_LLM_AVAILABLE = False
SCAFFOLD_STT_AVAILABLE = False
SCAFFOLD_TTS_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    SCAFFOLD_LLM_AVAILABLE = True
    print("[SCAFFOLD] LLM available (transformers)")
except ImportError:
    print("[SCAFFOLD] LLM not available - install: pip install transformers")

try:
    import speech_recognition as sr
    SCAFFOLD_STT_AVAILABLE = True
    print("[SCAFFOLD] STT available (speech_recognition)")
except ImportError:
    print("[SCAFFOLD] STT not available - install: pip install SpeechRecognition")

try:
    import pyttsx3
    SCAFFOLD_TTS_AVAILABLE = True
    print("[SCAFFOLD] TTS available (pyttsx3)")
except ImportError:
    print("[SCAFFOLD] TTS not available - install: pip install pyttsx3")


class ScaffoldLLM:
    """
    The YOLK - A tiny pre-trained LLM that provides initial language capability.

    The AI uses this heavily at first, then gradually learns to rely on its own
    understanding. Like training wheels that get removed.

    Reliance factor:
    - 0.9 at birth (almost fully dependent)
    - Decreases as own vocab/understanding grows
    - Eventually ~0.1 (barely uses it, mostly independent)
    """

    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'

        if SCAFFOLD_LLM_AVAILABLE:
            try:
                print("[LLM YOLK] Loading tiny GPT-2...")
                # Use the smallest GPT-2 variant
                self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
                self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # Try to use GPU if available
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.to(self.device)

                self.model.eval()  # Inference only
                self.available = True
                print(f"[LLM YOLK] Ready on {self.device}")
            except Exception as e:
                print(f"[LLM YOLK] Failed to load: {e}")
                self.available = False

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate text continuation from prompt"""
        if not self.available:
            return ""

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Return only the new part
            new_text = generated[len(prompt):].strip()
            # Take just the first few words
            words = new_text.split()[:3]
            return " ".join(words)
        except Exception as e:
            print(f"[LLM YOLK] Generation error: {e}")
            return ""

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get semantic embedding of text for understanding boost"""
        if not self.available:
            return None

        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.transformer(inputs)
                # Get last hidden state, average over sequence
                embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.cpu()
        except Exception:
            return None


class ScaffoldSTT:
    """
    Speech-to-Text scaffold - lets the AI "hear" spoken words.
    Runs in background, queues recognized speech.
    """

    def __init__(self):
        self.available = SCAFFOLD_STT_AVAILABLE
        self.recognizer = None
        self.microphone = None
        self.speech_queue = queue.Queue()
        self.listening = False

        if self.available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                print("[STT] Ready")
            except Exception as e:
                print(f"[STT] Init error: {e}")
                self.available = False

    def listen_once(self, timeout: float = 2.0) -> Optional[str]:
        """Try to capture and recognize speech once"""
        if not self.available:
            return None

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)

            # Try Google Speech Recognition (free, no API key needed)
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None  # Couldn't understand
        except Exception as e:
            return None


class ScaffoldTTS:
    """
    Text-to-Speech scaffold - gives the AI a voice.

    The AI can LEARN to turn this off (tts_gate output from PFC).
    Initially always on (gate=1.0), but as it develops, it might
    choose silence (gate→0.0).
    """

    def __init__(self):
        self.available = SCAFFOLD_TTS_AVAILABLE
        self.engine = None
        self.speaking = False
        self.speech_queue = queue.Queue()

        if self.available:
            try:
                self.engine = pyttsx3.init()
                # Configure voice
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 0.8)

                # Try to get a neutral voice
                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)

                print("[TTS] Ready")
            except Exception as e:
                print(f"[TTS] Init error: {e}")
                self.available = False

    def speak(self, text: str, gate: float = 1.0):
        """
        Speak text, modulated by gate.
        gate=1.0: Always speak
        gate=0.0: Stay silent (AI learned to be quiet)
        """
        if not self.available or gate < 0.3:  # Below 0.3 = effectively muted
            return

        if not text or len(text.strip()) == 0:
            return

        try:
            # Don't speak if already speaking
            if self.speaking:
                return

            self.speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.speaking = False
        except Exception as e:
            self.speaking = False

    def speak_async(self, text: str, gate: float = 1.0):
        """Queue speech to not block main thread"""
        if gate < 0.3:
            return
        self.speech_queue.put((text, gate))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QProgressBar, QFrame, QGridLayout
)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 📊 SECTION 0: UTILITIES & DATA STRUCTURES
# ============================================================================

@dataclass
class EpisodeMemory:
    """A single memory trace - stores embedding + context"""
    embedding: torch.Tensor
    timestamp: float
    importance: float = 1.0
    access_count: int = 0

    def decay(self, rate: float = 0.995):
        self.importance *= rate


class SessionLogger:
    """Flight recorder for debugging and analysis"""

    def __init__(self, filename: str = "brain_log.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.last_log = 0
        self._init_file()

    def _init_file(self):
        try:
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Time", "Event", "Loss", "Layers", "Vocab_Size",
                    "Energy", "Stress", "FPS"
                ])
        except Exception:
            pass

    def log(self, loss: float, layers: int, vocab_size: int,
            energy: float, stress: float, fps: float, event: str = None):
        now = time.time()
        if (now - self.last_log > 1.0) or event:
            self.last_log = now
            try:
                with open(self.filename, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        f"{now - self.start_time:.2f}",
                        event or "-",
                        f"{loss:.4f}",
                        layers,
                        vocab_size,
                        f"{energy:.2f}",
                        f"{stress:.2f}",
                        f"{fps:.1f}"
                    ])
            except Exception:
                pass


class TabularRasaVocabulary:
    """
    CRITICAL: Starts COMPLETELY EMPTY.
    Learns every word it encounters. No pre-loaded vocabulary.
    """

    def __init__(self, max_vocab: int = 10000):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.word_counts: Dict[str, int] = {}
        self.recent_words: deque = deque(maxlen=100)  # Echo reflex buffer
        self.max_vocab = max_vocab
        self.total_words_seen = 0

    def learn_word(self, word: str) -> int:
        """Learn a new word, return its index"""
        word = word.lower().strip()
        if not word:
            return -1

        self.total_words_seen += 1

        if word not in self.word2idx:
            if len(self.idx2word) >= self.max_vocab:
                return -1  # Vocabulary full
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            self.word_counts[word] = 0

        self.word_counts[word] += 1
        self.recent_words.append(self.word2idx[word])
        return self.word2idx[word]

    def learn_text(self, text: str) -> List[int]:
        """Learn all words from text, return indices"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.learn_word(w) for w in words if self.learn_word(w) >= 0]

    def get_recent_indices(self) -> List[int]:
        """Get indices of recently seen words (for echo reflex)"""
        return list(self.recent_words)

    def __len__(self) -> int:
        return len(self.idx2word)


class EnergySystem:
    """
    Biological energy economics - NOW WITH LEARNED CONSERVATION

    The PFC learns an energy_conservation gain that modulates:
    - How much energy speaking costs
    - How quickly to regenerate

    This way the AI learns NOT to burn energy too fast.
    """

    def __init__(self):
        self.energy = 1.0
        self.max_energy = 1.0
        self.base_regen_rate = 0.015       # Base recovery per tick (slower)
        self.base_speak_cost = 0.03        # Base cost per word (lower)
        self.think_cost = 0.0005           # Very small baseline cost
        self.fatigue_threshold = 0.2       # Below this = too tired to speak

        # Learned modulation (comes from PFC)
        self.conservation_gain = 0.5       # 0 = wasteful, 1 = very conservative

    def set_conservation(self, gain: float):
        """PFC sets this based on learning"""
        self.conservation_gain = max(0.0, min(1.0, gain))

    def can_speak(self) -> bool:
        return self.energy > self.fatigue_threshold

    def spend_speaking(self, num_words: int = 1):
        # Cost is REDUCED when conservation is high (AI learned to be efficient)
        effective_cost = self.base_speak_cost * (1.5 - self.conservation_gain)
        self.energy = max(0, self.energy - (effective_cost * num_words))

    def spend_thinking(self):
        self.energy = max(0, self.energy - self.think_cost)

    def regenerate(self, multiplier: float = 1.0):
        # Regen is INCREASED when conservation is high
        effective_regen = self.base_regen_rate * (0.5 + self.conservation_gain)
        self.energy = min(self.max_energy, self.energy + effective_regen * multiplier)

    def get_output_scale(self) -> float:
        """Scale output volume/activity by energy level"""
        return 0.3 + (self.energy * 0.7)


@dataclass
class InputEvent:
    """
    Tracks WHERE input came from - critical for avoiding feedback loops.

    Sources:
    - EXTERNAL: User typing, file ingestion (PRIMARY importance)
    - SELF: AI's own output (SECONDARY importance - reduced learning)
    - AMBIENT: Background noise, silence (minimal learning)
    """
    text: str
    source: str  # "EXTERNAL", "SELF", "AMBIENT"
    timestamp: float
    importance: float = 1.0  # Learning weight multiplier


# ============================================================================
# 🧠 SECTION 1: NEURAL COMPONENTS
# ============================================================================

class SubconsciousMind(nn.Module):
    """
    4-Layer Subconscious:
    1. Sea of Noise - Creative randomness from LEARNED patterns
    2. Peak Detector - Filter for relevant activations
    3. Future Generator - Simulate possible outcomes
    4. Evaluator - Pick the best path

    Also maintains goal_momentum for emergent goal formation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 1. Sea of Noise - uses learned basis vectors (not hardcoded)
        self.noise_basis = nn.Parameter(torch.randn(32, dim) * 0.02)
        self.noise_proj = nn.Linear(dim, dim)

        # 2. Peak Detector
        self.peak_scorer = nn.Linear(dim, 1)

        # 3. Future Generator (GRU for temporal simulation)
        self.future_gen = nn.GRUCell(dim, dim)

        # 4. Evaluator
        self.evaluator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Goal momentum - emergent attractor
        self.goal_momentum: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor, stress: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = h.device

        # 1. Sea of Noise - creative randomness scaled by stress
        noise_weights = torch.randn(1, 32, device=device) * (0.1 + stress * 0.3)
        creative_noise = noise_weights @ self.noise_basis
        noisy_h = self.noise_proj(h + creative_noise)

        # 2. Peak Detection - filter relevant activations
        peaks = torch.sigmoid(self.peak_scorer(noisy_h))
        filtered_h = noisy_h * peaks

        # 3. Future Generation - simulate outcomes
        if self.goal_momentum is None:
            self.goal_momentum = torch.zeros_like(h)
        future_h = self.future_gen(filtered_h, self.goal_momentum.detach())

        # 4. Evaluation - assess path quality
        value = self.evaluator(future_h)

        # Update goal momentum (emergent goals from experience)
        self.goal_momentum = 0.95 * self.goal_momentum + 0.05 * future_h.detach()

        return future_h, value, self.goal_momentum


class MultiSpeedProcessor(nn.Module):
    """
    Multi-scale abstraction learning:
    - Fast channels (8, 16 dim): Quick pattern recognition
    - Medium channels (32, 64 dim): Structural understanding
    - Slow channels (128 dim): Deep comprehension

    Trust weights shift over time (fast starts high, slow grows).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.input_dim = dim
        self.speed_dims = [8, 16, 32, 64, 128]

        # Projections to each speed
        self.proj_in = nn.ModuleList([
            nn.Linear(dim, d) for d in self.speed_dims
        ])

        # Projections back to common dimension
        self.proj_out = nn.ModuleList([
            nn.Linear(d, dim) for d in self.speed_dims
        ])

        # Trust weights - fast channels start trusted, slow grow over time
        # Initial: [1.0, 0.8, 0.5, 0.2, 0.1]
        self.trust = nn.Parameter(torch.tensor([1.0, 0.8, 0.5, 0.2, 0.1]))

        # Age counter for trust evolution
        self.age = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for proj_in, proj_out in zip(self.proj_in, self.proj_out):
            # Think at this speed
            thought = F.gelu(proj_in(x))
            # Project back to common language
            result = proj_out(thought)
            outputs.append(result)

        # Stack and mix based on trust
        stacked = torch.stack(outputs, dim=0)  # [5, batch, seq, dim]
        weights = F.softmax(self.trust, dim=0)

        # Weighted sum across speeds
        mixed = torch.einsum('sbld,s->bld', stacked, weights)

        return mixed

    def age_tick(self):
        """Called periodically to shift trust toward slower channels"""
        self.age += 1
        if self.age % 1000 == 0:
            with torch.no_grad():
                # Slowly increase trust in deeper channels
                shift = torch.tensor([-0.01, -0.005, 0.0, 0.005, 0.01])
                self.trust.data = torch.clamp(self.trust.data + shift, 0.05, 2.0)


class BiologicalMemory(nn.Module):
    """
    Unified memory system - NOT separate from the model.
    Stores embeddings of significant moments.
    Retrieval is by pattern matching, not lookup.
    """

    def __init__(self, dim: int, capacity: int = 1000):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.memories: List[EpisodeMemory] = []

        # Memory encoder/decoder
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, dim)

    def store(self, state: torch.Tensor, importance: float = 1.0):
        """Store a memory if significant enough"""
        if importance < 0.3:
            return  # Not significant enough

        encoded = self.encoder(state.detach().mean(dim=0 if state.dim() > 1 else None))

        memory = EpisodeMemory(
            embedding=encoded.cpu(),
            timestamp=time.time(),
            importance=importance
        )

        # Decay existing memories
        for m in self.memories:
            m.decay()

        self.memories.append(memory)

        # Prune if over capacity
        if len(self.memories) > self.capacity:
            # Keep most important
            self.memories.sort(key=lambda m: m.importance * (m.access_count + 1), reverse=True)
            self.memories = self.memories[:int(self.capacity * 0.9)]

    def recall(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve most similar memory"""
        if not self.memories:
            return None

        query_flat = query.detach().view(-1)
        if query_flat.shape[0] > self.dim:
            query_flat = query_flat[:self.dim]
        elif query_flat.shape[0] < self.dim:
            query_flat = F.pad(query_flat, (0, self.dim - query_flat.shape[0]))

        best_memory = None
        best_sim = -1.0

        # Sample subset for efficiency
        candidates = random.sample(self.memories, min(len(self.memories), 50))

        for memory in candidates:
            mem_flat = memory.embedding.view(-1)
            if mem_flat.shape[0] != query_flat.shape[0]:
                continue
            sim = F.cosine_similarity(query_flat.cpu(), mem_flat, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_memory = memory

        if best_sim > 0.6 and best_memory:
            best_memory.access_count += 1
            best_memory.importance = min(2.0, best_memory.importance * 1.1)
            return self.decoder(best_memory.embedding.to(query.device))

        return None


class Imaginarium(nn.Module):
    """
    Dream generator - mixes reality, memory, and subconscious.
    Creates internal visualizations and imaginings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, reality: torch.Tensor, memory: Optional[torch.Tensor],
                subconscious: torch.Tensor, gain: float) -> Tuple[torch.Tensor, torch.Tensor]:

        if memory is None:
            memory = torch.zeros_like(reality)

        # Ensure all have same shape
        if memory.shape != reality.shape:
            memory = memory.view_as(reality) if memory.numel() >= reality.numel() else torch.zeros_like(reality)

        combined = torch.cat([reality, memory, subconscious], dim=-1)
        dream_state = torch.tanh(self.mixer(combined)) * (0.5 + gain)
        dream_image = self.decoder(dream_state)

        return dream_image, dream_state


class PrefrontalCortex(nn.Module):
    """
    Executive control - learns to regulate the system.

    Outputs (8 learned gains):
    - vis_gain: How much to attend to vision
    - aud_gain: How much to attend to audio
    - dream_gain: How much imagination to mix in
    - speak_impulse: Drive to vocalize
    - cry_suppression: Learned emotional regulation
    - energy_conservation: Learned energy management
    - llm_reliance: How much to rely on scaffold LLM (starts HIGH, learns to reduce)
    - tts_gate: Whether to use text-to-speech (starts HIGH, can learn to mute)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.planner = nn.GRUCell(dim + 2, dim)  # +2 for stress and energy
        self.policy = nn.Sequential(
            nn.Linear(dim, 8),  # 8 outputs now
            nn.Sigmoid()
        )

        # Initialize bias for llm_reliance and tts_gate to start HIGH
        # This makes the AI dependent on scaffolds initially
        with torch.no_grad():
            # Outputs 6 and 7 (llm_reliance, tts_gate) should start near 1.0
            self.policy[0].bias[6] = 2.0  # Sigmoid(2) ≈ 0.88
            self.policy[0].bias[7] = 2.0  # Sigmoid(2) ≈ 0.88

    def forward(self, h: torch.Tensor, stress: float, energy: float,
                prev_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if prev_state is None:
            prev_state = torch.zeros_like(h)
        else:
            prev_state = prev_state.detach()

        # Include stress and energy as inputs
        context = torch.tensor([[stress, energy]], device=h.device)
        inp = torch.cat([h, context], dim=1)

        new_state = self.planner(inp, prev_state)
        actions = self.policy(new_state)

        # actions: [vis_gain, aud_gain, dream_gain, speak_impulse, cry_suppression,
        #           energy_conservation, llm_reliance, tts_gate]
        return actions, new_state


# ============================================================================
# 🧬 SECTION 2: THE OMNIBRAIN (Main Model)
# ============================================================================

class OmniBrain(nn.Module):
    """
    The complete living brain.

    Starts microscopic, grows organically.
    All knowledge is learned, nothing hardcoded.
    """

    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.base_dim = 128

        # === SENSORY INPUTS ===
        self.eye = nn.Sequential(
            nn.Linear(32 * 32, 64),
            nn.Tanh()
        )
        self.ear_external = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Tanh()
        )
        self.ear_self = nn.Sequential(  # Efference copy
            nn.Linear(1024, 32),
            nn.Tanh()
        )
        self.self_proj = nn.Linear(32, self.base_dim)
        self.text_embed = nn.Embedding(vocab_size, self.base_dim)

        # === CORE SYSTEMS ===
        self.pfc = PrefrontalCortex(self.base_dim)
        self.subconscious = SubconsciousMind(self.base_dim)
        self.multi_speed = MultiSpeedProcessor(self.base_dim)
        self.memory = BiologicalMemory(self.base_dim)
        self.imaginarium = Imaginarium(self.base_dim)

        # === CORTEX (Growable) ===
        self.cortex = nn.ModuleList([nn.Linear(self.base_dim, self.base_dim)])

        # === OUTPUTS ===
        self.vis_out = nn.Linear(self.base_dim, 32 * 32)
        self.voice_ctrl = nn.Linear(self.base_dim, 3)  # tension, chaos, raw_impulse
        self.text_out = nn.Linear(self.base_dim, vocab_size)
        self.text_gate = nn.Linear(self.base_dim, 1)

        # === STATE ===
        self.pfc_state: Optional[torch.Tensor] = None
        self.growth_pressure = 0.0
        self.growth_threshold = 5.0     # Higher threshold (was 3.5)
        self.growth_patience = 100      # Need more sustained confusion (was 50)
        self.steps_above_threshold = 0

        self.to(self.device)

    def grow_cortex(self) -> int:
        """Add a new cortex layer (depth growth)"""
        new_layer = nn.Linear(self.base_dim, self.base_dim).to(self.device)
        # Initialize as identity + small noise
        with torch.no_grad():
            new_layer.weight.copy_(torch.eye(self.base_dim) + torch.randn(self.base_dim, self.base_dim) * 0.01)
            new_layer.bias.zero_()
        self.cortex.append(new_layer)
        return len(self.cortex)

    def check_growth(self, loss: float) -> bool:
        """Check if we should grow based on sustained confusion"""
        # Only grow if loss is REALLY high and sustained
        # Also require minimum steps between growths
        if loss > self.growth_threshold:
            self.steps_above_threshold += 1
            # Need MORE patience - 100 steps instead of 50
            if self.steps_above_threshold >= self.growth_patience:
                self.steps_above_threshold = 0
                # Cap max layers to prevent runaway growth
                if len(self.cortex) < 20:  # Max 20 layers
                    return True
        else:
            # Reset faster when doing well
            self.steps_above_threshold = max(0, self.steps_above_threshold - 2)
        return False

    def forward(self, vision: torch.Tensor, audio_ext: torch.Tensor,
                audio_self: torch.Tensor, text_indices: torch.Tensor,
                stress: float, energy: float,
                input_source: str = "EXTERNAL") -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the brain.

        Args:
            input_source: "EXTERNAL" (user/file), "SELF" (own output), "AMBIENT" (noise)
                         This affects learning weight.

        Returns dict with all outputs for flexible use.
        """
        # === SENSORY PROCESSING ===
        v = self.eye(vision)
        a_ext = self.ear_external(audio_ext)
        a_self = self.self_proj(self.ear_self(audio_self))

        # Text embedding (handle empty/invalid)
        if text_indices.numel() > 0 and text_indices.max() < self.vocab_size:
            t = self.text_embed(text_indices).mean(dim=1)
        else:
            t = torch.zeros(1, self.base_dim, device=self.device)

        # === EXECUTIVE CONTROL ===
        # Bootstrap hidden state for PFC
        if self.pfc_state is None:
            ghost_h = torch.zeros(1, self.base_dim, device=self.device)
        else:
            ghost_h = self.pfc_state.detach()

        actions, self.pfc_state = self.pfc(ghost_h, stress, energy, self.pfc_state)
        vis_gain, aud_gain, dream_gain, speak_impulse, cry_suppression, energy_conservation, llm_reliance, tts_gate = actions[0]

        # Apply gains to sensory input
        v = v * (1.0 + vis_gain)
        a_ext = a_ext * (1.0 + aud_gain)

        # === INPUT SOURCE WEIGHTING ===
        # External input gets full attention, self-output gets reduced
        if input_source == "EXTERNAL":
            input_weight = 1.0
        elif input_source == "SELF":
            input_weight = 0.3  # Learn less from own output to avoid loops
        else:  # AMBIENT
            input_weight = 0.1

        # === INTEGRATION ===
        # Combine all inputs (weighted by source)
        combined = torch.cat([v, a_ext], dim=1)  # [batch, 128]
        h = combined + a_self + (t * input_weight)

        # === SUBCONSCIOUS ===
        sub_state, sub_value, goal = self.subconscious(h, stress)
        h = h + sub_state * 0.3  # Subtle subconscious influence

        # === MULTI-SPEED PROCESSING ===
        h = h.unsqueeze(1)  # Add sequence dim for multi-speed
        h = self.multi_speed(h)
        h = h.squeeze(1)

        # === CORTEX (Deep processing) ===
        for layer in self.cortex:
            h = F.gelu(layer(h)) + h  # Residual

        # === MEMORY ===
        memory_recall = self.memory.recall(h)

        # === IMAGINATION ===
        dream_img, dream_h = self.imaginarium(h, memory_recall, sub_state, dream_gain.item())

        # === OUTPUTS ===
        cortex_img = torch.sigmoid(self.vis_out(h))
        voice_params = torch.sigmoid(self.voice_ctrl(h))
        text_logits = self.text_out(dream_h)  # From imaginarium for creativity
        text_gate_raw = torch.sigmoid(self.text_gate(h))

        # Combine speak impulse from PFC with text gate
        # BUT reduce speak drive when stressed (should cry, not talk)
        stress_penalty = max(0, stress - 0.3) * 0.8  # High stress = less talking
        final_speak_drive = (text_gate_raw + speak_impulse * 0.5) * (1.0 - stress_penalty)

        return {
            'cortex_image': cortex_img,
            'dream_image': dream_img,
            'voice_params': voice_params,  # [tension, chaos, raw_impulse]
            'text_logits': text_logits,
            'speak_drive': final_speak_drive,
            'speak_impulse': speak_impulse,
            'cry_suppression': cry_suppression,
            'energy_conservation': energy_conservation,
            'llm_reliance': llm_reliance,      # How much to use scaffold LLM
            'tts_gate': tts_gate,              # Whether to use text-to-speech
            'input_weight': torch.tensor(input_weight),
            'hidden': h,
            'actions': actions,
            'goal_momentum': goal,
            'sub_value': sub_value
        }


# ============================================================================
# 🔊 SECTION 3: AUDIO ENGINE (SYRINX)
# ============================================================================

class EmotionalSyrinx:
    """
    Voice synthesizer with emotional layers:
    1. Drone - Base thinking hum
    2. Crying - Distress vocalization (learnable suppression)
    3. Speech - Data transmission tones
    4. Growth - One-shot expansion sound
    """

    def __init__(self, sample_rate: int = 16000):
        self.fs = sample_rate
        self.phase = 0.0
        self.base_freq = 55.0
        self.mod_phase = 0.0
        self.cry_phase = 0.0
        self.data_phase = 0.0
        self.growth_buffer: Optional[np.ndarray] = None
        self.growth_position = 0

    def trigger_growth_sound(self):
        """Queue a growth sound"""
        duration = 0.5
        frames = int(self.fs * duration)
        t = np.arange(frames) / self.fs

        # Low gong
        gong = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 4)
        # Shimmer
        shimmer = np.sin(2 * np.pi * 400 * t) * np.exp(-t * 6) * 0.3
        # Rising tone
        rise = np.sin(2 * np.pi * (100 + 200 * t) * t) * np.exp(-t * 3) * 0.2

        self.growth_buffer = np.clip(gong + shimmer + rise, -0.9, 0.9)
        self.growth_position = 0

    def generate(self, frames: int, tension: float, chaos: float,
                 speak_impulse: float, stress: float, energy: float,
                 cry_suppression: float) -> np.ndarray:
        """
        Generate audio frame.

        Args:
            tension: Thinking intensity (0-1)
            chaos: Randomness/confusion (0-1)
            speak_impulse: Drive to vocalize (0-1)
            stress: Current stress level (0-1)
            energy: Current energy level (0-1)
            cry_suppression: Learned suppression of crying (0-1)
        """
        t = np.arange(frames) / self.fs
        output = np.zeros(frames)

        # === LAYER 1: DRONE (Always on) ===
        target_freq = 55.0 + (tension * 55.0)
        self.base_freq = 0.95 * self.base_freq + 0.05 * target_freq

        throb_rate = 2.0 + (tension * 8.0)
        throb = np.sin(2 * np.pi * throb_rate * t + self.mod_phase)
        self.mod_phase += 2 * np.pi * throb_rate * (frames / self.fs)

        carrier = np.tanh(5.0 * np.sin(2 * np.pi * self.base_freq * t + self.phase))
        drone = carrier * (0.25 + 0.15 * throb)
        self.phase += 2 * np.pi * self.base_freq * (frames / self.fs)

        output += drone

        # === LAYER 2: CRYING (Triggered by confusion/stress, learnable suppression) ===
        # Cry more easily when confused - lower threshold
        cry_amount = max(0, stress - 0.3) * (1.0 - cry_suppression) * (1.0 - energy * 0.3)
        if cry_amount > 0.05:  # Lower threshold to trigger crying
            # Warbling cry frequency - more distressed
            cry_freq = 350 + 250 * np.sin(self.cry_phase * 6)  # More warble
            cry = np.sin(2 * np.pi * cry_freq * t) * cry_amount * 0.6  # Louder
            # Sob modulation - more pronounced
            sob = 0.4 + 0.6 * np.sin(2 * np.pi * 4 * t + self.cry_phase)  # Faster sobs
            cry *= sob
            self.cry_phase += 2 * np.pi * 4 * (frames / self.fs)
            output += cry

        # === LAYER 3: SPEECH (When impulse high and energy sufficient) ===
        if speak_impulse > 0.5 and energy > 0.15:
            data_freq = 800.0 + 400.0 * np.round(np.sin(
                2 * np.pi * (10 + chaos * 20) * t + self.data_phase
            ))
            speech = np.sign(np.sin(2 * np.pi * data_freq * t)) * 0.25 * speak_impulse
            self.data_phase += 2 * np.pi * 20 * (frames / self.fs)
            output += speech

        # === LAYER 4: GROWTH SOUND (One-shot) ===
        if self.growth_buffer is not None:
            remaining = len(self.growth_buffer) - self.growth_position
            to_add = min(frames, remaining)
            output[:to_add] += self.growth_buffer[self.growth_position:self.growth_position + to_add] * 0.5
            self.growth_position += to_add
            if self.growth_position >= len(self.growth_buffer):
                self.growth_buffer = None

        # === MASTER VOLUME (Energy-aware) ===
        output *= (0.4 + energy * 0.6)

        return np.clip(output, -0.9, 0.9).reshape(-1, 1).astype(np.float32)


# ============================================================================
# 📂 SECTION 4: FILE READER (PACMAN)
# ============================================================================

class Pacman(QThread):
    """
    Background file reader - continuously ingests training data.
    Watches the training_data folder for new .txt files.
    """

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
                                # Chunk into words
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
# ⚙️ SECTION 5: MAIN WORKER THREAD
# ============================================================================

class VitalisWorker(QThread):
    """
    Main processing loop - runs the brain continuously.
    """

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

        # === SCAFFOLD SYSTEMS (Training Wheels) ===
        print("[INIT] Loading scaffold systems...")
        self.scaffold_llm = ScaffoldLLM()      # The "yolk" - pre-trained language
        self.scaffold_stt = ScaffoldSTT()      # Speech-to-text input
        self.scaffold_tts = ScaffoldTTS()      # Text-to-speech output

        # Track scaffold usage for logging
        self.llm_reliance = 0.9   # Start heavily dependent
        self.tts_gate = 1.0       # TTS on by default

        # === BRAIN ===
        self.brain = OmniBrain(self.vocab.max_vocab)
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)

        # === STATE (Initialize ALL before any threads start) ===
        self.last_loss = 0.5
        self.current_stress = 0.0
        self.last_hidden = None
        self.efference_copy = np.zeros(1024)
        self.current_volume = 0.0
        self.is_speaking = False
        self.last_voice = np.array([0.0, 0.0, 0.0])  # Initialize BEFORE audio starts
        self.last_cry_suppression = 0.5
        self.last_ai_output = None  # Track self-output for feedback prevention

        # Recent context for LLM prompting
        self.recent_words = []  # Last few words for context

        # Audio
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size

        # Flags for optional features
        self.use_camera = True
        self.use_microphone = True
        self.use_audio_output = True

        # Start Pacman
        self.pacman.start()

        print("[INIT] Worker initialized successfully")

    def run(self):
        print("[WORKER] Starting main loop...")

        # Setup audio OUTPUT stream
        stream = None
        if self.use_audio_output:
            try:
                stream = sd.OutputStream(
                    samplerate=16000,
                    channels=1,
                    callback=self._audio_callback,
                    blocksize=512
                )
                stream.start()
                print("[AUDIO] Output stream started")
            except Exception as e:
                print(f"[AUDIO] Output error (continuing without): {e}")
                stream = None
                self.use_audio_output = False

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
                    print("[CAMERA] Could not open (continuing without)")
                    cap = None
                    self.use_camera = False
            except Exception as e:
                print(f"[CAMERA] Error (continuing without): {e}")
                cap = None
                self.use_camera = False

        # Setup microphone
        mic_stream = None
        if self.use_microphone:
            try:
                mic_stream = sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    blocksize=1024,
                    callback=self._mic_callback
                )
                mic_stream.start()
                print("[MIC] Input stream started")
            except Exception as e:
                print(f"[MIC] Error (continuing without): {e}")
                mic_stream = None
                self.use_microphone = False

        frame_time = time.time()
        fps = 10.0
        loop_count = 0

        print("[WORKER] Entering main loop...")

        while self.running:
            try:
                loop_start = time.time()
                loop_count += 1

                if loop_count % 100 == 0:
                    print(f"[LOOP] Frame {loop_count}, FPS: {fps:.1f}, Vocab: {len(self.vocab)}")

                # === GATHER SENSORY INPUT ===

                # Vision
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (32, 32)) / 255.0
                    else:
                        gray = np.random.rand(32, 32) * 0.1
                else:
                    gray = np.random.rand(32, 32) * 0.1

                vision_tensor = torch.tensor(gray.flatten(), dtype=torch.float32).unsqueeze(0).to(self.brain.device)

                # Audio (external)
                try:
                    audio_data = self.audio_queue.get_nowait()
                    audio_fft = np.abs(np.fft.fft(audio_data.flatten(), 1024))[:1024]
                    self.current_volume = float(np.linalg.norm(audio_data))
                except queue.Empty:
                    audio_fft = np.zeros(1024)

                audio_ext_tensor = torch.tensor(audio_fft, dtype=torch.float32).unsqueeze(0).to(self.brain.device)

                # Audio (self - efference copy)
                audio_self_tensor = torch.tensor(self.efference_copy, dtype=torch.float32).unsqueeze(0).to(self.brain.device)

                # === INPUT SOURCE TRACKING ===
                # Critical: differentiate EXTERNAL input from SELF output
                text_input = ""
                user_msg = None
                input_source = "AMBIENT"  # Default

                # Text from user (HIGHEST PRIORITY - EXTERNAL)
                try:
                    user_msg = self.text_queue.get_nowait()
                    text_input = user_msg
                    input_source = "EXTERNAL"  # User input is primary
                except queue.Empty:
                    pass

                # === SPEECH-TO-TEXT (Scaffold) ===
                # Check for speech input periodically (not every frame - expensive)
                if self.scaffold_stt.available and loop_count % 30 == 0:  # Every ~1.5 sec
                    try:
                        speech = self.scaffold_stt.listen_once(timeout=0.5)
                        if speech:
                            text_input = speech if not text_input else text_input + " " + speech
                            input_source = "EXTERNAL"
                            print(f"[STT] Heard: {speech}")
                    except Exception as e:
                        pass  # Don't block on STT errors

                # Text from Pacman/files (also EXTERNAL)
                pac_msg = None
                try:
                    pac_word = self.pacman.queue.get_nowait()
                    text_input += " " + pac_word if text_input else pac_word
                    pac_msg = f"[LEARN] {pac_word}"
                    if input_source != "EXTERNAL":  # Don't override user input
                        input_source = "EXTERNAL"
                except queue.Empty:
                    pass

                # Track if this is self-generated (from last AI output)
                # If we're learning from our own output, mark it as SELF
                if hasattr(self, 'last_ai_output') and self.last_ai_output:
                    if text_input == "" and self.last_ai_output:
                        # No new input, but we had output - could learn from self
                        # But we DON'T want to by default to avoid loops
                        pass
                    self.last_ai_output = None  # Clear after checking

                # Learn vocabulary from all text
                if text_input:
                    self.vocab.learn_text(text_input)

                # Convert text to indices
                text_indices = []
                if text_input:
                    words = re.findall(r'\b\w+\b', text_input.lower())
                    for w in words:
                        if w in self.vocab.word2idx:
                            text_indices.append(self.vocab.word2idx[w])

                if text_indices:
                    text_tensor = torch.tensor([text_indices], dtype=torch.long).to(self.brain.device)
                else:
                    text_tensor = torch.zeros(1, 1, dtype=torch.long).to(self.brain.device)

                # === FORWARD PASS (with input source) ===
                self.brain.train()

                with torch.no_grad():  # Use no_grad for inference to save memory
                    outputs = self.brain(
                        vision_tensor,
                        audio_ext_tensor,
                        audio_self_tensor,
                        text_tensor,
                        self.current_stress,
                        self.energy.energy,
                        input_source  # Pass the source
                    )

                # === UPDATE ENERGY CONSERVATION FROM PFC ===
                energy_conservation = float(outputs['energy_conservation'].item())
                self.energy.set_conservation(energy_conservation)

                # === LEARNING (weighted by input source) ===
                loss_val = 0.0
                input_weight = float(outputs['input_weight'].item())

                if text_indices and len(text_indices) > 1 and input_source == "EXTERNAL":
                    # Only do full learning from EXTERNAL sources
                    # This prevents autistic-like feedback loops

                    outputs = self.brain(
                        vision_tensor,
                        audio_ext_tensor,
                        audio_self_tensor,
                        text_tensor,
                        self.current_stress,
                        self.energy.energy,
                        input_source
                    )

                    # Predict next token
                    target = torch.tensor([text_indices[1:] + [text_indices[-1]]], dtype=torch.long).to(self.brain.device)
                    logits = outputs['text_logits']

                    # Ensure logits match target length
                    if logits.dim() == 2:
                        logits = logits.unsqueeze(1).expand(-1, target.size(1), -1)

                    # Weight loss by input importance
                    loss = F.cross_entropy(
                        logits.view(-1, self.brain.vocab_size),
                        target.view(-1),
                        ignore_index=-1
                    ) * input_weight  # Scale by source importance

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                    self.optimizer.step()

                    loss_val = loss.item()
                    self.last_loss = loss_val

                # === UPDATE STRESS ===
                confusion = min(1.0, self.last_loss / 5.0)
                sensory_load = min(1.0, self.current_volume * 2.0)
                self.current_stress = 0.7 * self.current_stress + 0.3 * (confusion * 0.7 + sensory_load * 0.3)

                # === STORE MEMORY (occasionally) ===
                if loop_count % 10 == 0:
                    importance = self.last_loss + self.current_stress
                    if importance > 0.5:
                        self.brain.memory.store(outputs['hidden'].detach(), importance)

                # === CHECK GROWTH ===
                if self.brain.check_growth(self.last_loss):
                    new_layers = self.brain.grow_cortex()
                    self.syrinx.trigger_growth_sound()
                    self.sig_growth.emit(new_layers)
                    self.logger.log(self.last_loss, new_layers, len(self.vocab),
                                  self.energy.energy, self.current_stress, fps, "GROWTH")
                    print(f"[GROWTH] New cortex layer! Total: {new_layers}")

                # === GENERATE OUTPUT ===
                ai_msg = None
                speak_drive = float(outputs['speak_drive'].item())
                speak_impulse = float(outputs['speak_impulse'].item())
                cry_suppression = float(outputs['cry_suppression'].item())
                self.last_cry_suppression = cry_suppression

                # Get scaffold controls from PFC
                self.llm_reliance = float(outputs['llm_reliance'].item())
                self.tts_gate = float(outputs['tts_gate'].item())

                # === SPEAKING THRESHOLD (stress = less talking, more crying) ===
                # When confused/stressed, the AI should CRY, not TALK
                # Higher stress = higher threshold = harder to speak
                base_threshold = 0.7  # Raised from 0.6
                stress_penalty = self.current_stress * 0.4  # Stress makes it harder to talk
                threshold = base_threshold + stress_penalty

                # Also need minimum vocab and energy
                can_speak = (
                    speak_drive > threshold and
                    self.energy.can_speak() and
                    len(self.vocab) > 10 and  # Need some vocab first
                    self.current_stress < 0.6  # Too stressed = can't talk coherently
                )

                if can_speak:
                    # === YOLK SYSTEM: Mix LLM with own output ===
                    words_out = []

                    # Get own prediction
                    logits = outputs['text_logits'].clone().detach()
                    recent = self.vocab.get_recent_indices()
                    for idx in recent:
                        if idx < logits.size(-1):
                            logits[0, idx] += 1.5

                    valid_len = len(self.vocab)
                    if valid_len < logits.size(-1):
                        logits[0, valid_len:] = float('-inf')

                    probs = F.softmax(logits, dim=-1)

                    # How many words to generate
                    num_words = 1
                    if speak_drive > threshold + 0.2:
                        num_words = 2

                    for _ in range(num_words):
                        try:
                            # Try own prediction first
                            own_word_idx = torch.multinomial(probs, 1).item()
                            own_word = self.vocab.idx2word.get(own_word_idx, "")

                            # If LLM available and reliance is high, maybe use LLM instead
                            if self.scaffold_llm.available and self.llm_reliance > 0.3:
                                # Build context from recent words
                                context = " ".join(self.recent_words[-5:]) if self.recent_words else ""
                                llm_word = self.scaffold_llm.generate(context, max_tokens=3)

                                if llm_word and random.random() < self.llm_reliance:
                                    # Use LLM word (scaffolded)
                                    words_out.append(llm_word.split()[0] if llm_word else own_word)
                                else:
                                    # Use own word (independent)
                                    if own_word:
                                        words_out.append(own_word)
                            else:
                                # No LLM or low reliance - use own word
                                if own_word:
                                    words_out.append(own_word)

                        except Exception:
                            break

                    if words_out:
                        ai_msg = " ".join(words_out)
                        self.energy.spend_speaking(len(words_out))
                        self.is_speaking = True
                        self.last_ai_output = ai_msg

                        # Update recent words for context
                        self.recent_words.extend(words_out)
                        self.recent_words = self.recent_words[-10:]  # Keep last 10

                        # === TTS OUTPUT (if gate is open) ===
                        if self.scaffold_tts.available and self.tts_gate > 0.3:
                            # Speak asynchronously to not block
                            self.scaffold_tts.speak_async(ai_msg, self.tts_gate)
                else:
                    self.is_speaking = False
                    self.energy.regenerate(1.5 if speak_drive < 0.3 else 1.0)
                    self.last_ai_output = None

                self.energy.spend_thinking()

                # Age the multi-speed processor
                self.brain.multi_speed.age_tick()

                # === UPDATE VOICE PARAMS (for audio callback) ===
                voice_params = outputs['voice_params'][0].detach().cpu().numpy()
                self.last_voice = voice_params

                # === EMIT UPDATE ===
                dt = time.time() - frame_time
                frame_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, dt))

                activations = outputs['hidden'].detach().cpu().numpy().flatten()

                update_data = {
                    'activations': activations,
                    'stress': self.current_stress,
                    'speaking': self.is_speaking,
                    'volume': self.current_volume,
                    'loss': self.last_loss,
                    'layers': len(self.brain.cortex),
                    'vocab_size': len(self.vocab),
                    'energy': self.energy.energy,
                    'energy_conservation': self.energy.conservation_gain,
                    'llm_reliance': self.llm_reliance,     # Scaffold: how much using LLM
                    'tts_gate': self.tts_gate,             # Scaffold: TTS on/off
                    'input_source': input_source,
                    'fps': fps,
                    'real_image': gray,
                    'cortex_image': outputs['cortex_image'].detach().cpu().numpy().reshape(32, 32),
                    'dream_image': outputs['dream_image'].detach().cpu().numpy().reshape(32, 32),
                    'actions': outputs['actions'].detach().cpu().numpy().flatten(),
                    'ai_msg': ai_msg,
                    'user_msg': user_msg,
                    'pac_msg': pac_msg,
                    'cry_suppression': cry_suppression
                }

                self.sig_update.emit(update_data)

                # Logging (less frequent)
                if loop_count % 30 == 0:
                    self.logger.log(
                        self.last_loss,
                        len(self.brain.cortex),
                        len(self.vocab),
                        self.energy.energy,
                        self.current_stress,
                        fps
                    )

                # Frame timing
                elapsed = time.time() - loop_start
                sleep_time = max(0.01, 0.05 - elapsed)  # ~20 FPS target (more conservative)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"[ERROR] Main loop exception: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Don't spam on repeated errors

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
        print("[WORKER] Shutdown complete")

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio output callback - must be thread-safe and never crash"""
        try:
            # Get voice parameters safely
            voice = getattr(self, 'last_voice', np.array([0.0, 0.0, 0.0]))

            # Ensure voice is numpy array with correct shape
            if torch.is_tensor(voice):
                voice = voice.detach().cpu().numpy()
            if not isinstance(voice, np.ndarray):
                voice = np.array([0.0, 0.0, 0.0])
            if len(voice) < 3:
                voice = np.array([0.0, 0.0, 0.0])

            # Get other params safely
            stress = getattr(self, 'current_stress', 0.0)
            energy = getattr(self, 'energy', None)
            energy_val = energy.energy if energy else 0.5
            cry_sup = getattr(self, 'last_cry_suppression', 0.5)

            wave = self.syrinx.generate(
                frames,
                tension=float(voice[0]),
                chaos=float(voice[1]),
                speak_impulse=float(voice[2]),
                stress=float(stress),
                energy=float(energy_val),
                cry_suppression=float(cry_sup)
            )

            # Store efference copy safely
            try:
                self.efference_copy = np.abs(np.fft.fft(wave.flatten(), 1024))[:1024]
            except Exception:
                pass

            outdata[:] = wave

        except Exception as e:
            # On any error, output silence
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)

    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone input callback - must be thread-safe"""
        try:
            vol = float(np.linalg.norm(indata))
            self.current_volume = vol
            if vol > 0.01:
                # Don't block if queue is full
                try:
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass
        except Exception:
            pass


# ============================================================================
# 🖥️ SECTION 6: UI COMPONENTS
# ============================================================================

class LivingOrb(gl.GLViewWidget):
    """
    Organic neural visualization with data-driven waves:

    WAVE CHANNELS (each visible as distinct patterns):
    1. LEARNING - Cyan ribbons, pulse when loss decreases
    2. STRESS - Red turbulent waves, intensity = stress level
    3. GROWTH - Purple expanding pulse on cortex growth
    4. VOICE - All waves synchronize + sphere breathes when speaking

    Based on reference images: flowing ribbon topology around hollow core
    """

    def __init__(self):
        super().__init__()
        self.opts['distance'] = 35
        self.opts['fov'] = 60
        self.setBackgroundColor('#030308')

        # === GEOMETRY PARAMETERS ===
        self.n_rings = 48          # MORE horizontal slices for detail
        self.n_points = 100        # MORE points per ring for smoother waves
        self.base_radius = 8.0
        self.current_radius = 8.0

        # === WAVE STATE (driven by actual data) ===
        self.learning_wave = np.zeros(self.n_rings)      # Learning activity
        self.stress_wave = np.zeros(self.n_rings)        # Stress level
        self.growth_wave = np.zeros(self.n_rings)        # Growth pulses
        self.voice_wave = 0.0                            # Speaking intensity

        # Wave propagation phases
        self.learning_phase = 0.0
        self.stress_phase = 0.0
        self.growth_phase = 0.0
        self.time = 0.0

        # Data tracking for derivatives
        self.last_loss = 1.0
        self.loss_derivative = 0.0
        self.growth_triggered = False
        self.speak_intensity = 0.0

        # === CREATE GEOMETRY ===
        self.ring_items = []
        self._create_rings()

        # Add subtle core glow
        self._create_core()

    def _create_rings(self):
        """Create the ring structure - stacked circles forming sphere"""
        for i in range(self.n_rings):
            # Latitude from -π/2 to π/2
            lat = -np.pi/2 + (np.pi * i / (self.n_rings - 1))

            # Create ring at this latitude
            theta = np.linspace(0, 2*np.pi, self.n_points)
            r = self.base_radius * np.cos(lat)
            z = self.base_radius * np.sin(lat)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z_arr = np.full(self.n_points, z)

            pos = np.column_stack((x, y, z_arr))

            # Base color - gradient from cyan (top) to magenta (bottom)
            t = i / (self.n_rings - 1)
            base_color = np.array([
                0.2 + 0.3 * t,      # R: more at bottom
                0.8 - 0.3 * t,      # G: more at top
                0.9,                 # B: constant high
                0.8                  # A: more opaque for visibility
            ])
            colors = np.tile(base_color, (self.n_points, 1))

            line = gl.GLLinePlotItem(
                pos=pos,
                color=colors,
                width=3.5,           # THICKER lines for visibility
                antialias=True,
                mode='line_strip'
            )
            self.addItem(line)
            self.ring_items.append({
                'item': line,
                'base_lat': lat,
                'base_r': r,
                'base_z': z,
                'theta': theta,
                'ring_idx': i
            })

    def _create_core(self):
        """Create inner core glow effect - BRIGHTER"""
        # Larger inner sphere of points for core glow
        n = 400  # More points
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices

        r = 3.0  # Bigger core
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        self.core_pos = np.column_stack((x, y, z))
        self.core_colors = np.ones((n, 4)) * np.array([0.4, 0.6, 0.9, 0.5])

        self.core = gl.GLScatterPlotItem(
            pos=self.core_pos,
            color=self.core_colors,
            size=5,  # Bigger points
            pxMode=True
        )
        self.addItem(self.core)

    def set_data(self, loss: float, stress: float, is_speaking: bool,
                 growth_event: bool, voice_intensity: float = 0.0):
        """
        Feed actual data to drive the visualization.
        Call this every frame with current brain state.
        """
        # === LEARNING WAVE ===
        # Driven by loss DECREASE (active learning)
        loss_delta = self.last_loss - loss
        self.loss_derivative = 0.8 * self.loss_derivative + 0.2 * max(0, loss_delta * 10)
        self.last_loss = loss

        # Propagate learning wave from center outward - MODERATE
        self.learning_phase += 0.15 * (1 + self.loss_derivative * 1.5)
        for i in range(self.n_rings):
            dist_from_center = abs(i - self.n_rings/2) / (self.n_rings/2)
            wave_val = np.sin(self.learning_phase - dist_from_center * 3) * 0.5 + 0.5
            # Base wave + learning boost
            self.learning_wave[i] = wave_val * (self.loss_derivative * 2.5 + 0.2)

        # === STRESS WAVE ===
        # Turbulent, chaotic when stressed - MODERATE
        self.stress_phase += 0.2 * (1 + stress * 2)
        for i in range(self.n_rings):
            # Multiple frequencies for turbulence
            f1 = np.sin(self.stress_phase * 1.5 + i * 0.5)
            f2 = np.sin(self.stress_phase * 2.3 + i * 0.8) * 0.6
            f3 = np.sin(self.stress_phase * 3.7 + i * 0.3) * 0.4
            # Moderate stress response
            self.stress_wave[i] = (f1 + f2 + f3) * (stress * 1.0 + 0.05)

        # === GROWTH WAVE ===
        # One-shot expanding pulse - MODERATE
        if growth_event:
            self.growth_triggered = True
            self.growth_phase = 0.0

        if self.growth_triggered:
            self.growth_phase += 0.08
            for i in range(self.n_rings):
                dist = abs(i - self.n_rings/2) / (self.n_rings/2)
                if dist < self.growth_phase:
                    intensity = max(0, 1 - (self.growth_phase - dist) * 1.5)
                    self.growth_wave[i] = intensity * 2.0
                else:
                    self.growth_wave[i] *= 0.92

            if self.growth_phase > 2.5:
                self.growth_triggered = False
        else:
            self.growth_wave *= 0.92

        # === VOICE WAVE ===
        # When speaking, all waves synchronize and sphere breathes - MODERATE
        if is_speaking:
            self.speak_intensity = min(1.0, self.speak_intensity + 0.15)
            self.voice_wave = 0.7 + 0.3 * np.sin(self.time * 8)
        else:
            self.speak_intensity = max(0, self.speak_intensity - 0.04)
            self.voice_wave = 0.0

        # Sphere breathing when speaking - MODERATE
        if is_speaking:
            self.current_radius = self.base_radius * (1.0 + 0.18 * np.sin(self.time * 7))
        else:
            self.current_radius = 0.93 * self.current_radius + 0.07 * self.base_radius

        self.time += 0.05

    def update_visualization(self):
        """Update all ring geometries based on current wave state"""
        try:
            for ring_data in self.ring_items:
                i = ring_data['ring_idx']
                lat = ring_data['base_lat']
                theta = ring_data['theta']

                # Base radius at this latitude (sphere shape)
                base_r = self.current_radius * np.cos(lat)
                base_z = self.current_radius * np.sin(lat)

                # === APPLY WAVE DEFORMATIONS (BALANCED) ===
                # Each wave adds radial displacement

                # Learning: smooth sine wave - moderate amplitude
                learn_disp = self.learning_wave[i] * np.sin(theta * 3 + self.learning_phase) * 1.8

                # Stress: turbulent high-frequency - noticeable but not crazy
                stress_disp = self.stress_wave[i] * (
                    np.sin(theta * 7 + self.stress_phase) * 1.2 +
                    np.sin(theta * 11 + self.stress_phase * 1.5) * 0.8 +
                    np.sin(theta * 5 + self.stress_phase * 2.3) * 0.5
                )

                # Growth: uniform expansion pulse - visible but not extreme
                growth_disp = self.growth_wave[i] * np.sin(theta * 2 + self.growth_phase * 5) * 2.5

                # Voice: synchronized pulse on all channels - moderate sync
                if self.speak_intensity > 0.1:
                    voice_sync = self.speak_intensity * np.sin(self.time * 10) * 1.5
                    learn_disp += voice_sync
                    stress_disp += voice_sync * 0.5
                    growth_disp += voice_sync * 0.3

                # Add subtle ambient wave for liveliness even when idle
                ambient = np.sin(theta * 4 + self.time * 2) * 0.15 + np.sin(theta * 6 - self.time * 1.5) * 0.1

                # Combined radial displacement
                total_disp = learn_disp + stress_disp + growth_disp + ambient
                r = base_r + total_disp

                # Compute positions with Z variation too for 3D wave effect
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z_wave = (learn_disp * 0.2) + (stress_disp * 0.15) + (growth_disp * 0.25)
                z = np.full(self.n_points, base_z) + z_wave

                pos = np.column_stack((x, y, z))

                # === COMPUTE COLORS (MORE VIVID) ===
                # Base: cyan/blue gradient
                t = i / (self.n_rings - 1)

                colors = np.zeros((self.n_points, 4))

                for j in range(self.n_points):
                    # Start with base color
                    r_col = 0.1 + 0.15 * t
                    g_col = 0.6 - 0.15 * t
                    b_col = 0.85

                    # Add CYAN for learning - MORE INTENSE
                    if self.learning_wave[i] > 0.05:
                        intensity = min(1.0, self.learning_wave[i])
                        g_col += intensity * 0.5
                        b_col += intensity * 0.3
                        r_col -= intensity * 0.1

                    # Add RED for stress - MUCH MORE VISIBLE
                    if abs(self.stress_wave[i]) > 0.05:
                        intensity = min(1.0, abs(self.stress_wave[i]))
                        r_col += intensity * 1.0
                        g_col -= intensity * 0.4
                        b_col -= intensity * 0.3

                    # Add PURPLE for growth - VIBRANT
                    if self.growth_wave[i] > 0.05:
                        intensity = min(1.0, self.growth_wave[i])
                        r_col += intensity * 0.7
                        b_col += intensity * 0.5
                        g_col -= intensity * 0.2

                    # Brighten all when speaking - GLOWING EFFECT
                    if self.speak_intensity > 0.1:
                        brightness = 1 + self.speak_intensity * 0.8
                        r_col *= brightness
                        g_col *= brightness
                        b_col *= brightness

                    # Alpha based on wave activity - MORE OPAQUE when active
                    wave_activity = abs(total_disp[j]) / 3.0
                    alpha = 0.5 + 0.5 * min(1.0, wave_activity)

                    colors[j] = [
                        min(1.0, r_col),
                        min(1.0, g_col),
                        min(1.0, b_col),
                        min(0.9, alpha)
                    ]

                ring_data['item'].setData(pos=pos, color=colors)

            # Update core glow based on activity - MORE DYNAMIC
            total_activity = (
                np.mean(np.abs(self.learning_wave)) * 2 +
                np.mean(np.abs(self.stress_wave)) * 1.5 +
                np.mean(self.growth_wave) * 2 +
                self.speak_intensity * 2
            )

            core_brightness = 0.4 + min(0.6, total_activity * 0.4)

            # Dynamic core colors based on dominant wave
            base_r = 0.3 + self.speak_intensity * 0.6 + np.mean(np.abs(self.stress_wave)) * 0.5
            base_g = 0.5 + np.mean(self.learning_wave) * 0.5
            base_b = 0.8 + np.mean(self.growth_wave) * 0.2

            self.core_colors[:, 0] = min(1.0, base_r)
            self.core_colors[:, 1] = min(1.0, base_g)
            self.core_colors[:, 2] = min(1.0, base_b)
            self.core_colors[:, 3] = min(0.8, core_brightness)

            # Pulse core when speaking - BIGGER pulse
            if self.speak_intensity > 0.1:
                pulse = 1 + 0.5 * np.sin(self.time * 10)
                self.core.setData(pos=self.core_pos * pulse, color=self.core_colors)
            else:
                self.core.setData(color=self.core_colors)

            # Slow rotation
            self.opts['azimuth'] = self.opts.get('azimuth', 0) + 0.3

        except Exception as e:
            print(f"[ORB] Visualization error: {e}")

    def trigger_growth(self):
        """Call this when cortex growth occurs"""
        self.growth_triggered = True
        self.growth_phase = 0.0
        # Also add more rings for visual growth
        self.base_radius += 0.5


class SciFiPanel(QFrame):
    """Styled panel for UI sections"""

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
            font-family: 'Segoe UI';
            font-weight: bold;
            font-size: 10pt;
            border: none;
            background: none;
        """)
        layout.addWidget(label)

        self.content = layout


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MagnumOpusVitalis: The Living Intelligence")
        self.resize(1600, 900)
        self.setStyleSheet("""
            background-color: #050505;
            font-family: 'Consolas';
            color: #00FF9D;
        """)
        self.showMaximized()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === LEFT: NEURAL CORE ===
        left_panel = SciFiPanel(":: NEURAL CORE ::")
        self.orb = LivingOrb()
        left_panel.content.addWidget(self.orb, 1)

        self.stats_label = QLabel("INITIALIZING...")
        self.stats_label.setStyleSheet("color: #00FF9D; font-size: 11pt; border: none;")
        left_panel.content.addWidget(self.stats_label)

        # Wave legend
        legend = QLabel("◆ CYAN=Learning  ◆ RED=Stress  ◆ PURPLE=Growth  ◆ ALL=Voice")
        legend.setStyleSheet("color: #666; font-size: 8pt; border: none;")
        left_panel.content.addWidget(legend)

        main_layout.addWidget(left_panel, 2)

        # === RIGHT: DATA STREAMS ===
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)

        # Optical Array
        vis_panel = SciFiPanel(":: OPTICAL ARRAY ::")
        h_vis = QHBoxLayout()

        self.screens = []
        for title in ["RETINA", "CORTEX", "DREAM"]:
            v = QLabel()
            v.setFixedSize(180, 180)
            v.setStyleSheet("border: 1px solid #004455; background: #000;")
            v.setScaledContents(True)

            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setStyleSheet("color: #00FFFF; font-size: 9pt;")
            box.addWidget(lbl)
            box.addWidget(v)
            h_vis.addLayout(box)
            self.screens.append(v)

        vis_panel.content.addLayout(h_vis)
        r_layout.addWidget(vis_panel)

        # Telemetry
        tel_panel = SciFiPanel(":: PREFRONTAL TELEMETRY ::")
        self.bars = []
        grid = QGridLayout()

        # 8 bars now: 6 PFC outputs + 2 scaffold indicators
        labels = ["VIS GAIN", "AUD GAIN", "DREAM", "SPEAK", "CRY SUP", "CONSERVE", "LLM YOLK", "TTS GATE"]
        colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#FF3300", "#00FF00", "#00AAFF", "#FF8800", "#88FF00"]

        for i, txt in enumerate(labels):
            l = QLabel(txt)
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            l.setStyleSheet("color: #888; font-size: 9pt;")

            b = QProgressBar()
            b.setRange(0, 100)
            b.setFixedHeight(15)
            b.setStyleSheet(f"""
                QProgressBar {{ background: #111; border: none; }}
                QProgressBar::chunk {{ background: {colors[i]}; }}
            """)

            grid.addWidget(l, i, 0)
            grid.addWidget(b, i, 1)
            self.bars.append(b)

        # Energy bar
        l = QLabel("ENERGY")
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        l.setStyleSheet("color: #888; font-size: 9pt;")
        self.energy_bar = QProgressBar()
        self.energy_bar.setRange(0, 100)
        self.energy_bar.setFixedHeight(15)
        self.energy_bar.setStyleSheet("""
            QProgressBar { background: #111; border: none; }
            QProgressBar::chunk { background: #00AAFF; }
        """)
        grid.addWidget(l, len(labels), 0)
        grid.addWidget(self.energy_bar, len(labels), 1)

        tel_panel.content.addLayout(grid)
        r_layout.addWidget(tel_panel)

        # Chat panels
        split = QHBoxLayout()

        # Ingestion stream
        mat_panel = SciFiPanel(":: KNOWLEDGE INGESTION ::")
        self.matrix_txt = QTextEdit()
        self.matrix_txt.setReadOnly(True)
        self.matrix_txt.setStyleSheet("""
            background: #000500;
            color: #005500;
            border: none;
            font-size: 8pt;
            font-family: 'Courier New';
        """)
        mat_panel.content.addWidget(self.matrix_txt)
        split.addWidget(mat_panel, 1)

        # Communication
        chat_panel = SciFiPanel(":: COMM LINK ::")
        self.chat_txt = QTextEdit()
        self.chat_txt.setReadOnly(True)
        self.chat_txt.setStyleSheet("""
            background: #001122;
            color: #00FF9D;
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

        chat_panel.content.addWidget(self.chat_txt)
        chat_panel.content.addWidget(self.input_field)
        split.addWidget(chat_panel, 2)

        r_layout.addLayout(split)
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

    def _on_update(self, data: dict):
        try:
            # Feed actual data to the orb visualization
            self.orb.set_data(
                loss=data['loss'],
                stress=data['stress'],
                is_speaking=data['speaking'],
                growth_event=False,  # Growth handled separately via signal
                voice_intensity=data['volume'] if data['speaking'] else 0.0
            )

            # Update the visualization
            self.orb.update_visualization()

            # Update stats
            mode = "SPEAKING" if data['speaking'] else "LISTENING"
            mode_color = "#FF3300" if data['speaking'] else "#00FFFF"

            # Show input source
            src = data.get('input_source', 'AMBIENT')
            src_color = "#00FF00" if src == "EXTERNAL" else "#666666"

            self.stats_label.setText(
                f"LAYERS: {data['layers']} | "
                f"VOCAB: {data['vocab_size']} | "
                f"LOSS: {data['loss']:.3f} | "
                f"<span style='color:{mode_color}'>{mode}</span> | "
                f"<span style='color:{src_color}'>IN:{src[:3]}</span> | "
                f"LLM:{int(data.get('llm_reliance', 0.9)*100)}% | "
                f"{data['fps']:.1f} FPS"
            )

            # Update bars (8 PFC outputs)
            actions = data['actions']
            for i, bar in enumerate(self.bars):
                if i < len(actions):
                    bar.setValue(int(min(actions[i], 1.0) * 100))

            self.energy_bar.setValue(int(min(data['energy'], 1.0) * 100))

            # Update screens
            def to_pixmap(arr):
                try:
                    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    img = cv2.resize(img, (180, 180))
                    return QPixmap.fromImage(QImage(img, 180, 180, 180, QImage.Format_Grayscale8))
                except Exception:
                    return QPixmap()

            self.screens[0].setPixmap(to_pixmap(data['real_image']))
            self.screens[1].setPixmap(to_pixmap(data['cortex_image']))
            self.screens[2].setPixmap(to_pixmap(data['dream_image']))

            # Update chat
            if data.get('ai_msg'):
                self.chat_txt.append(f"<span style='color:#00FF00'>AI: {data['ai_msg']}</span>")

            if data.get('pac_msg'):
                self.matrix_txt.append(f"<span style='color:#004400'>{data['pac_msg']}</span>")

        except Exception as e:
            print(f"[UI] Update error: {e}")

    def _on_growth(self, layers: int):
        self.orb.trigger_growth()
        self.chat_txt.append(
            "<b style='color:#FF3300'>*** CORTICAL EXPANSION DETECTED ***</b>"
        )

    def _on_pacman(self, msg: str):
        self.matrix_txt.append(f"<span style='color:#00AA00'>{msg}</span>")

    def closeEvent(self, event):
        self.worker.running = False
        self.worker.wait()
        event.accept()


# ============================================================================
# 🚀 SECTION 7: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           MagnumOpusVitalis: The Living Intelligence          ║
    ║                                                               ║
    ║  "A seed that grows, not a machine that thinks."              ║
    ║                                                               ║
    ║  ARCHITECT: Alan Hourmand                                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())