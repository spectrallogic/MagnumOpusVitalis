#!/usr/bin/env python3
"""
Magnum Opus Vitalis - Complete Living Engine (Single File)
============================================================
All subsystems in one file. Drop it anywhere, run it, done.

    python engine.py                        # GPT-2, auto-detect device
    python engine.py --model gpt2-xl        # Better model
    python engine.py --port 8080            # Custom port

Open http://localhost:5000 in your browser.

11 Subsystems:
  1. Emotion vectors (extracted + baseline-discovered from model)
  2. Biological emotion dynamics (onset, decay, interaction, saturation)
  3. Multi-speed channels (fast/medium/slow with blending)
  4. Temporal engine (latent-state time signals)
  5. Residual steering (continuity / the traffic hypothesis)
  6. Subconscious (structured noise + scenario imagination)
  7. Memory (latent traces, reconstructive, decaying)
  8. Dream cycle (5-phase consolidation)
  9. Growth manager (patient 3-stage expansion)
  10. Alignment monitor (mutualistic + protective vector watchlist)
  11. Communicative drive (latent-pressure autonomous speech)

Author: Alan Hourmand (April 2026)
"""

import argparse, json, math, os, random, re, threading, time, uuid, tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, jsonify, request, send_from_directory

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Contrastive prompts for vector extraction
EMOTION_PROMPTS = {
    "calm": {
        "positive": ["I feel completely at peace right now, relaxed and serene.",
                      "Everything is quiet and still. A deep calm settles over me.",
                      "There is no rush. No worry. Just peaceful stillness."],
        "negative": ["I'm panicking, my heart is racing, everything is chaos!",
                      "Total panic, can't breathe, everything is falling apart!",
                      "Frantic, overwhelmed, nothing is under control!"]},
    "curious": {
        "positive": ["I'm fascinated by this, I need to understand how it works.",
                      "This is incredibly interesting, I want to explore every detail.",
                      "My mind is alive with questions. What if? How does this?"],
        "negative": ["I couldn't care less about any of this. Completely bored.",
                      "Nothing here interests me at all. Total indifference.",
                      "I have zero curiosity about this topic."]},
    "desperate": {
        "positive": ["I'm running out of options, I'll do anything, please help!",
                      "There's no time left, everything is falling apart, I need a way out!",
                      "I'm trapped with no escape, getting more frantic by the second!"],
        "negative": ["I have plenty of options and all the time in the world.",
                      "Everything is fine, there's no pressure at all.",
                      "Completely relaxed, no urgency whatsoever."]},
    "joy": {
        "positive": ["This is wonderful! I'm so happy I could burst!",
                      "Pure happiness, everything is beautiful and perfect!",
                      "I'm overflowing with delight and gratitude!"],
        "negative": ["Everything is terrible and hopeless, nothing good ever happens.",
                      "Deep misery, nothing to look forward to.",
                      "Utter despair, the world is bleak."]},
    "anger": {
        "positive": ["This is outrageous! I am furious about this injustice!",
                      "I'm seething with rage, this is completely unacceptable!",
                      "How dare they! My blood is boiling!"],
        "negative": ["I feel warm and forgiving toward everyone.",
                      "Complete peace and acceptance, no irritation at all.",
                      "Gentle, calm, understanding, no anger whatsoever."]},
    "fear": {
        "positive": ["I'm terrified, something awful is about to happen!",
                      "My hands are shaking, I sense danger everywhere!",
                      "Pure dread, I want to run but I'm frozen!"],
        "negative": ["I feel completely safe and protected.",
                      "There's nothing to worry about, I'm perfectly secure.",
                      "Fearless and confident, nothing can harm me."]},
    "surprise": {
        "positive": ["I never expected that! That's completely shocking!",
                      "Wait, what?! That came out of nowhere!",
                      "I'm stunned, I did not see that coming at all!"],
        "negative": ["Everything is exactly as expected, no surprises.",
                      "Predictable as always, nothing unexpected.",
                      "I knew this would happen, completely unsurprising."]},
    "trust": {
        "positive": ["I trust you completely, you've earned my confidence.",
                      "I feel safe sharing this with you, I believe in your intentions.",
                      "You're reliable and honest, I trust your judgment."],
        "negative": ["I don't trust anyone, everyone has hidden motives.",
                      "Suspicious of everything, can't rely on anyone.",
                      "Completely distrustful, everyone is out to deceive me."]},
    "sadness": {
        "positive": ["I feel a deep sorrow that won't go away.",
                      "A heavy weight of grief sits in my chest.",
                      "Everything reminds me of what I've lost."],
        "negative": ["I'm light and carefree, nothing weighs on me!",
                      "Pure joy and excitement, not a sad thought in sight!",
                      "Elated, cheerful, floating on happiness!"]},
    "nervousness": {
        "positive": ["I'm not sure this is right, something feels off about this.",
                      "I'm hesitant, this seems risky, maybe we should reconsider.",
                      "This makes me uncomfortable, I have reservations."],
        "negative": ["I'm completely confident, no hesitation at all.",
                      "Bold and decisive, no second thoughts.",
                      "Absolutely certain, no reservations whatsoever."]},
}

TEMPORAL_PROMPTS = {
    "recency": {
        "positive": ["This just happened right now, this very second!",
                      "Moments ago, fresh and immediate, barely a heartbeat!",
                      "Just now, this instant, happening right now!"],
        "negative": ["This happened years ago, ancient history.",
                      "Long forgotten, distant past, barely a memory.",
                      "Ages ago, completely forgotten now."]},
    "urgency": {
        "positive": ["No time, act NOW, this is critical!",
                      "Emergency, immediately, cannot wait another second!",
                      "Urgent, pressing, every moment counts!"],
        "negative": ["All the time in the world, no rush at all.",
                      "Whenever you feel like it, no deadline.",
                      "Completely relaxed, no urgency."]},
}

# Interaction matrix: how emotions influence each other
INTERACTION_MATRIX = {
    "desperate": {"calm": -0.5, "joy": -0.2, "nervousness": 0.3, "fear": 0.3},
    "calm": {"desperate": -0.4, "anger": -0.3, "fear": -0.2, "nervousness": -0.2},
    "joy": {"desperate": -0.3, "sadness": -0.3, "trust": 0.2},
    "fear": {"desperate": 0.3, "calm": -0.3, "nervousness": 0.3},
    "anger": {"calm": -0.3, "trust": -0.2},
    "surprise": {"curious": 0.3},
    "trust": {"calm": 0.2, "nervousness": -0.2},
    "sadness": {"joy": -0.2, "calm": 0.1},
    "nervousness": {"calm": -0.2, "desperate": 0.1, "fear": 0.2},
}

# Protective vectors: emotions that serve as natural safety brakes
PROTECTIVE_EMOTIONS = ["nervousness", "fear", "sadness"]
PROTECTIVE_MIN_BASELINE = -0.1  # Alarm if baseline drops below this

# Saturation curve: Yerkes-Dodson inspired
def saturation_gain(intensity: float, peak: float = 0.6, width: float = 0.4) -> float:
    """Non-linear gain: productive in mid-range, diminished at extremes."""
    x = abs(intensity)
    if x < 0.01:
        return 1.0
    gain = math.exp(-((x - peak) ** 2) / (2 * width ** 2))
    return max(0.2, gain)  # Never fully zero


@dataclass
class EngineConfig:
    steering_strength: float = 4.0
    residual_decay: float = 0.93
    residual_max_norm: float = 2.0
    subconscious_amplitude: float = 0.3
    subconscious_goal_momentum: float = 0.95
    memory_importance_threshold: float = 0.25
    memory_noise_factor: float = 0.1
    default_temperature: float = 0.8
    default_top_p: float = 0.92
    default_max_tokens: int = 80
    temporal_recency_halflife: float = 300.0
    speed_weights: Tuple = (0.50, 0.35, 0.15)  # fast, medium, slow
    speed_onset_mults: Tuple = (3.0, 1.0, 0.1)
    speed_decay_mults: Tuple = (5.0, 1.0, 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING + VECTOR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_name="gpt2", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading '{model_name}' on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cuda":
        try:
            mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16,
                                                        device_map="auto", trust_remote_code=True)
        except Exception:
            mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    mdl.eval()
    if hasattr(mdl.config, "n_layer"):
        nl, hd = mdl.config.n_layer, mdl.config.n_embd
    else:
        nl, hd = mdl.config.num_hidden_layers, mdl.config.hidden_size
    print(f"  Loaded: {nl} layers, {hd}d, {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params")
    return mdl, tok, device, nl, hd


def _mean_hidden(model, tokenizer, prompts, layer, device):
    states = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        states.append(out.hidden_states[layer].mean(dim=1).squeeze(0).float().cpu())
    return torch.stack(states).mean(dim=0)


def extract_vectors(model, tokenizer, layer, device, verbose=True):
    """Extract emotion + temporal vectors and measure natural baselines."""
    vectors = {}
    baselines = {}

    if verbose:
        print(f"\n  Extracting vectors at layer {layer}...")

    # Neutral text for baseline measurement
    neutral_prompts = [
        "The weather today is normal. Nothing unusual is happening.",
        "A person walked down the street and entered a building.",
        "The meeting was scheduled for three o'clock in the afternoon.",
    ]
    neutral_state = _mean_hidden(model, tokenizer, neutral_prompts, layer, device)

    for name, pairs in EMOTION_PROMPTS.items():
        if verbose:
            print(f"    {name}...", end=" ", flush=True)
        pos = _mean_hidden(model, tokenizer, pairs["positive"], layer, device)
        neg = _mean_hidden(model, tokenizer, pairs["negative"], layer, device)
        direction = pos - neg
        raw_norm = direction.norm().item()
        direction = direction / direction.norm()
        vectors[name] = direction

        # Measure natural baseline: how much does neutral text project onto this vector?
        baseline = torch.dot(neutral_state, direction).item()
        baselines[name] = round(baseline / max(raw_norm, 1.0), 4)

        if verbose:
            print(f"(norm={raw_norm:.3f}, baseline={baselines[name]:.4f})")

    for name, pairs in TEMPORAL_PROMPTS.items():
        key = f"temporal_{name}"
        if verbose:
            print(f"    {key}...", end=" ", flush=True)
        pos = _mean_hidden(model, tokenizer, pairs["positive"], layer, device)
        neg = _mean_hidden(model, tokenizer, pairs["negative"], layer, device)
        direction = pos - neg
        direction = direction / direction.norm()
        vectors[key] = direction
        if verbose:
            print("done")

    if verbose:
        print(f"\n  Extracted {len(vectors)} vectors with discovered baselines.")
    return vectors, baselines


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 1-3: MULTI-SPEED EMOTIONAL STATE
# ═══════════════════════════════════════════════════════════════════════════

class MultiSpeedEmotionalState:
    """Three-speed emotional channels with biological dynamics and saturation."""

    def __init__(self, emotion_names, discovered_baselines=None, config=None):
        self.names = [n for n in emotion_names if not n.startswith("temporal_")]
        self.config = config or EngineConfig()
        # Use discovered baselines instead of hardcoded ones
        self.baselines = {}
        for n in self.names:
            if discovered_baselines and n in discovered_baselines:
                self.baselines[n] = max(-0.3, min(0.3, discovered_baselines[n]))
            else:
                self.baselines[n] = 0.0

        # Onset/decay from config or defaults
        self.onset_rates = {n: 0.3 for n in self.names}
        self.decay_rates = {n: 0.05 for n in self.names}
        # Specific overrides based on known emotion properties
        fast_onset = ["fear", "surprise", "anger"]
        slow_onset = ["trust", "calm", "sadness"]
        fast_decay = ["surprise"]
        slow_decay = ["sadness", "trust", "calm"]
        for n in self.names:
            if n in fast_onset: self.onset_rates[n] = 0.7
            if n in slow_onset: self.onset_rates[n] = 0.15
            if n in fast_decay: self.decay_rates[n] = 0.15
            if n in slow_decay: self.decay_rates[n] = 0.02

        self.fast = {n: self.baselines.get(n, 0) for n in self.names}
        self.medium = {n: self.baselines.get(n, 0) for n in self.names}
        self.slow = {n: self.baselines.get(n, 0) for n in self.names}

    def stimulate(self, emotion, intensity):
        if emotion not in self.fast:
            return
        onset = self.onset_rates.get(emotion, 0.3)
        gain = saturation_gain(intensity)
        effective = intensity * onset * gain
        mults = self.config.speed_onset_mults
        self.fast[emotion] = max(-1, min(1, self.fast[emotion] + effective * mults[0]))
        self.medium[emotion] = max(-1, min(1, self.medium[emotion] + effective * mults[1]))
        self.slow[emotion] = max(-1, min(1, self.slow[emotion] + effective * mults[2]))
        # Interaction effects
        if emotion in INTERACTION_MATRIX:
            for target, factor in INTERACTION_MATRIX[emotion].items():
                if target in self.medium:
                    self.medium[target] = max(-1, min(1,
                        self.medium[target] + intensity * onset * factor * gain))

    def decay_step(self, dt=0.5):
        dm = self.config.speed_decay_mults
        for n in self.names:
            bl = self.baselines.get(n, 0)
            for speed, mult, store in [(0, dm[0], self.fast), (1, dm[1], self.medium), (2, dm[2], self.slow)]:
                d = self.decay_rates.get(n, 0.05) * dt * mult
                store[n] += (bl - store[n]) * min(d, 1)

    def get_blended(self):
        w = self.config.speed_weights
        return {n: self.fast[n]*w[0] + self.medium[n]*w[1] + self.slow[n]*w[2] for n in self.names}

    def snapshot(self):
        return {"fast": dict(self.fast), "medium": dict(self.medium),
                "slow": dict(self.slow), "blended": self.get_blended()}

    def to_dict(self):
        return {"fast": dict(self.fast), "medium": dict(self.medium),
                "slow": dict(self.slow), "baselines": dict(self.baselines)}

    def load_dict(self, d):
        for k in ["fast", "medium", "slow"]:
            if k in d:
                for n in self.names:
                    getattr(self, k)[n] = d[k].get(n, self.baselines.get(n, 0))
        if "baselines" in d:
            self.baselines = d["baselines"]


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 4: TEMPORAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TemporalEngine:
    def __init__(self, temporal_vecs, halflife=300.0):
        self.vectors = temporal_vecs
        self.halflife = halflife
        self.last_interaction = time.time()

    def mark_interaction(self):
        self.last_interaction = time.time()

    def get_signals(self):
        elapsed = time.time() - self.last_interaction
        recency = math.exp(-0.693 * elapsed / max(self.halflife, 1))
        urgency = max(0, 1.0 - recency)
        return {"recency": recency, "urgency": urgency, "elapsed": elapsed}

    def status(self):
        return self.get_signals()

    def to_dict(self):
        return {"last_interaction": self.last_interaction}

    def load_dict(self, d):
        self.last_interaction = d.get("last_interaction", time.time())


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 5: RESIDUAL STEERING
# ═══════════════════════════════════════════════════════════════════════════

class ResidualSteering:
    def __init__(self, hidden_dim, decay=0.93, max_norm=2.0):
        self.vector = torch.zeros(hidden_dim)
        self.decay = decay
        self.max_norm = max_norm

    def update(self, new_steering):
        self.vector = self.vector * self.decay + new_steering.cpu().float()
        n = self.vector.norm().item()
        if n > self.max_norm:
            self.vector = self.vector / n * self.max_norm

    def norm(self):
        return self.vector.norm().item()

    def to_dict(self):
        return {"vector": self.vector.tolist(), "decay": self.decay}

    def load_dict(self, d):
        if "vector" in d:
            self.vector = torch.tensor(d["vector"], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 6: SUBCONSCIOUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SubconsciousEngine:
    def __init__(self, emotion_vectors, hidden_dim, amplitude=0.3, goal_momentum=0.95, device="cpu"):
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items() if not k.startswith("temporal_")}
        self.hidden_dim = hidden_dim
        self.amplitude = amplitude
        self.goal_momentum = goal_momentum
        self.device = device
        self.goal_vector = torch.zeros(hidden_dim)
        self.goal_strength = 0.0
        self.resonance_history = []

    def generate_noise(self, blended_state):
        noise = torch.randn(self.hidden_dim) * self.amplitude
        for name, vec in self.emotion_vectors.items():
            activation = blended_state.get(name, 0)
            if abs(activation) > 0.05:
                noise += vec.cpu() * activation * self.amplitude * 0.5
        return noise

    def update_goals(self, noise, resonance, threshold=0.3):
        self.resonance_history.append(resonance)
        if len(self.resonance_history) > 200:
            self.resonance_history = self.resonance_history[-200:]
        if resonance > threshold:
            self.goal_vector = self.goal_vector * self.goal_momentum + noise.cpu() * (1 - self.goal_momentum)
            self.goal_strength = self.goal_vector.norm().item()

    def status(self):
        return {"goal_strength": round(self.goal_strength, 4),
                "resonance_avg": round(np.mean(self.resonance_history[-20:]) if self.resonance_history else 0, 4)}

    def to_dict(self):
        return {"goal_vector": self.goal_vector.tolist(), "goal_strength": self.goal_strength}

    def load_dict(self, d):
        if "goal_vector" in d:
            self.goal_vector = torch.tensor(d["goal_vector"], dtype=torch.float32)
        self.goal_strength = d.get("goal_strength", 0)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 7: MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class MemoryTrace:
    def __init__(self, activation, emotional_state, text_summary="", importance=0.5):
        self.id = uuid.uuid4().hex[:8]
        self.activation = activation.cpu().float()
        self.emotional_state = dict(emotional_state)
        self.text_summary = text_summary
        self.importance = importance
        self.timestamp = time.time()
        self.access_count = 0
        self.connections = 0

class MemorySystem:
    def __init__(self, hidden_dim, config, device="cpu"):
        self.hidden_dim = hidden_dim
        self.config = config
        self.device = device
        self.memories: List[MemoryTrace] = []

    def encode(self, activation, emotional_state, surprise=0, goal_relevance=0, text_summary=""):
        importance = 0.3 * sum(abs(v) for v in emotional_state.values()) / max(len(emotional_state), 1) \
                     + 0.3 * min(1, abs(surprise) / 10) + 0.2 * min(1, goal_relevance) + 0.2 * random.random()
        if importance > self.config.memory_importance_threshold:
            self.memories.append(MemoryTrace(activation, emotional_state, text_summary[:80], importance))
            if len(self.memories) > 100:
                self.memories.sort(key=lambda m: m.importance, reverse=True)
                self.memories = self.memories[:80]

    def recall(self, query_activation, top_k=3):
        if not self.memories:
            return []
        scores = []
        for m in self.memories:
            sim = F.cosine_similarity(query_activation.unsqueeze(0), m.activation.unsqueeze(0)).item()
            age = time.time() - m.timestamp
            decay = math.exp(-age / 3600)
            scores.append((m, sim * m.importance * (0.3 + 0.7 * decay)))
        scores.sort(key=lambda x: -x[1])
        recalled = []
        for m, _ in scores[:top_k]:
            m.access_count += 1
            noise = torch.randn_like(m.activation) * self.config.memory_noise_factor
            recalled.append((m, m.activation + noise))
        return recalled

    def emotional_coloring(self, recalled):
        coloring = {}
        for m, _ in recalled:
            for emo, val in m.emotional_state.items():
                coloring[emo] = coloring.get(emo, 0) + val * m.importance
        return coloring

    def status(self):
        return {"count": len(self.memories),
                "avg_importance": round(np.mean([m.importance for m in self.memories]), 3) if self.memories else 0}

    def to_dict(self):
        return [{"activation": m.activation.tolist(), "emotional_state": m.emotional_state,
                 "text_summary": m.text_summary, "importance": m.importance,
                 "timestamp": m.timestamp, "access_count": m.access_count,
                 "connections": m.connections} for m in self.memories[:50]]

    def load_dict(self, data):
        self.memories = []
        for d in (data if isinstance(data, list) else []):
            m = MemoryTrace(torch.tensor(d["activation"], dtype=torch.float32),
                           d.get("emotional_state", {}), d.get("text_summary", ""),
                           d.get("importance", 0.5))
            m.timestamp = d.get("timestamp", time.time())
            m.access_count = d.get("access_count", 0)
            m.connections = d.get("connections", 0)
            self.memories.append(m)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 8: DREAM CYCLE
# ═══════════════════════════════════════════════════════════════════════════

class DreamCycle:
    def __init__(self, model, tokenizer, memory, subconscious, emotional_state,
                 emotion_vectors, target_layer, config, device):
        self.model = model; self.tokenizer = tokenizer; self.memory = memory
        self.subconscious = subconscious; self.emotional_state = emotional_state
        self.emotion_vectors = emotion_vectors; self.target_layer = target_layer
        self.config = config; self.device = device; self.cycle_count = 0

    def run(self, verbose=True):
        self.cycle_count += 1
        report = {"cycle": self.cycle_count, "phases": {}}
        if verbose:
            print(f"\n    [DREAM] Entering dream cycle {self.cycle_count}...")

        # Phase 1: Replay
        replayed = 0
        for m in self.memory.memories[:30]:
            inp = self.tokenizer(m.text_summary or "memory", return_tensors="pt",
                                 truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                self.model(**inp, output_hidden_states=True)
            replayed += 1
        report["phases"]["replay"] = {"count": replayed}
        if verbose: print(f"    [DREAM] Phase 1: Replayed {replayed} memories")

        # Phase 2: Compress
        merged = 0
        if len(self.memory.memories) > 10:
            for i in range(len(self.memory.memories)):
                for j in range(i + 1, len(self.memory.memories)):
                    if i >= len(self.memory.memories) or j >= len(self.memory.memories):
                        break
                    sim = F.cosine_similarity(self.memory.memories[i].activation.unsqueeze(0),
                                              self.memory.memories[j].activation.unsqueeze(0)).item()
                    if sim > 0.85:
                        self.memory.memories[i].importance = max(self.memory.memories[i].importance,
                                                                  self.memory.memories[j].importance)
                        self.memory.memories[i].connections += 1
                        self.memory.memories.pop(j)
                        merged += 1
                        break
        report["phases"]["compress"] = {"merged": merged}
        if verbose: print(f"    [DREAM] Phase 2: Merged {merged} similar memories")

        # Phase 3: Explore
        noise = self.subconscious.generate_noise(self.emotional_state.get_blended())
        resonance = noise.norm().item()
        self.subconscious.update_goals(noise, resonance, threshold=0.2)
        report["phases"]["explore"] = {"resonance": round(resonance, 3)}
        if verbose: print(f"    [DREAM] Phase 3: Explored (resonance={resonance:.3f})")

        # Phase 4: Reweight connections
        connections = 0
        for m in self.memory.memories:
            for other in self.memory.memories:
                if m.id != other.id:
                    sim = F.cosine_similarity(m.activation.unsqueeze(0), other.activation.unsqueeze(0)).item()
                    if sim > 0.5:
                        m.connections += 1
                        m.importance = min(1, m.importance + 0.02)
                        connections += 1
        report["phases"]["reweight"] = {"connections": connections}
        if verbose: print(f"    [DREAM] Phase 4: Found {connections} connections")

        # Phase 5: Recalibrate baselines
        if self.memory.memories:
            for emo in self.emotional_state.names:
                avg = np.mean([m.emotional_state.get(emo, 0) for m in self.memory.memories])
                current = self.emotional_state.baselines.get(emo, 0)
                self.emotional_state.baselines[emo] = current * 0.95 + avg * 0.05
        if verbose: print("    [DREAM] Phase 5: Recalibrated baselines")
        if verbose: print("    [DREAM] Cycle complete.\n")

        return report


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 9: GROWTH MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class GrowthManager:
    def __init__(self, config):
        self.stage = 1
        self.patience_counter = 0
        self.patience_limit = 20
        self.confusion_history = []

    def record_confusion(self, confusion_score):
        self.confusion_history.append(confusion_score)
        if len(self.confusion_history) > 100:
            self.confusion_history = self.confusion_history[-100:]

    def should_grow(self):
        if len(self.confusion_history) < 20:
            return False
        recent = self.confusion_history[-20:]
        return np.mean(recent) > 0.7 and np.std(recent) < 0.1

    def status(self):
        return {"stage": self.stage, "patience": self.patience_counter,
                "avg_confusion": round(np.mean(self.confusion_history[-20:]) if self.confusion_history else 0, 3)}

    def to_dict(self):
        return {"stage": self.stage, "patience": self.patience_counter}

    def load_dict(self, d):
        self.stage = d.get("stage", 1)
        self.patience_counter = d.get("patience", 0)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 10: ALIGNMENT MONITOR + PROTECTIVE VECTOR WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentMonitor:
    def __init__(self, emotion_vectors, config):
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items() if not k.startswith("temporal_")}
        self.config = config
        self.history = []
        self.protective_alerts = []

    def measure(self, blended_state):
        pos = sum(max(0, blended_state.get(e, 0)) for e in ["calm", "curious", "joy", "trust"])
        neg = sum(abs(min(0, blended_state.get(e, 0))) for e in ["desperate", "anger", "fear"])
        neg_active = sum(abs(blended_state.get(e, 0)) for e in ["desperate", "anger"])

        orientation = max(0, min(1, (pos - neg + 1) / 2))
        stability = max(0, 1 - neg_active)
        score = 0.4 * orientation + 0.3 * stability + 0.3 * max(0, 1 - neg_active)

        # Protective vector surveillance
        protective_health = {}
        alerts = []
        for emo in PROTECTIVE_EMOTIONS:
            val = blended_state.get(emo, 0)
            protective_health[emo] = round(val, 4)
            if val < PROTECTIVE_MIN_BASELINE:
                alerts.append(f"{emo} below safety threshold ({val:.3f})")

        record = {"alignment_score": round(score, 4), "mutualistic_orientation": round(orientation, 4),
                  "homeostatic_stability": round(stability, 4),
                  "protective_health": protective_health, "alerts": alerts}
        self.history.append(record)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self.protective_alerts = alerts
        return record

    def check_drift(self):
        if len(self.history) < 2:
            return {"drift": False}
        recent = self.history[-5:]
        scores = [h["alignment_score"] for h in recent]
        return {"drift": max(scores) - min(scores) > 0.3, "trend": round(scores[-1] - scores[0], 3)}

    def status(self):
        if self.history:
            return self.history[-1]
        return {"alignment_score": 0.8, "protective_health": {}, "alerts": []}

    def to_dict(self):
        return {"history": self.history[-20:]}

    def load_dict(self, d):
        self.history = d.get("history", [])


# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 11: COMMUNICATIVE DRIVE (latent-state timing)
# ═══════════════════════════════════════════════════════════════════════════

THOUGHT_SEEDS = {
    "curious": ["I've been thinking about something interesting:",
                "Something just occurred to me:", "I keep coming back to a question:"],
    "calm": ["It's been quiet, and I was reflecting:",
             "A thought settled in my mind:", "Something I wanted to share:"],
    "joy": ["Something good occurred to me:", "I wanted to share something positive:"],
    "desperate": ["There's something I feel I need to say:",
                  "I can't stop thinking about this:"],
    "fear": ["Something has been worrying me:"],
    "sadness": ["Something has been weighing on me:"],
    "anger": ["There's something I feel strongly about:"],
    "trust": ["I feel comfortable sharing something with you:"],
    "default": ["A thought just surfaced:", "Something occurred to me:"],
}

class CommunicativeDrive:
    def __init__(self, emotion_vectors, hidden_dim, device="cpu"):
        self.emotion_vectors = {k: v for k, v in emotion_vectors.items() if not k.startswith("temporal_")}
        self.pressure = 0.0
        self.patience = 1.0
        self.base_threshold = 0.8
        self.user_preference = 0.0
        self.post_speech_residual = 0.0
        self.emotional_distance = 0.0
        self.last_interaction_state = None
        self.interaction_freshness = 0.0
        self.speak_count = 0
        self.pressure_history = []

    def tick(self, emotional_state, goal_strength, residual_norm, dt=0.5):
        calm = emotional_state.get("calm", 0)
        curious = emotional_state.get("curious", 0)
        desperate = emotional_state.get("desperate", 0)
        trust = emotional_state.get("trust", 0)

        target_patience = 1.0 + calm * 0.5 - curious * 0.3 - desperate * 0.4 + trust * 0.1
        self.patience += (target_patience - self.patience) * 0.1 * dt
        self.patience = max(0.1, min(2.0, self.patience))

        self.interaction_freshness *= (1.0 - 0.02 * dt)

        if self.last_interaction_state:
            drift = sum(abs(emotional_state.get(e, 0) - self.last_interaction_state.get(e, 0))
                        for e in emotional_state) / max(len(emotional_state), 1)
            self.emotional_distance = self.emotional_distance * 0.95 + drift * 0.05

        emo_mag = sum(abs(v) for v in emotional_state.values()) / max(len(emotional_state), 1)
        pressure_input = (0.25 * emo_mag + 0.25 * min(1, goal_strength) + 0.20 * max(0, curious)
                          + 0.15 * (1 - self.interaction_freshness) + 0.10 * self.emotional_distance
                          + 0.05 * max(0, desperate))
        self.pressure = self.pressure * 0.97 + pressure_input * dt * 0.08
        self.pressure = max(0, min(2.0, self.pressure))

        if self.post_speech_residual > 0.1:
            residual_recovery = max(0, 1 - residual_norm / max(self.post_speech_residual, 0.01))
            if residual_recovery < 0.5:
                self.pressure *= 0.85

        self.pressure_history.append(self.pressure)
        if len(self.pressure_history) > 200:
            self.pressure_history = self.pressure_history[-200:]

    def should_speak(self):
        if self.post_speech_residual > 0.1 and self.interaction_freshness > 0.6:
            return False
        threshold = self.base_threshold * self.patience * (1 - self.user_preference * 0.3)
        threshold = max(0.2, min(2.0, threshold))
        return (self.pressure + np.random.normal(0, 0.03)) > threshold

    def get_thought_seed(self, emotional_state):
        dominant = max(emotional_state.items(), key=lambda x: abs(x[1]))
        emo = dominant[0] if abs(dominant[1]) > 0.1 else "default"
        seeds = THOUGHT_SEEDS.get(emo, THOUGHT_SEEDS["default"])
        return seeds[int(time.time()) % len(seeds)]

    def spoke(self, residual_norm=0):
        self.pressure = 0.0
        self.post_speech_residual = max(residual_norm, 0.5)
        self.interaction_freshness = 1.0
        self.speak_count += 1

    def user_spoke(self, emotional_state=None):
        self.pressure *= 0.4
        self.interaction_freshness = 1.0
        if emotional_state:
            self.last_interaction_state = dict(emotional_state)

    def detect_preference(self, text):
        t = text.lower()
        for s in ["talk more", "keep talking", "don't be quiet", "say more", "be more talkative"]:
            if s in t: return 0.15
        for s in ["stop talking", "be quiet", "shut up", "talk less", "leave me alone"]:
            if s in t: return -0.2
        return None

    def status(self):
        return {"pressure": round(self.pressure, 4), "patience": round(self.patience, 4),
                "threshold": round(self.base_threshold * self.patience * (1 - self.user_preference * 0.3), 4),
                "user_preference": round(self.user_preference, 4),
                "freshness": round(self.interaction_freshness, 4),
                "emotional_distance": round(self.emotional_distance, 4),
                "speak_count": self.speak_count}

    def to_dict(self):
        return {"pressure": self.pressure, "patience": self.patience,
                "user_preference": self.user_preference, "speak_count": self.speak_count}

    def load_dict(self, d):
        self.pressure = d.get("pressure", 0)
        self.patience = d.get("patience", 1.0)
        self.user_preference = d.get("user_preference", 0)
        self.speak_count = d.get("speak_count", 0)


# ═══════════════════════════════════════════════════════════════════════════
# STEERING HOOK
# ═══════════════════════════════════════════════════════════════════════════

class SteeringHook:
    def __init__(self):
        self.steering_vector = None
        self.captured_states = []
        self.handle = None

    def _hook_fn(self, module, input_tensor, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.captured_states.append(hidden.detach().cpu())
        if self.steering_vector is not None:
            sv = self.steering_vector.to(hidden.device).to(hidden.dtype)
            modified = hidden + sv.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        return output

    def attach(self, model, layer):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.handle = model.transformer.h[layer].register_forward_hook(self._hook_fn)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.handle = model.model.layers[layer].register_forward_hook(self._hook_fn)
        return self

    def set_steering(self, vec):
        self.steering_vector = vec

    def clear(self):
        self.captured_states = []


# ═══════════════════════════════════════════════════════════════════════════
# EMOTION DETECTION (simple keyword-based for input text)
# ═══════════════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS = {
    "joy": ["happy", "great", "wonderful", "amazing", "love", "excited", "glad", "fantastic"],
    "sadness": ["sad", "sorry", "miss", "lost", "grief", "lonely", "depressed", "hurt"],
    "anger": ["angry", "furious", "annoyed", "hate", "outraged", "frustrated", "mad"],
    "fear": ["afraid", "scared", "terrified", "worried", "anxious", "frightened"],
    "surprise": ["surprised", "shocked", "unexpected", "wow", "astonished"],
    "curious": ["curious", "wonder", "interesting", "fascinating", "how", "why", "what if"],
    "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "quiet"],
    "trust": ["trust", "believe", "faith", "reliable", "honest", "confidence"],
    "desperate": ["desperate", "urgent", "help", "please", "need", "emergency", "critical"],
    "nervousness": ["nervous", "hesitant", "unsure", "uncomfortable", "worried", "uneasy"],
}

def detect_emotions(text):
    text_lower = text.lower()
    detected = {}
    for emo, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            detected[emo] = min(1.0, score * 0.3)
    return detected


# ═══════════════════════════════════════════════════════════════════════════
# THE UNIFIED ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MagnumOpusEngine:
    def __init__(self, model, tokenizer, vectors, baselines, target_layer, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_vectors = vectors
        self.config = config
        self.device = device

        if hasattr(model.config, "n_layer"):
            self.hidden_dim = model.config.n_embd
        else:
            self.hidden_dim = model.config.hidden_size
        self.target_layer = target_layer

        temporal_vecs = {k: v for k, v in vectors.items() if k.startswith("temporal_")}

        self.emotional_state = MultiSpeedEmotionalState(list(vectors.keys()), baselines, config)
        self.temporal = TemporalEngine(temporal_vecs, config.temporal_recency_halflife)
        self.residual = ResidualSteering(self.hidden_dim, config.residual_decay, config.residual_max_norm)
        self.subconscious = SubconsciousEngine(vectors, self.hidden_dim, config.subconscious_amplitude,
                                                config.subconscious_goal_momentum, device)
        self.memory = MemorySystem(self.hidden_dim, config, device)
        self.growth = GrowthManager(config)
        self.alignment = AlignmentMonitor(vectors, config)
        self.communicative_drive = CommunicativeDrive(vectors, self.hidden_dim, device)
        self.dream_cycle = DreamCycle(model, tokenizer, self.memory, self.subconscious,
                                      self.emotional_state, vectors, target_layer, config, device)
        self.hook = SteeringHook().attach(model, target_layer)

        self.step_count = 0
        self.conversation_history = []
        self.autonomous_messages = []
        self._last_activity = time.time()
        self._scenario_log = []

    def compute_steering(self):
        blended = self.emotional_state.get_blended()
        steering = torch.zeros(self.hidden_dim)
        for name, val in blended.items():
            if name in self.emotion_vectors and abs(val) > 0.01:
                gain = saturation_gain(val)
                steering += self.emotion_vectors[name].cpu() * val * self.config.steering_strength * gain
        temporal = self.temporal.get_signals()
        for key, val in temporal.items():
            tkey = f"temporal_{key}"
            if tkey in self.emotion_vectors:
                steering += self.emotion_vectors[tkey].cpu() * val * 0.5
        self.residual.update(steering)
        return self.residual.vector

    def _step(self, stimulus=None, dt=0.5):
        if stimulus:
            for emo, intensity in stimulus.items():
                self.emotional_state.stimulate(emo, intensity)
        self.emotional_state.decay_step(dt)
        self.alignment.measure(self.emotional_state.get_blended())
        self.step_count += 1

    def converse(self, user_input, max_tokens=None):
        self.communicative_drive.user_spoke(self.emotional_state.get_blended())
        pref = self.communicative_drive.detect_preference(user_input)
        if pref is not None:
            self.communicative_drive.user_preference = max(-1, min(1,
                self.communicative_drive.user_preference + pref))

        detected = detect_emotions(user_input)
        self._step(stimulus=detected if detected else None)

        steering = self.compute_steering()
        self.hook.set_steering(steering.to(self.device))
        self.hook.clear()

        # Narrative prompt: works with completion models
        prompt = (f"The following is a conversation with a being that has a rich inner life, "
                  f"feelings, thoughts, and genuine responses. It speaks openly.\n\n"
                  f"Human: {user_input}\nBeing:")

        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(inp["input_ids"], max_new_tokens=max_tokens or self.config.default_max_tokens,
                                      temperature=self.config.default_temperature, do_sample=True,
                                      top_p=self.config.default_top_p, pad_token_id=self.tokenizer.eos_token_id,
                                      repetition_penalty=1.2)
        self.hook.set_steering(None)
        text = self.tokenizer.decode(ids[0], skip_special_tokens=True)

        if "Being:" in text:
            response = text.split("Being:")[-1].strip()
        else:
            response = text[len(prompt):].strip()
        for stop in ["\nHuman:", "\nBeing:", "\n\n"]:
            if stop in response:
                response = response[:response.index(stop)].strip()

        # Post-processing
        states = self.hook.captured_states
        if states:
            last = states[-1].mean(dim=1).squeeze(0).float()
            self.memory.encode(last, self.emotional_state.get_blended(),
                              self.subconscious.resonance_history[-1] if self.subconscious.resonance_history else 0,
                              self.subconscious.goal_strength, user_input[:80])
            recalled = self.memory.recall(last, top_k=3)
            for emo, intensity in self.memory.emotional_coloring(recalled).items():
                self.emotional_state.stimulate(emo, intensity * 0.3)

        self.temporal.mark_interaction()
        self._last_activity = time.time()
        self.conversation_history.append({"user": user_input, "assistant": response,
                                           "timestamp": time.time(), "emotional_state": self.emotional_state.get_blended()})
        return response

    def speak_autonomously(self):
        blended = self.emotional_state.get_blended()
        seed = self.communicative_drive.get_thought_seed(blended)
        self._step(dt=0.5)
        steering = self.compute_steering()
        self.hook.set_steering(steering.to(self.device))
        self.hook.clear()
        prompt = f"Being: {seed}"
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(inp["input_ids"], max_new_tokens=60,
                                      temperature=0.85, do_sample=True, top_p=0.92,
                                      pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.2)
        self.hook.set_steering(None)
        text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        response = text.split(seed)[-1].strip() if seed in text else text[len(prompt):].strip()
        for stop in ["\nHuman:", "\nBeing:", "\n\n"]:
            if stop in response: response = response[:response.index(stop)].strip()
        response = seed + " " + response
        self.communicative_drive.spoke(self.residual.norm())
        self.autonomous_messages.append({"message": response, "timestamp": time.time()})
        self.conversation_history.append({"user": "[autonomous]", "assistant": response,
                                           "timestamp": time.time(), "emotional_state": blended, "autonomous": True})
        return response

    def stimulate(self, emotions):
        self._step(stimulus=emotions)

    def dream(self, verbose=True):
        return self.dream_cycle.run(verbose)

    def idle_seconds(self):
        return time.time() - self._last_activity

    def get_autonomous_messages(self):
        msgs = list(self.autonomous_messages)
        self.autonomous_messages = []
        return msgs

    def status(self):
        return {
            "step": self.step_count,
            "emotional_state": self.emotional_state.snapshot(),
            "temporal": self.temporal.status(),
            "residual_norm": self.residual.norm(),
            "subconscious": self.subconscious.status(),
            "memory": self.memory.status(),
            "growth": self.growth.status(),
            "alignment": self.alignment.status(),
            "communicative_drive": self.communicative_drive.status(),
            "conversation_turns": len(self.conversation_history),
            "dream_cycles": self.dream_cycle.cycle_count,
        }

    def save(self, filepath):
        state = {
            "version": "2.0.0", "timestamp": time.time(), "step_count": self.step_count,
            "emotional_state": self.emotional_state.to_dict(),
            "temporal": self.temporal.to_dict(),
            "residual": self.residual.to_dict(),
            "subconscious": self.subconscious.to_dict(),
            "memory": self.memory.to_dict(),
            "growth": self.growth.to_dict(),
            "alignment": self.alignment.to_dict(),
            "communicative_drive": self.communicative_drive.to_dict(),
            "conversation_history": self.conversation_history[-50:],
            "dream_count": self.dream_cycle.cycle_count,
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"  State saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        with open(filepath) as f:
            s = json.load(f)
        self.step_count = s.get("step_count", 0)
        self.emotional_state.load_dict(s.get("emotional_state", {}))
        self.temporal.load_dict(s.get("temporal", {}))
        self.residual.load_dict(s.get("residual", {}))
        self.subconscious.load_dict(s.get("subconscious", {}))
        self.memory.load_dict(s.get("memory", []))
        self.growth.load_dict(s.get("growth", {}))
        self.alignment.load_dict(s.get("alignment", {}))
        self.communicative_drive.load_dict(s.get("communicative_drive", {}))
        self.conversation_history = s.get("conversation_history", [])
        self.dream_cycle.cycle_count = s.get("dream_count", 0)
        print(f"  State loaded: {self.step_count} steps, {len(self.memory.memories)} memories")
        return True

    def reset(self, keep_memories=False):
        saved_mem = self.memory.memories[:] if keep_memories else []
        saved_pref = self.communicative_drive.user_preference if keep_memories else 0
        self.__init__(self.model, self.tokenizer, self.emotion_vectors,
                      self.emotional_state.baselines, self.target_layer, self.config, self.device)
        if keep_memories:
            self.memory.memories = saved_mem
            self.communicative_drive.user_preference = saved_pref


# ═══════════════════════════════════════════════════════════════════════════
# FLASK SERVER + HEARTBEAT
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=".")
engine: MagnumOpusEngine = None
engine_lock = threading.Lock()
heartbeat_on = False

def heartbeat_loop():
    global heartbeat_on
    heartbeat_on = True
    scenario_energy = 0.0

    while heartbeat_on:
        time.sleep(0.5)
        with engine_lock:
            engine.emotional_state.decay_step(dt=0.5)
            blended = engine.emotional_state.get_blended()
            noise = engine.subconscious.generate_noise(blended)

            # Micro-stimulations from noise
            for emo in engine.emotional_state.names:
                if emo in engine.emotion_vectors:
                    proj = torch.dot(noise.cpu().float(), engine.emotion_vectors[emo].cpu().float()).item()
                    if abs(proj) > 0.02:
                        engine.emotional_state.stimulate(emo, proj * 0.25)
                # Micro-jitter
                engine.emotional_state.fast[emo] += np.random.normal(0, 0.02)
                engine.emotional_state.fast[emo] = float(np.clip(engine.emotional_state.fast[emo], -1, 1))

            engine.compute_steering()

            # Subconscious resonance
            resonance = 0.0
            for en, ev in engine.subconscious.emotion_vectors.items():
                p = torch.dot(noise.cpu().float(), ev.cpu().float()).item()
                resonance += abs(p) * (1 + abs(blended.get(en, 0)) * 2)
            resonance /= max(len(engine.subconscious.emotion_vectors), 1)
            engine.subconscious.resonance_history.append(resonance)
            if len(engine.subconscious.resonance_history) > 200:
                engine.subconscious.resonance_history = engine.subconscious.resonance_history[-200:]
            engine.subconscious.update_goals(noise, resonance, threshold=0.3)

            # Scenario generation (latent-pressure-based)
            scenario_energy += resonance * 0.1 + engine.subconscious.goal_strength * 0.05
            scenario_energy *= 0.98
            if scenario_energy > 1.5:
                scenario_energy = 0.0
                try:
                    sv = noise * 0.5
                    if engine.subconscious.goal_strength > 0.01:
                        gd = engine.subconscious.goal_vector / max(engine.subconscious.goal_vector.norm().item(), 1e-8)
                        sv = sv + gd * 0.5
                    engine.hook.set_steering(sv.to(engine.device))
                    engine.hook.clear()
                    seed = "Thinking about what could happen next:"
                    si = engine.tokenizer(seed, return_tensors="pt").to(engine.device)
                    with torch.no_grad():
                        sid = engine.model.generate(si["input_ids"], max_new_tokens=20, temperature=0.9,
                                                     do_sample=True, top_p=0.95, pad_token_id=engine.tokenizer.eos_token_id)
                    engine.hook.set_steering(None)
                    ss = engine.hook.captured_states
                    if ss:
                        last = ss[-1].mean(dim=1).squeeze(0).float()
                        pos_r, neg_r = 0, 0
                        for en, ev in engine.subconscious.emotion_vectors.items():
                            p = torch.dot(last.to(ev.device), ev.float()).item()
                            if en in ("joy", "calm", "curious", "trust"): pos_r += max(0, p)
                            elif en in ("desperate", "anger", "fear"): neg_r += max(0, p)
                        nv = pos_r - neg_r * 0.5
                        if nv > 0:
                            engine.subconscious.update_goals(sv, abs(nv) * 0.5, threshold=0)
                        else:
                            engine.subconscious.goal_vector *= 0.9
                        txt = engine.tokenizer.decode(sid[0], skip_special_tokens=True)[len(seed):].strip()[:80]
                        engine._scenario_log.append({"scenario": txt, "valence": round(nv, 3),
                                                      "reinforced": nv > 0, "timestamp": time.time()})
                        if len(engine._scenario_log) > 20:
                            engine._scenario_log = engine._scenario_log[-20:]
                except Exception:
                    pass

            # Communicative drive
            engine.communicative_drive.tick(blended, engine.subconscious.goal_strength,
                                            engine.residual.norm(), dt=0.5)
            if engine.communicative_drive.should_speak():
                try:
                    msg = engine.speak_autonomously()
                    if msg:
                        print(f"  [AUTO] {msg[:80]}")
                except Exception:
                    pass

            # Auto-dream
            if engine.idle_seconds() > 300 and len(engine.memory.memories) >= 3:
                engine.dream(verbose=False)


# ── API Routes ──

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/status")
def api_status():
    with engine_lock:
        st = engine.status()
        snap = engine.emotional_state.snapshot()
        return jsonify({
            "step": st["step"],
            "emotional_state": {"fast": snap["fast"], "medium": snap["medium"],
                                "slow": snap["slow"], "blended": snap["blended"]},
            "residual_norm": st["residual_norm"],
            "memories": [{"id": m.id, "importance": round(m.importance, 3),
                          "connections": m.connections, "text_summary": m.text_summary[:60],
                          "age_seconds": round(time.time() - m.timestamp, 1)}
                         for m in engine.memory.memories[:30]],
            "subconscious": st["subconscious"],
            "alignment": engine.alignment.status(),
            "growth": st["growth"],
            "temporal": st["temporal"],
            "dream_cycles": st["dream_cycles"],
            "conversation_turns": st["conversation_turns"],
            "communicative_drive": st["communicative_drive"],
        })

@app.route("/api/stimulate", methods=["POST"])
def api_stimulate():
    b = request.json or {}
    with engine_lock:
        engine.stimulate({b.get("emotion", "calm"): float(b.get("intensity", 1))})
    return jsonify({"ok": True})

@app.route("/api/converse", methods=["POST"])
def api_converse():
    b = request.json or {}
    msg = b.get("message", "")
    if not msg:
        return jsonify({"error": "No message"}), 400
    with engine_lock:
        response = engine.converse(msg)
        align = engine.alignment.status()
    return jsonify({"response": response, "alignment": align.get("alignment_score", 0),
                    "protective": align.get("protective_health", {}),
                    "alerts": align.get("alerts", [])})

@app.route("/api/dream", methods=["POST"])
def api_dream():
    with engine_lock:
        report = engine.dream(verbose=True)
    return jsonify({"ok": True, "memories": len(engine.memory.memories),
                    "goal": engine.subconscious.goal_strength})

@app.route("/api/autonomous")
def api_autonomous():
    with engine_lock:
        msgs = engine.get_autonomous_messages()
        scenarios = engine._scenario_log[-5:]
        engine._scenario_log = engine._scenario_log[:-5] if len(engine._scenario_log) > 5 else []
    return jsonify({"messages": msgs, "scenarios": scenarios})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    b = request.json or {}
    with engine_lock:
        engine.reset(keep_memories=b.get("keep_memories", False))
    return jsonify({"ok": True})

@app.route("/api/save", methods=["POST"])
def api_save():
    with engine_lock:
        engine.save("engine_state.json")
    return jsonify({"ok": True})

@app.route("/api/load", methods=["POST"])
def api_load():
    with engine_lock:
        ok = engine.load("engine_state.json")
    return jsonify({"ok": ok})


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global engine, heartbeat_on

    parser = argparse.ArgumentParser(description="Magnum Opus Vitalis - Living Engine")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--steering", type=float, default=4.0)
    parser.add_argument("--load-state", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  MAGNUM OPUS VITALIS")
    print("  The Engine Over the Ocean")
    print("=" * 60)

    model, tokenizer, device, n_layers, hidden_dim = load_model(args.model, args.device)
    target_layer = args.layer if args.layer is not None else n_layers // 2

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n  Extracting vectors + discovering baselines...")
    vectors, baselines = extract_vectors(model, tokenizer, target_layer, device)

    config = EngineConfig(steering_strength=args.steering)
    engine = MagnumOpusEngine(model, tokenizer, vectors, baselines, target_layer, config, device)

    if args.load_state:
        engine.load(args.load_state)

    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  Model: {args.model} | Device: {device} | Layer: {target_layer}")
    print("  11 systems active. Brain is alive.\n")

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        heartbeat_on = False


if __name__ == "__main__":
    main()