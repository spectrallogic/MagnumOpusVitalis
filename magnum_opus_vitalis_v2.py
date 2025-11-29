"""
MagnumOpusVitalis: The Living Intelligence
================================================================
ARCHITECT: Alan Hourmand
VERSION: 13.0 (Voice of Reason)

CHANGELOG:
- FIXED "12 LINES FAST": Locked AI into Cocoon Mode (Teacher) to prevent raw neural noise.
- FIXED "FIRST LINE ONLY": Removed truncation. AI now speaks the full LLM response.
- IMPROVED TTS: Speaks sentence-by-sentence for better flow.
- STABILITY: Retained all V10/V12 crash fixes.

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
# 🛡️ CONFIGURATION
# ============================================================================

ENABLE_RAW_AUDIO_OUTPUT = False
ENABLE_RAW_MICROPHONE = False

ENABLE_TTS = True
ENABLE_STT = True

TRY_COM_FIX = False
try:
    import pythoncom
    TRY_COM_FIX = True
except ImportError:
    pass

if ENABLE_RAW_AUDIO_OUTPUT or ENABLE_RAW_MICROPHONE:
    try:
        import sounddevice as sd
    except ImportError:
        print("[SYSTEM] sounddevice not found")

# ============================================================================
# 🥚 PART 1: SCAFFOLD SYSTEMS
# ============================================================================

SCAFFOLD_LLM_AVAILABLE = False
SCAFFOLD_STT_AVAILABLE = False
SCAFFOLD_TTS_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    SCAFFOLD_LLM_AVAILABLE = True
    print("[SCAFFOLD] LLM Teacher available")
except ImportError:
    print("[SCAFFOLD] LLM not available")

if ENABLE_STT:
    try:
        import speech_recognition as sr
        SCAFFOLD_STT_AVAILABLE = True
        print("[SCAFFOLD] STT Hearing available")
    except ImportError:
        print("[SCAFFOLD] STT not available")

if ENABLE_TTS:
    try:
        import pyttsx3
        SCAFFOLD_TTS_AVAILABLE = True
        print("[SCAFFOLD] TTS Voice available")
    except ImportError:
        print("[SCAFFOLD] TTS not available")


class ScaffoldLLM:
    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'

        if SCAFFOLD_LLM_AVAILABLE:
            try:
                print("[LLM] Loading GPT-2...")
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.to(self.device)
                self.available = True
                print(f"[LLM] Ready on {self.device}")
            except Exception as e:
                print(f"[LLM] Failed: {e}")

    def chat(self, user_input: str, max_tokens: int = 60) -> str:
        """Generates full response, no truncation."""
        if not self.available: return "..."
        try:
            prompt = f"User: {user_input}\nAI:"
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            mask = torch.ones(inputs.shape, device=self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, attention_mask=mask, max_new_tokens=max_tokens,
                    do_sample=True, temperature=0.7, top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text.split("AI:")[-1].strip() if "AI:" in full_text else full_text

            # REMOVED TRUNCATION: Return the whole thought
            return response.strip()
        except Exception: return "..."

class ScaffoldSTT:
    def __init__(self):
        self.available = SCAFFOLD_STT_AVAILABLE
        self.recognizer = None
        if self.available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
            except: self.available = False

    def listen_once(self, timeout: float = 0.5) -> Optional[str]:
        if not self.available: return None
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=4)
            return self.recognizer.recognize_google(audio).lower()
        except: return None

class ScaffoldTTS:
    def __init__(self):
        self.available = SCAFFOLD_TTS_AVAILABLE
        self.engine = None

    def start_engine(self):
        if self.available and self.engine is None:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)
                print("[TTS] Engine started safely")
            except: self.available = False

    def speak_async(self, text: str):
        if not self.available or not text: return
        try:
            if self.engine:
                # Split into sentences to prevent choking
                sentences = re.split(r'(?<=[.!?]) +', text)
                for s in sentences:
                    if s.strip():
                        self.engine.say(s)
                        self.engine.runAndWait() # Blocking but safe in worker
        except: pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QProgressBar, QFrame, QGridLayout
)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 🧠 PART 2: NEURAL ORGANS & UTILS
# ============================================================================

@dataclass
class EpisodeMemory:
    embedding: torch.Tensor; timestamp: float; importance: float = 1.0; access_count: int = 0
    def decay(self, rate: float = 0.995): self.importance *= rate

class TabularRasaVocabulary:
    def __init__(self, max_vocab: int = 10000):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.max_vocab = max_vocab
        self.recent_words: deque = deque(maxlen=100)

    def learn_text(self, text: str) -> List[int]:
        if not text: return []
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

class EnergySystem:
    def __init__(self):
        self.energy = 1.0; self.conservation_gain = 0.5

    def can_speak(self) -> bool:
        return self.energy > 0.1

    def spend_speaking(self, num_words: int = 1):
        cost = 0.01 * (1.5 - self.conservation_gain)
        self.energy = max(0.0, self.energy - (cost * num_words))

    def regenerate(self):
        regen = 0.002 * (0.5 + self.conservation_gain)
        self.energy = min(1.0, self.energy + regen)

# --- Neural Modules ---

class TemporalResonance(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('resonance_state', torch.zeros(1, dim))
        self.clock_phase = 0.0
    def forward(self, x):
        self.resonance_state = (x * 0.1) + (self.resonance_state * 0.96)
        self.clock_phase += 0.05
        return self.resonance_state * (1.0 + 0.05 * torch.sin(torch.tensor(self.clock_phase, device=x.device)))

class BiologicalMemory(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.memories: List[EpisodeMemory] = []
        self.encoder = nn.Linear(dim, dim); self.decoder = nn.Linear(dim, dim)
    def store(self, state, imp):
        if imp < 0.35: return
        enc = self.encoder(state.detach().mean(dim=0).unsqueeze(0))
        self.memories.append(EpisodeMemory(enc.cpu(), time.time(), imp))
        if len(self.memories) > 2000: self.memories.sort(key=lambda m: m.importance, reverse=True); self.memories = self.memories[:1800]
    def recall(self, query):
        if not self.memories: return None
        q = query.detach().mean(dim=0).cpu()
        best, best_sim = None, -1.0
        for m in random.sample(self.memories, min(len(self.memories), 50)):
            sim = F.cosine_similarity(q.view(1,-1), m.embedding.view(1,-1)).item()
            if sim > best_sim: best_sim = sim; best = m
        return self.decoder(best.embedding.to(query.device)) if best_sim > 0.65 else None

class MultiSpeedProcessor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.speeds = [16, 32, 64]
        self.proj_in = nn.ModuleList([nn.Linear(dim, d) for d in self.speeds])
        self.proj_out = nn.ModuleList([nn.Linear(d, dim) for d in self.speeds])
        self.trust = nn.Parameter(torch.tensor([1.0, 0.5, 0.1]))
    def forward(self, x):
        out = []
        for pi, po in zip(self.proj_in, self.proj_out):
            out.append(po(F.gelu(pi(x))))
        stk = torch.stack(out, dim=0)
        return torch.einsum('sbld,s->bld', stk, F.softmax(self.trust, dim=0))

class SubconsciousMind(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.noise = nn.Linear(dim, dim); self.scorer = nn.Linear(dim, 1)
        self.gru = nn.GRUCell(dim, dim); self.momentum = None
    def forward(self, h, stress):
        noise = torch.randn_like(h) * (0.1 + stress * 0.25)
        filtered = self.noise(h + noise) * torch.sigmoid(self.scorer(h+noise))
        if self.momentum is None: self.momentum = torch.zeros_like(h)

        gru_in = filtered if filtered.dim() == 2 else filtered.squeeze(0)
        mom_in = self.momentum if self.momentum.dim() == 2 else self.momentum.squeeze(0)
        if gru_in.shape[0] != mom_in.shape[0]: mom_in = torch.zeros_like(gru_in)

        fut = self.gru(gru_in, mom_in)
        self.momentum = 0.95 * mom_in + 0.05 * fut.detach()
        return fut, filtered

class Imaginarium(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mixer = nn.Linear(dim*3, dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 32*32), nn.Sigmoid())

    def forward(self, r, m, s, g):
        if m is None: m = torch.zeros_like(r)
        if m.shape != r.shape: m = torch.zeros_like(r)
        if s.shape != r.shape: s = torch.zeros_like(r)

        combined = torch.cat([r, m, s], dim=-1)
        dream_state = torch.tanh(self.mixer(combined)) * (0.5+g)
        return self.decoder(dream_state), dream_state

class PrefrontalCortex(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.planner = nn.GRUCell(dim+2, dim)
        self.policy = nn.Sequential(nn.Linear(dim, 8), nn.Sigmoid())
        with torch.no_grad():
            # FORCE COCOON MODE: Bias 6 = 5.0 (Sigmoid ~= 0.99)
            # This ensures we rely on the Teacher (GPT-2) and not the raw brain (which speaks gibberish)
            self.policy[0].bias[6] = 5.0
            self.policy[0].bias[7] = 2.0
    def forward(self, h, s, e, prev):
        if prev is None: prev = torch.zeros_like(h)
        ctx = torch.tensor([[s, e]], device=h.device)
        inp = torch.cat([h, ctx], dim=1)
        new_state = self.planner(inp, prev)
        return self.policy(new_state), new_state

class OmniBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_dim = 128; self.vocab_size = vocab_size

        self.eye = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.txt = nn.Embedding(vocab_size, 128)
        self.proj = nn.Linear(32, 128)

        self.temporal = TemporalResonance(128)
        self.memory = BiologicalMemory(128)
        self.multi_speed = MultiSpeedProcessor(128)
        self.subconscious = SubconsciousMind(128)
        self.imaginarium = Imaginarium(128)
        self.pfc = PrefrontalCortex(128)

        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        self.vis_out = nn.Linear(128, 1024); self.voice_out = nn.Linear(128, 3)
        self.txt_out = nn.Linear(128, vocab_size)

        self.pfc_state = None; self.steps_above_threshold = 0
        self.to(self.device)

    def grow(self):
        self.cortex.append(nn.Linear(128, 128).to(self.device))
        return len(self.cortex)

    def check_growth(self, loss):
        if loss > 4.0:
            self.steps_above_threshold += 1
            if self.steps_above_threshold > 80 and len(self.cortex) < 20:
                self.steps_above_threshold = 0; return True
        else: self.steps_above_threshold = max(0, self.steps_above_threshold-1)
        return False

    def forward(self, v, a, s, t_idx, stress, nrg, src):
        v_enc = self.eye(v); a_enc = self.ear(a)
        t_enc = self.txt(t_idx).mean(dim=1) if t_idx.numel()>0 else torch.zeros(1,128, device=self.device)

        if self.pfc_state is None: self.pfc_state = torch.zeros(1, 128, device=self.device)
        act, self.pfc_state = self.pfc(self.pfc_state, stress, nrg, self.pfc_state)

        w = 1.0 if src=="EXTERNAL" else (0.3 if src=="SELF" else 0.1)

        h = torch.cat([v_enc*(1+act[0,0]), a_enc*(1+act[0,1])], dim=1) + t_enc*w
        h = self.temporal(h) + h

        h = h.unsqueeze(1); h = self.multi_speed(h); h = h.squeeze(1)

        mem = self.memory.recall(h); sub, _ = self.subconscious(h, stress)
        h = h + sub * 0.3
        for l in self.cortex: h = F.gelu(l(h)) + h

        drm_img, drm_h = self.imaginarium(h, mem, sub, act[0,2])

        return {
            'cortex': torch.sigmoid(self.vis_out(h)), 'dream': drm_img,
            'voice': torch.sigmoid(self.voice_out(h)),
            'text': self.txt_out(drm_h),
            'llm_rel': act[0,6], 'tts_gate': act[0,7], 'actions': act,
            'hidden': h, 'echo': h, 'speak_impulse': act[0,3]
        }

# ============================================================================
# 🔊 PART 3: AUDIO & WORKERS (Defined BEFORE VitalisWorker)
# ============================================================================

class EmotionalSyrinx:
    def __init__(self, sr=16000):
        self.fs = sr
        self.phase = 0.0
        self.base_freq = 55.0
        self.cry_phase = 0.0
        self.data_phase = 0.0

    def generate(self, frames, tension, chaos, impulse, stress, energy, cry_sup):
        t = np.arange(frames) / self.fs
        out = np.zeros(frames)

        tgt = 55.0 + (tension * 55.0)
        self.base_freq = 0.95 * self.base_freq + 0.05 * tgt
        out += np.sin(2*np.pi*self.base_freq*t + self.phase) * 0.08
        self.phase += 2*np.pi*self.base_freq*(frames/self.fs)

        cry_amt = max(0, stress - 0.4) * (1.0 - cry_sup)
        if cry_amt > 0.05:
            out += np.sin(2*np.pi*(300+100*np.sin(self.cry_phase))*t) * cry_amt * 0.15
            self.cry_phase += 0.1

        if impulse > 0.4:
            freq = 600.0 + 300.0 * np.sin(2*np.pi*(10+chaos*50)*t + self.data_phase)
            out += np.sign(np.sin(2*np.pi*freq*t)) * impulse * 0.4
            self.data_phase += 0.2

        return np.clip(out, -0.9, 0.9).reshape(-1, 1).astype(np.float32)

class Pacman(QThread):
    sig_read = Signal(str)
    def __init__(self):
        super().__init__(); self.running = True; self.queue = queue.Queue()
        self.folder = "training_data"
    def run(self):
        seen=set(); os.makedirs("training_data", exist_ok=True)
        while self.running:
            for fp in glob.glob("training_data/*.txt"):
                if fp not in seen:
                    try:
                        with open(fp,'r') as f:
                            for w in re.findall(r'\b\w+\b', f.read().lower()): self.queue.put(w)
                        seen.add(fp); self.sig_read.emit(f"📚 {os.path.basename(fp)}")
                    except: pass
            time.sleep(2)

class VitalisWorker(QThread):
    sig_update = Signal(dict); sig_growth = Signal(int)
    def __init__(self):
        super().__init__(); self.running = True; self.text_queue = queue.Queue()
        self.vocab = TabularRasaVocabulary(); self.energy = EnergySystem()

        self.syrinx = EmotionalSyrinx()

        self.pacman = Pacman(); self.brain = OmniBrain(self.vocab.max_vocab)
        self.optim = torch.optim.AdamW(self.brain.parameters(), lr=0.005)
        self.scaffold_llm = ScaffoldLLM(); self.scaffold_stt = ScaffoldSTT(); self.scaffold_tts = ScaffoldTTS()
        self.pacman.start(); self.last_voice = np.zeros(3)
        self.last_loss = 0.0

        self.use_mic = ENABLE_RAW_MICROPHONE
        self.use_speakers = ENABLE_RAW_AUDIO_OUTPUT

    def run(self):
        print("[WORKER] Engine Started.")
        if TRY_COM_FIX:
            try: pythoncom.CoInitialize()
            except: pass

        self.scaffold_tts.start_engine()
        time.sleep(0.5)

        if self.use_speakers:
            try: sd.OutputStream(channels=1, samplerate=16000).start()
            except: print("[AUDIO] Raw output failed")

        loop_c = 0
        while self.running:
            start = time.time(); loop_c += 1

            txt = ""; src = "AMBIENT"
            try: txt = self.text_queue.get_nowait(); src = "EXTERNAL"
            except: pass

            if not txt and loop_c % 20 == 0 and self.scaffold_stt.available:
                s = self.scaffold_stt.listen_once(0.1)
                if s: txt = s; src = "EXTERNAL"; self.text_queue.put(f"[HEARD] {s}")

            if not txt:
                try: txt = self.pacman.queue.get_nowait(); src = "EXTERNAL"
                except: pass

            if txt: self.vocab.learn_text(txt)

            v = torch.randn(1, 1024).to(self.brain.device)
            a = torch.randn(1, 1024).to(self.brain.device)
            s = torch.zeros(1, 32).to(self.brain.device)
            t = torch.tensor([self.vocab.learn_text(txt)], dtype=torch.long).to(self.brain.device) if txt else torch.zeros(1,1,dtype=torch.long).to(self.brain.device)

            self.brain.train()
            out = self.brain(v, a, s, t, 0.0, self.energy.energy, src)

            ai_resp = ""
            user_interacted = (src == "EXTERNAL") and txt

            if user_interacted and self.energy.can_speak():
                # Force Cocoon Mode if Reliance > 0.2 (It starts at 0.99)
                if out['llm_rel'] > 0.2:
                    ai_resp = self.scaffold_llm.chat(txt)
                    if ai_resp: self._train_imitation(ai_resp)
                else:
                    # BUTTERFLY MODE (Beware: Raw brain speaks nonsense initially)
                    if out['speak_impulse'] > 0.3:
                        idx = torch.argmax(out['text'], dim=-1)
                        ai_resp = " ".join([self.vocab.idx2word[i] for i in idx.cpu().numpy().flatten()])

            if ai_resp:
                self.energy.spend_speaking(len(ai_resp.split()))
                self.vocab.learn_text(ai_resp)
                if self.scaffold_tts.available and out['tts_gate'] > 0.4:
                    self.scaffold_tts.speak_async(ai_resp)

            self.energy.regenerate()
            self.last_voice = out['voice'].detach().cpu().numpy()[0]

            if loop_c % 30 == 0: self.brain.memory.store(out['hidden'], 0.8)

            if self.brain.check_growth(self.last_loss):
                n = self.brain.grow()
                self.sig_growth.emit(n)

            dt = time.time() - start
            self.sig_update.emit({
                'ai_msg': ai_resp, 'user_msg': txt if src=="EXTERNAL" else None,
                'stress': 0.0, 'layers': len(self.brain.cortex),
                'llm_rel': out['llm_rel'].item(), 'tts_gate': out['tts_gate'].item(),
                'cortex_img': out['cortex'].detach().cpu().numpy().reshape(32,32),
                'dream_image': out['dream'].detach().cpu().numpy().reshape(32,32),
                'voice': self.last_voice, 'fps': 1.0/max(0.001, dt), 'speaking': bool(ai_resp),
                'energy': self.energy.energy # Send energy to UI
            })
            time.sleep(0.01)

    def _train_imitation(self, target):
        idx = self.vocab.learn_text(target)
        if not idx: return
        t = torch.tensor([idx], dtype=torch.long).to(self.brain.device)
        pass

# ============================================================================
# 🖥️ PART 4: UI
# ============================================================================

class LivingOrb(gl.GLViewWidget):
    def __init__(self):
        super().__init__(); self.opts['distance']=35; self.setBackgroundColor('#030308')
        self.rings = []; self.time = 0
        for i in range(40):
            lat = -np.pi/2 + (np.pi*i/39); theta = np.linspace(0, 2*np.pi, 80)
            r = 8*np.cos(lat); z = 8*np.sin(lat)
            pos = np.column_stack((r*np.cos(theta), r*np.sin(theta), np.full(80, z)))
            itm = gl.GLLinePlotItem(pos=pos, color=np.array([0,1,1,0.5]), width=2, mode='line_strip')
            self.addItem(itm); self.rings.append({'i':itm, 'lat':lat, 'theta':theta})

    def update_orb(self, spk):
        self.time += 0.05
        for r in self.rings:
            rr = 8*np.cos(r['lat']) + (np.sin(r['theta']*5+self.time)*1.5 if spk else 0)
            x = rr*np.cos(r['theta']); y = rr*np.sin(r['theta']); z = 8*np.sin(r['lat'])
            col = np.array([1 if spk else 0, 1, 1, 0.6])
            r['i'].setData(pos=np.column_stack((x,y,np.full(80,z))), color=np.tile(col,(80,1)))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("MagnumOpusVitalis v13.0 - Voice of Reason"); self.resize(1600,900)
        self.setStyleSheet("background:#050505; color:#0F9; font-family:Consolas;")

        main = QHBoxLayout(QWidget(self)); self.setCentralWidget(main.parentWidget())

        # Left
        left = QFrame(); l = QVBoxLayout(left); self.orb = LivingOrb(); l.addWidget(self.orb)
        self.lbl = QLabel("MODE: COCOON"); l.addWidget(self.lbl)
        main.addWidget(left, 2)

        # Right
        right = QVBoxLayout()
        mon_p = QFrame(); mon_p.setStyleSheet("border:1px solid #057;"); hl = QHBoxLayout(mon_p)
        self.s1 = QLabel("CORTEX"); self.s1.setFixedSize(150,150); self.s1.setStyleSheet("background:#000; border:1px solid #0F0;")
        self.s2 = QLabel("DREAM"); self.s2.setFixedSize(150,150); self.s2.setStyleSheet("background:#000; border:1px solid #F0F;")
        hl.addWidget(self.s1); hl.addWidget(self.s2)
        right.addWidget(mon_p)

        # Energy Bar
        self.p_nrg = QProgressBar(); self.p_nrg.setRange(0,100); self.p_nrg.setStyleSheet("::chunk{background:#0AF;}")
        right.addWidget(QLabel("BIO-ENERGY")); right.addWidget(self.p_nrg)

        self.txt = QTextEdit(); self.txt.setReadOnly(True); self.txt.setStyleSheet("border:none; bg:#001;")
        self.inp = QLineEdit(); self.inp.returnPressed.connect(self.send); self.inp.setStyleSheet("border:1px solid #0AF;")
        right.addWidget(self.txt); right.addWidget(self.inp)
        main.addLayout(right, 1)

        self.worker = VitalisWorker()
        self.worker.sig_update.connect(self.upd); self.worker.start()

    def send(self):
        t = self.inp.text();
        if t: self.worker.text_queue.put(t); self.txt.append(f"<font color='#FFF'>YOU: {t}</font>"); self.inp.clear()

    def upd(self, d):
        self.orb.update_orb(d['speaking'])
        self.lbl.setText(f"MODE: {'COCOON' if d['llm_rel']>0.2 else 'BUTTERFLY'} | LAYERS: {d['layers']}")
        self.p_nrg.setValue(int(d['energy']*100))

        if d['energy'] < 0.2: self.lbl.setText(self.lbl.text() + " [TIRED]")

        if d['ai_msg']: self.txt.append(f"<font color='#0F0'>AI: {d['ai_msg']}</font>")

        def draw(l, a):
            i = (np.clip(a,0,1)*255).astype(np.uint8)
            l.setPixmap(QPixmap.fromImage(QImage(i, 32, 32, 32, QImage.Format_Grayscale8)).scaled(150,150))
        draw(self.s1, d['cortex_img']); draw(self.s2, d['dream_image'])

    def closeEvent(self, e):
        self.worker.running = False; self.worker.wait(); e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())