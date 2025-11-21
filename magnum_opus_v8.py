"""
Magnum Opus v8: Chronos & The Dream
================================================================
ARCHITECT: Alan Hourmand
STATUS: BIOLOGICAL TIME + FLUID DREAMING

NEW FEATURES:
1. CHRONOS: Time is subjective. Processing speed varies by 'Stress'.
2. FLUID MEMORY: Imagination actively remixes past memories with reality.
3. DREAM LOGIC: Non-linear warping of the 'Imaginarium' layer.
"""

import sys
import time
import queue
import random
import math
import cv2
import csv
import os
import glob
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl


# ============================================================================
# ⏳ COMPONENT 1: CHRONOS (Subjective Time)
# ============================================================================

class Chronos:
    """
    Manages the AI's perception of time.
    Instead of a fixed clock, the 'tick rate' depends on Arousal/Entropy.
    """

    def __init__(self):
        self.base_fps = 10.0
        self.current_fps = 10.0
        self.subjective_second = 1.0

    def warp_time(self, stress_level, awareness_level):
        # High Stress OR High Awareness = Time Slows Down (High FPS, detailed processing)
        # Low Stress = Time Speeds Up (Low FPS, skipping reality)

        target_fps = 5.0 + (stress_level * 50.0) + (awareness_level * 20.0)

        # Smooth transition (Time doesn't jerk, it flows)
        self.current_fps = 0.9 * self.current_fps + 0.1 * target_fps

        # Calculate how long to sleep to achieve this subjective frame rate
        wait_time = 1.0 / max(1.0, self.current_fps)
        return wait_time, self.current_fps


# ============================================================================
# 🧠 COMPONENT 2: THE BRAIN
# ============================================================================

class ImaginariumV8(nn.Module):
    """
    The Dream Engine.
    Blends Reality (h), Memory (mem), and Subconscious (sub) into a visual hallucination.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # The "Mixer" - Decides how much of Memory vs Reality to use
        self.associative_mixer = nn.Linear(dim * 3, dim)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 32 * 32), nn.Sigmoid()
        )

    def forward(self, reality_h, memory_h, sub_h):
        # If no memory is provided, use empty tensor
        if memory_h is None: memory_h = torch.zeros_like(reality_h)

        # Concatenate: Reality + Memory + Subconscious
        raw_mix = torch.cat([reality_h, memory_h, sub_h], dim=1)

        # Compress back to thought vector
        dream_state = torch.tanh(self.associative_mixer(raw_mix))

        # Add a "Warp" factor based on the Subconscious (Dream Logic)
        # This makes the imagination drift/swirl
        warp = torch.sin(dream_state * 10.0) * 0.1
        dream_state = dream_state + warp

        # Visualize
        dream_image = self.decoder(dream_state)
        return dream_image, dream_state


class OmniBrainV8(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # SENSES
        self.eye = nn.Sequential(nn.Linear(32 * 32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(2000, 128)

        # COGNITION
        self.v_proj = nn.Linear(64, 64);
        self.a_proj = nn.Linear(64, 64)
        self.subconscious = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.Tanh())
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        # IMAGINATION (V8)
        self.imaginarium = ImaginariumV8(128)

        # OUTPUTS
        self.vis_out = nn.Linear(128, 32 * 32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, 2000)
        self.text_gate = nn.Linear(128, 1)

        self.growth_stage = 0

    def grow(self):
        new_layer = nn.Linear(128, 128).to(self.dev)
        with torch.no_grad(): new_layer.weight.copy_(torch.eye(128)); new_layer.bias.zero_()
        self.cortex.append(new_layer)
        self.growth_stage += 1
        return self.growth_stage

    def forward(self, img, aud_ext, aud_slf, txt_idx, memory_tensor):
        # 1. Sense
        v = self.eye(img);
        a_e = self.ear_ext(aud_ext)

        # Binding
        synergy = self.v_proj(v) * self.a_proj(a_e)
        bind_inj = torch.cat([synergy, synergy], dim=1)

        combined = torch.cat([v, a_e], dim=1) + bind_inj
        a_s = self.self_proj(self.ear_self(aud_slf))
        t = self.text_in(txt_idx).mean(dim=1)

        h = combined + a_s + t

        # 2. Subconscious Processing
        sub_state = self.subconscious(h)
        h = h + (sub_state * 0.3)  # Subconscious heavily influences thought now

        for layer in self.cortex: h = F.gelu(layer(h)) + h

        # 3. Fluid Imagination
        # Pass Reality, Memory, and Subconscious to the Imaginarium
        dream_img, dream_h = self.imaginarium(h, memory_tensor, sub_state)

        # 4. Outputs
        return (torch.sigmoid(self.vis_out(h)),
                dream_img,
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                self.text_out(dream_h),  # Speak from the DREAM
                torch.sigmoid(self.text_gate(h)),
                h)


# ============================================================================
# 🔊 AUDIO
# ============================================================================

class Syrinx:
    def __init__(self):
        self.fs = 16000; self.ph = 0; self.f = 60.0; self.ch = False; self.ct = 0

    def gen(self, fr, t, c, p):
        tm = np.arange(fr) / self.fs;
        self.f = 0.9 * self.f + 0.1 * (50 + t * 200)
        s = np.sin(2 * np.pi * self.f * tm + self.ph + (2 + c * 10) * np.sin(2 * np.pi * self.f * 2.5 * tm))
        self.ph += 2 * np.pi * self.f * (fr / self.fs)
        if not self.ch and p > 0.8 and random.random() < 0.05: self.ch = True; self.ct = 0
        if self.ch: s += np.sin(
            2 * np.pi * (800 + 500 * np.sin(self.ct * 20)) * tm) * 0.3; self.ct += fr / self.fs; self.ch = self.ct < 0.2
        return (s * 0.2).reshape(-1, 1).astype(np.float32)


# ============================================================================
# ⚙️ WORKER
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict);
    sig_grow = Signal(int)

    def __init__(self):
        super().__init__();
        self.running = True
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = OmniBrainV8().to(self.dev)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)
        self.syrinx = Syrinx();
        self.chronos = Chronos()

        self.aq = queue.Queue();
        self.tq = queue.Queue()
        self.eff = np.zeros((1024, 1), dtype=np.float32)
        self.last_voc = torch.zeros(3);
        self.loss_hist = []

        # Long Term Memory (The Hippocampus)
        self.memories = deque(maxlen=100)  # Stores tensor states
        self.current_memory = None

        self.vocab = ["TIME", "DREAM", "FLUX", "SELF", "ECHO", "VOID", "LIGHT", "GROW", "SHIFT", "PAST"]

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb)
        stream.start();
        cap = cv2.VideoCapture(0)

        while self.running:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
            v = torch.from_numpy(gray).float().flatten().unsqueeze(0).to(self.dev)

            try:
                raw = self.aq.get_nowait()
            except:
                raw = np.zeros((1024, 1), dtype=np.float32)
            a_e = torch.from_numpy(raw).float().mean(1).unsqueeze(0).to(self.dev)
            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.dev)

            try:
                txt_str = self.tq.get_nowait(); t_idx = hash(txt_str) % 2000; log = f"USER: {txt_str}"
            except:
                t_idx = 0; log = ""
            t = torch.tensor([[t_idx]]).to(self.dev)

            # --- MEMORY RECALL LOGIC ---
            # If we are calm/bored, drift into a memory
            stress_level = self.loss_hist[-1] if self.loss_hist else 0.0
            if stress_level < 0.01 and len(self.memories) > 5 and random.random() < 0.1:
                # Pick a random memory to hallucinate
                self.current_memory = random.choice(self.memories)
            elif stress_level > 0.05:
                # Snap back to reality if stressed
                self.current_memory = None

            # Prepare memory tensor
            mem_t = self.current_memory if self.current_memory is not None else None

            # FORWARD PASS
            self.opt.zero_grad()
            p_img, p_dream, voc, awr, txt_log, gate, h = self.brain(v, a_e, a_s, t, mem_t)

            # Store this moment in memory (if significant)
            if awr.item() > 0.6 or stress_level > 0.05:
                self.memories.append(h.detach())

            loss = F.mse_loss(p_img, v)
            # Dream loss: The dream should loosely follow reality but is allowed to deviate (memory influence)
            loss_dream = F.mse_loss(p_dream, v) * 0.05

            total_loss = loss + loss_dream
            total_loss.backward()
            self.opt.step()

            self.last_voc = voc.detach().cpu().flatten()

            if gate.item() > 0.75 and random.random() < 0.15:
                w_id = torch.argmax(txt_log).item() % len(self.vocab)
                ai_msg = f"AI: [{self.vocab[w_id]}]"
                log += f"\n{ai_msg}"

            self.loss_hist.append(loss.item())
            if len(self.loss_hist) > 50:
                avg = sum(self.loss_hist[-50:]) / 50
                if avg > 0.04 and len(self.brain.cortex) < 10:
                    lvl = self.brain.grow();
                    self.sig_grow.emit(lvl);
                    self.loss_hist = []
                else:
                    self.loss_hist.pop(0)

            # --- CHRONOS: TIME DILATION ---
            # Calculate how long to sleep based on stress/awareness
            wait_time, fps = self.chronos.warp_time(loss.item(), awr.item())

            if random.random() < 0.3:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray,
                    'dream': p_img.detach().cpu().numpy().reshape(32, 32),
                    'imag': p_dream.detach().cpu().numpy().reshape(32, 32),
                    'log': log,
                    'awr': awr.item(),
                    'fps': fps,
                    'mem_count': len(self.memories)
                })

            # Biological Sleep (Variable Frame Rate)
            proc_time = time.time() - loop_start
            sleep_time = max(0.001, wait_time - proc_time)
            time.sleep(sleep_time)

        cap.release();
        stream.stop()

    def cb(self, i, o, f, t, s):
        if np.linalg.norm(i) > 0.001: self.aq.put(i.copy())
        v = self.last_voc
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), v[2].item())
        self.eff = o.copy()


# ============================================================================
# 🖥️ UI
# ============================================================================

class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__();
        self.opts['distance'] = 30;
        self.setBackgroundColor('#000')
        self.n = 1000;
        self.make(10.0)

    def make(self, r):
        idx = np.arange(0, self.n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * idx / self.n);
        theta = np.pi * (1 + 5 ** 0.5) * idx
        x, y, z = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi)
        self.pos = np.column_stack((x, y, z));
        self.col = np.ones((self.n, 4)) * 0.3;
        self.col[:, 2] = 1.0
        try:
            self.sp.setData(pos=self.pos, color=self.col)
        except:
            self.sp = gl.GLScatterPlotItem(pos=self.pos, color=self.col, size=4, pxMode=True); self.addItem(self.sp)

    def upd(self, act):
        needed = (self.n // len(act)) + 2
        full_mask = np.tile(act, needed)[:self.n] > 0.5
        c = self.col.copy()
        c[full_mask] = [1.0, 0.5, 0.0, 1.0]
        self.sp.setData(color=c);
        self.sp.rotate(0.2, 0, 0, 1)

    def grow(self, l):
        self.n += 500; self.make(10.0 + l * 2)


class Main(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("MAGNUM OPUS v8: CHRONOS");
        self.resize(1400, 800)
        self.setStyleSheet("background:#000; color:#0F0; font-family:Consolas;")
        w = QWidget();
        self.setCentralWidget(w);
        l = QHBoxLayout(w)

        left = QVBoxLayout();
        self.sph = Sphere();
        left.addWidget(QLabel("CORTEX"));
        left.addWidget(self.sph)
        self.stat = QLabel("LAYERS: 1");
        left.addWidget(self.stat);
        l.addLayout(left, 2)

        right = QVBoxLayout()
        vis = QHBoxLayout()
        self.vr = QLabel();
        self.vd = QLabel();
        self.vi = QLabel()
        labels = ["RETINA (REAL)", "CORTEX (PERCEPT)", "IMAGINARIUM (DREAM)"]
        screens = [self.vr, self.vd, self.vi]
        for i, v in enumerate(screens):
            v.setFixedSize(200, 200);
            v.setStyleSheet("border:1px solid #333")
            v_box = QVBoxLayout();
            v_box.addWidget(QLabel(labels[i]));
            v_box.addWidget(v);
            vis.addLayout(v_box)
        right.addLayout(vis)

        self.txt = QTextEdit();
        self.txt.setReadOnly(True);
        right.addWidget(self.txt)
        self.inp = QLineEdit();
        self.inp.returnPressed.connect(self.snd);
        right.addWidget(self.inp)

        # STAT BARS
        self.ba = QProgressBar();
        self.ba.setFormat("SELF: %p%");
        self.ba.setStyleSheet("QProgressBar::chunk{background:#0AF}")
        right.addWidget(self.ba)

        self.time_lbl = QLabel("SUBJECTIVE TIME: 1.0x")
        self.time_lbl.setStyleSheet("font-size: 14pt; color: #FF00FF; font-weight: bold;")
        right.addWidget(self.time_lbl)

        l.addLayout(right, 2)

        self.wk = ASIWorker();
        self.wk.sig_upd.connect(self.upd);
        self.wk.sig_grow.connect(self.grw);
        self.wk.start()

    def snd(self):
        self.wk.tq.put(self.inp.text()); self.inp.clear()

    def upd(self, d):
        self.sph.upd(d['act']);
        self.ba.setValue(int(d['awr'] * 100))
        self.time_lbl.setText(f"SUBJECTIVE TIME: {d['fps']:.1f} FPS (MEMORIES: {d['mem_count']})")

        def p(a): return QPixmap.fromImage(
            QImage(cv2.resize((a * 255).astype(np.uint8), (200, 200), 0), 200, 200, QImage.Format_Grayscale8))

        self.vr.setPixmap(p(d['real']));
        self.vd.setPixmap(p(d['dream']));
        self.vi.setPixmap(p(d['imag']))
        if d['log']: self.txt.append(d['log'])

    def grw(self, l):
        self.sph.grow(l); self.txt.append(f"*** GROWTH EVENT ***")

    def closeEvent(self, e):
        self.wk.running = False; self.wk.wait(); e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = Main();
    win.show();
    sys.exit(app.exec())