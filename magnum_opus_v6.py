"""
Magnum Opus v6: The Omni-Presence (Final Integration)
================================================================
ARCHITECT: Alan Hourmand
STATUS: FULLY INTEGRATED

COMPONENTS:
1. BODY: Vision, Audio, Syrinx Voice, Mirror Neurons.
2. BRAIN: Multi-Speed, Subconscious, Real-time Growth.
3. MEMORY: Episodic Storage & REM Sleep (Dreaming).
4. PACMAN: Background File Ingestion.
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
import threading
from collections import deque
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QSplitter, QProgressBar)
from PySide6.QtCore import QTimer, Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap, QColor
import pyqtgraph.opengl as gl


# ============================================================================
# 🧠 COMPONENT 1: LONG-TERM MEMORY (The Hippocampus)
# ============================================================================

class EpisodicMemory:
    def __init__(self, capacity=5000):
        self.memory = deque(maxlen=capacity)

    def store(self, tensors, loss):
        # Only store "significant" moments (High emotion or High learning)
        # Tensors: (vis, aud_ext, aud_slf, txt)
        self.memory.append((tensors, loss))

    def recall(self, batch_size=1):
        if len(self.memory) < batch_size: return None
        batch = random.sample(self.memory, batch_size)
        return batch[0][0]  # Return tensors of a random memory


# ============================================================================
# 📂 COMPONENT 2: PACMAN (File Reader)
# ============================================================================

class PacmanFeeder(QThread):
    sig_read = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.queue = queue.Queue()

    def run(self):
        seen_files = set()
        while self.running:
            # Look for files
            if not os.path.exists("training_data"):
                os.makedirs("training_data", exist_ok=True)

            files = glob.glob("training_data/*.txt")
            for f in files:
                if f not in seen_files:
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            text = file.read()
                            # Chunk it
                            chunks = [text[i:i + 50] for i in range(0, len(text), 50)]
                            for c in chunks:
                                self.queue.put(c)
                        seen_files.add(f)
                        self.sig_read.emit(f"PACMAN: Digested {f}")
                    except:
                        pass
            time.sleep(2)  # Check every few seconds


# ============================================================================
# 🧠 COMPONENT 3: THE OMNI BRAIN (v2 + v5 Combined)
# ============================================================================

class SynestheticBinder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.a_proj = nn.Linear(dim, dim)
        self.binding_score = nn.Linear(dim, 1)

    def forward(self, v, a):
        synergy = self.v_proj(v) * self.a_proj(a)
        return synergy, torch.sigmoid(self.binding_score(synergy))


class BiologicalMultiSpeed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dims = [8, 32, 64, 128]  # Simplified for realtime speed
        self.projections = nn.ModuleList([nn.Linear(dim, d) for d in self.dims])
        self.out_projections = nn.ModuleList([nn.Linear(d, dim) for d in self.dims])
        self.weights = nn.Parameter(torch.ones(len(self.dims)))

    def forward(self, x):
        out = 0
        w = F.softmax(self.weights, dim=0)
        for i, (p_in, p_out) in enumerate(zip(self.projections, self.out_projections)):
            out += p_out(F.gelu(p_in(x))) * w[i]
        return out


class SubconsciousMind(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x + torch.randn_like(x) * 0.05)


class OmniBrain(nn.Module):
    def __init__(self):
        super().__init__()

        # SENSES
        self.eye = nn.Sequential(nn.Linear(32 * 32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(2000, 128)  # Larger vocab for Pacman

        # COGNITION
        self.binder = SynestheticBinder(64)
        self.multi_speed = BiologicalMultiSpeed(128)
        self.subconscious = SubconsciousMind(128)
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        # OUTPUTS
        self.vis_out = nn.Linear(128, 32 * 32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, 2000)
        self.text_gate = nn.Linear(128, 1)

        self.growth_stage = 0

    def grow(self):
        new_layer = nn.Linear(128, 128).to(next(self.parameters()).device)
        with torch.no_grad():
            new_layer.weight.copy_(torch.eye(128));
            new_layer.bias.zero_()
        self.cortex.append(new_layer)
        self.growth_stage += 1
        return self.growth_stage

    def forward(self, img, aud_ext, aud_slf, txt_idx):
        # 1. Sensation & Binding
        v = self.eye(img)
        a_e = self.ear_ext(aud_ext)
        bound_sig, bind_score = self.binder(v, a_e)
        bind_inj = torch.cat([bound_sig, bound_sig], dim=1)

        # 2. Integration
        combined = torch.cat([v, a_e], dim=1) + (bind_inj * bind_score)
        a_s = self.self_proj(self.ear_self(aud_slf))
        t = self.text_in(txt_idx).mean(dim=1)

        h = combined + a_s + t

        # 3. Deep Processing (v2 Logic)
        h = h + self.multi_speed(h)
        h = h + (self.subconscious(h) * 0.2)

        for layer in self.cortex:
            h = F.gelu(layer(h)) + h

        # 4. Outputs
        return (torch.sigmoid(self.vis_out(h)),
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                bind_score,
                self.text_out(h),
                torch.sigmoid(self.text_gate(h)),
                h)


# ============================================================================
# 🔊 AUDIO & LOGGING
# ============================================================================

class Syrinx:
    def __init__(self):
        self.fs = 16000;
        self.ph = 0;
        self.f = 60.0;
        self.ch = False;
        self.ct = 0

    def gen(self, fr, t, c, p):
        tm = np.arange(fr) / self.fs
        self.f = 0.9 * self.f + 0.1 * (50 + t * 200)
        s = np.sin(2 * np.pi * self.f * tm + self.ph + (2 + c * 10) * np.sin(2 * np.pi * self.f * 2.5 * tm))
        self.ph += 2 * np.pi * self.f * (fr / self.fs)
        if not self.ch and p > 0.8 and random.random() < 0.05: self.ch = True; self.ct = 0
        if self.ch:
            s += np.sin(2 * np.pi * (800 + 500 * np.sin(self.ct * 20)) * tm) * 0.3
            self.ct += fr / self.fs;
            self.ch = self.ct < 0.2
        return (s * 0.2).reshape(-1, 1).astype(np.float32)


class SessionLogger:
    def __init__(self):
        self.f = "brain_log_v6.csv";
        self.t0 = time.time();
        self.l = 0
        try:
            with open(self.f, 'w', newline='') as f:
                csv.writer(f).writerow(["Time", "Event", "Loss", "Layers", "Aware"])
        except:
            pass

    def log(self, loss, lay, awr, evt=None):
        if time.time() - self.l > 1 or evt:
            self.l = time.time()
            try:
                with open(self.f, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [round(time.time() - self.t0, 2), evt if evt else "-", f"{loss:.4f}", lay, f"{awr:.2f}"])
            except:
                pass


# ============================================================================
# ⚙️ MAIN WORKER (The Soul)
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict)
    sig_grow = Signal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = OmniBrain().to(self.dev)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)

        # Sub-Systems
        self.syrinx = Syrinx()
        self.memory = EpisodicMemory()
        self.pacman = PacmanFeeder()
        self.logger = SessionLogger()

        # Buffers
        self.aq = queue.Queue()
        self.tq = queue.Queue()
        self.eff = np.zeros((1024, 1), dtype=np.float32)
        self.last_voc = torch.zeros(3)
        self.loss_hist = []
        self.vocab = ["SELF", "NOISE", "DATA", "GROW", "MAMA", "PAPA", "USER", "PACMAN", "FILE", "DREAM"]

    def run(self):
        # Start Senses
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb)
        stream.start()
        cap = cv2.VideoCapture(0)
        self.pacman.start()  # Start background reader

        while self.running:
            # 1. GATHER INPUTS
            ret, frame = cap.read()
            if not ret: continue

            # Vision
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
            v = torch.from_numpy(gray).float().flatten().unsqueeze(0).to(self.dev)

            # Audio (Ext + Self)
            try:
                raw = self.aq.get_nowait()
            except:
                raw = np.zeros((1024, 1), dtype=np.float32)
            a_e = torch.from_numpy(raw).float().mean(1).unsqueeze(0).to(self.dev)
            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.dev)

            # Text (User OR Pacman OR Dream)
            txt_src = "SILENCE"
            t_idx = 0

            # Priority 1: User Chat
            try:
                txt_str = self.tq.get_nowait()
                t_idx = hash(txt_str) % 2000
                txt_src = f"USER: {txt_str}"
            except:
                # Priority 2: Pacman File
                try:
                    pac_str = self.pacman.queue.get_nowait()
                    t_idx = hash(pac_str) % 2000
                    txt_src = "READING FILE..."
                except:
                    # Priority 3: Dreaming (Memory Replay)
                    # If silence (no audio/text), replay a memory
                    if np.linalg.norm(raw) < 0.01 and random.random() < 0.2:
                        mem = self.memory.recall()
                        if mem:
                            v, a_e, a_s, t = mem  # Overwrite sensors with memory
                            txt_src = "DREAMING..."

            t = torch.tensor([[t_idx]]).to(self.dev)

            # 2. FORWARD PASS
            self.opt.zero_grad()
            p_img, voc, awr, bnd, txt_log, gate, h = self.brain(v, a_e, a_s, t)

            loss = F.mse_loss(p_img, v)
            loss.backward()
            self.opt.step()

            self.last_voc = voc.detach().cpu().flatten()

            # 3. MEMORY STORAGE
            # If something interesting happened (High Loss or High Awareness), save it
            if loss.item() > 0.05 or awr.item() > 0.8:
                self.memory.store((v.detach(), a_e.detach(), a_s.detach(), t.detach()), loss.item())

            # 4. OUTPUT LOGIC
            ai_msg = None
            if gate.item() > 0.75 and random.random() < 0.1:
                w_id = torch.argmax(txt_log).item() % len(self.vocab)
                ai_msg = f"AI: [{self.vocab[w_id]}]"
                if txt_src == "SILENCE":
                    txt_src = ai_msg
                else:
                    txt_src += f"\n{ai_msg}"

            # 5. GROWTH
            self.loss_hist.append(loss.item())
            if len(self.loss_hist) > 50:
                avg = sum(self.loss_hist[-50:]) / 50
                if avg > 0.05 and len(self.brain.cortex) < 10:
                    lvl = self.brain.grow()
                    self.sig_grow.emit(lvl)
                    self.loss_hist = []
                else:
                    self.loss_hist.pop(0)

            # 6. UI EMIT
            self.logger.log(loss.item(), len(self.brain.cortex), awr.item(), "GROW" if ai_msg else None)

            if random.random() < 0.3:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray,
                    'dream': p_img.detach().cpu().numpy().reshape(32, 32),
                    'log': txt_src,
                    'awr': awr.item(),
                    'bnd': bnd.item(),
                    'mem': len(self.memory.memory)
                })
            time.sleep(0.01)

        cap.release();
        stream.stop();
        self.pacman.running = False

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
        c = self.col.copy();
        m = np.tile(act, 10)[:self.n] > 0.5
        c[m] = [1.0, 0.5, 0.0, 1.0];
        self.sp.setData(color=c);
        self.sp.rotate(0.2, 0, 0, 1)

    def grow(self, l):
        self.n += 500; self.make(10.0 + l * 2)


class Main(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("MAGNUM OPUS v6: OMNI-PRESENCE");
        self.resize(1200, 800)
        self.setStyleSheet("background:#000; color:#0F0; font-family:Consolas;")
        w = QWidget();
        self.setCentralWidget(w);
        l = QHBoxLayout(w)

        left = QVBoxLayout();
        self.sph = Sphere();
        left.addWidget(QLabel("CORTEX"));
        left.addWidget(self.sph)
        self.stat = QLabel("LAYERS: 1 | MEMORY: 0");
        left.addWidget(self.stat);
        l.addLayout(left, 2)

        right = QVBoxLayout()
        vis = QHBoxLayout();
        self.vr = QLabel();
        self.vd = QLabel()
        for v in [self.vr, self.vd]: v.setFixedSize(200, 200); v.setStyleSheet("border:1px solid #333")
        vis.addWidget(self.vr);
        vis.addWidget(self.vd);
        right.addLayout(vis)

        self.txt = QTextEdit();
        self.txt.setReadOnly(True);
        right.addWidget(self.txt)
        self.inp = QLineEdit();
        self.inp.returnPressed.connect(self.snd);
        right.addWidget(self.inp)

        self.ba = QProgressBar();
        self.ba.setFormat("SELF: %p%");
        self.ba.setStyleSheet("QProgressBar::chunk{background:#0AF}")
        self.bb = QProgressBar();
        self.bb.setFormat("BIND: %p%");
        self.bb.setStyleSheet("QProgressBar::chunk{background:#F0F}")
        right.addWidget(self.ba);
        right.addWidget(self.bb);
        l.addLayout(right, 1)

        self.wk = ASIWorker();
        self.wk.sig_upd.connect(self.upd);
        self.wk.sig_grow.connect(self.grw)
        self.wk.pacman.sig_read.connect(self.pac);
        self.wk.start()

    def snd(self):
        self.wk.tq.put(self.inp.text()); self.inp.clear()

    def pac(self, msg):
        self.txt.append(msg)

    def upd(self, d):
        self.sph.upd(d['act']);
        self.ba.setValue(int(d['awr'] * 100));
        self.bb.setValue(int(d['bnd'] * 100))
        self.stat.setText(f"LAYERS: {len(self.wk.brain.cortex)} | MEMORY: {d['mem']}")

        def p(a): return QPixmap.fromImage(
            QImage(cv2.resize((a * 255).astype(np.uint8), (200, 200), 0), 200, 200, QImage.Format_Grayscale8))

        self.vr.setPixmap(p(d['real']));
        self.vd.setPixmap(p(d['dream']))
        if d['log'] != "SILENCE": self.txt.append(d['log'])

    def grw(self, l):
        self.sph.grow(l); self.txt.append(f"*** GROWTH EVENT ***")

    def closeEvent(self, e):
        self.wk.running = False; self.wk.pacman.running = False; self.wk.wait(); e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = Main();
    win.show();
    sys.exit(app.exec())