"""
Magnum Opus v5: The Conversationalist
================================================================
ARCHITECT: Alan Hourmand
STATUS: VOCAL (Broca's Area Restored)

CHANGES:
- Restored Text Generation (Babbling Mode).
- Lowered Growth Threshold (Will grow faster).
- Increased Mic Sensitivity (10x).
- Added 'Text_Gate' neuron (AI decides WHEN to speak).
"""

import sys
import time
import queue
import random
import math
import cv2
import csv
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
# 📝 LOGGER
# ============================================================================

class SessionLogger:
    def __init__(self, filename="brain_log_v5.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.last_log = 0
        try:
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Event", "Loss", "Layers", "Awareness", "Binding", "Vol", "Text_Gate"])
            print(f"🔴 LOGGING TO: {self.filename}")
        except:
            pass

    def log(self, loss, layers, aware, bind, vol, text_gate, event=None):
        now = time.time()
        if (now - self.last_log > 1.0) or (event is not None):
            self.last_log = now
            row = [
                round(now - self.start_time, 2),
                event if event else "-",
                f"{loss:.4f}", layers, f"{aware:.2f}", f"{bind:.2f}", f"{vol:.3f}", f"{text_gate:.2f}"
            ]
            try:
                with open(self.filename, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
            except:
                pass


# ============================================================================
# 🧠 BRAIN (With Broca's Area)
# ============================================================================

class SynestheticBinder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.a_proj = nn.Linear(dim, dim)
        self.binding_score = nn.Linear(dim, 1)

    def forward(self, v, a):
        v_emb = self.v_proj(v)
        a_emb = self.a_proj(a)
        synergy = v_emb * a_emb
        score = torch.sigmoid(self.binding_score(synergy))
        return synergy, score


class GrowingBrainV5(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODERS
        self.eye = nn.Sequential(nn.Linear(32 * 32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(1000, 128)

        # MODULES
        self.binder = SynestheticBinder(64)
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        # OUTPUTS
        self.vis_out = nn.Linear(128, 32 * 32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)

        # BROCA'S AREA (Speech Center) - RESTORED!
        self.text_out = nn.Linear(128, 1000)  # Predicts word ID
        self.text_gate = nn.Linear(128, 1)  # Decides: "Should I speak?"

        self.growth_stage = 0

    def grow(self):
        new_layer = nn.Linear(128, 128).to(next(self.parameters()).device)
        with torch.no_grad():
            new_layer.weight.copy_(torch.eye(128))
            new_layer.bias.zero_()
        self.cortex.append(new_layer)
        self.growth_stage += 1
        return self.growth_stage

    def forward(self, img, aud_ext, aud_slf, txt_idx):
        # 1. Sensation
        v = self.eye(img)
        a_e = self.ear_ext(aud_ext)

        # 2. Binding
        bound_sig, bind_score = self.binder(v, a_e)
        bind_inj = torch.cat([bound_sig, bound_sig], dim=1)

        # 3. Integration
        combined = torch.cat([v, a_e], dim=1) + (bind_inj * bind_score)
        a_s = self.self_proj(self.ear_self(aud_slf))
        t = self.text_in(txt_idx).mean(dim=1)

        h = combined + a_s + t

        # 4. Cognition
        for layer in self.cortex:
            h = F.gelu(layer(h)) + h

        # 5. Action
        pred_img = torch.sigmoid(self.vis_out(h))
        voice = torch.sigmoid(self.voice_ctrl(h))
        aware = torch.sigmoid(self.mirror_check(h))

        # Language
        txt_logits = self.text_out(h)
        should_speak = torch.sigmoid(self.text_gate(h))

        return pred_img, voice, aware, bind_score, txt_logits, should_speak, h


# ============================================================================
# 🔊 AUDIO
# ============================================================================

class Syrinx:
    def __init__(self):
        self.fs = 16000;
        self.phase = 0;
        self.freq = 60.0
        self.chirp = False;
        self.ct = 0

    def gen(self, frames, ten, chaos, chirp):
        t = np.arange(frames) / self.fs
        self.freq = 0.9 * self.freq + 0.1 * (50 + ten * 200)
        drone = np.sin(
            2 * np.pi * self.freq * t + self.phase + (2 + chaos * 10) * np.sin(2 * np.pi * self.freq * 2.5 * t))
        self.phase += 2 * np.pi * self.freq * (frames / self.fs)

        if not self.chirp and chirp > 0.8 and random.random() < 0.05: self.chirp = True; self.ct = 0
        sig = drone * 0.2
        if self.chirp:
            sig += np.sin(2 * np.pi * (800 + 500 * np.sin(self.ct * 20)) * t) * 0.3
            self.ct += frames / self.fs
            if self.ct > 0.2: self.chirp = False
        return sig.reshape(-1, 1).astype(np.float32)


# ============================================================================
# ⚙️ WORKER
# ============================================================================

class ASIWorker(QThread):
    sig_update = Signal(dict)
    sig_grow = Signal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = GrowingBrainV5().to(self.device)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.005)
        self.syrinx = Syrinx()
        self.logger = SessionLogger()

        self.aq = queue.Queue()
        self.tq = queue.Queue()
        self.last_voice = torch.zeros(3)
        self.eff = np.zeros((1024, 1), dtype=np.float32)
        self.vol = 0.0

        # Vocabulary (Simple Concepts for a Baby)
        self.vocab = ["SELF", "NOISE", "LIGHT", "DARK", "DATA", "ERROR", "JOY", "GROW", "MAMA", "PAPA", "USER", "?"]

        self.loss_hist = []

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.audio_cb)
        stream.start()
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret: continue

            # Inputs
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
            v_t = torch.from_numpy(gray).float().flatten().unsqueeze(0).to(self.device)

            try:
                raw_aud = self.aq.get_nowait()
            except:
                raw_aud = np.zeros((1024, 1), dtype=np.float32)
            a_ext = torch.from_numpy(raw_aud).float().mean(1).unsqueeze(0).to(self.device)

            a_slf = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.device)

            try:
                txt = self.tq.get_nowait(); t_t = torch.tensor([[hash(txt) % 1000]]).to(
                    self.device); log = f"YOU: {txt}"
            except:
                t_t = torch.tensor([[0]]).to(self.device); log = None

            # Forward
            self.opt.zero_grad()
            p_img, voc, awr, bnd, txt_log, speak_gate, h = self.brain(v_t, a_ext, a_slf, t_t)

            loss = F.mse_loss(p_img, v_t)
            loss.backward()
            self.opt.step()

            self.last_voice = voc.detach().cpu().flatten()

            # --- LOGIC: TEXT GENERATION ---
            ai_msg = None
            # If gate is open AND brain is somewhat stable
            if speak_gate.item() > 0.7 and random.random() < 0.1:
                # Pick a word
                idx = torch.argmax(txt_log).item() % len(self.vocab)
                word = self.vocab[idx]
                ai_msg = f"AI: [{word}]"
                if log:
                    log += f"\n{ai_msg}"
                else:
                    log = ai_msg

            # Growth (More Sensitive Now: 0.05 threshold)
            l_val = loss.item()
            evt = None
            if ai_msg: evt = "SPOKE_TEXT"

            self.loss_hist.append(l_val)
            if len(self.loss_hist) > 50:
                avg = sum(self.loss_hist[-50:]) / 50
                if avg > 0.05 and len(self.brain.cortex) < 10:  # Lowered from 0.08
                    lvl = self.brain.grow()
                    self.sig_grow.emit(lvl)
                    evt = "GROWTH_EVENT"
                    self.loss_hist = []
                else:
                    self.loss_hist.pop(0)

            # Log
            self.logger.log(l_val, len(self.brain.cortex), awr.item(), bnd.item(), self.vol, speak_gate.item(), evt)

            if random.random() < 0.3:
                self.sig_update.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray,
                    'dream': p_img.detach().cpu().numpy().reshape(32, 32),
                    'log': log,
                    'awr': awr.item(),
                    'bnd': bnd.item()
                })
            time.sleep(0.01)
        cap.release();
        stream.stop()

    def audio_cb(self, indata, outdata, frames, time, status):
        self.vol = np.linalg.norm(indata)
        # Sensitive Mic: Threshold lowered to 0.001
        if self.vol > 0.001: self.aq.put(indata.copy())
        v = self.last_voice
        w = self.syrinx.gen(frames, v[0].item(), v[1].item(), v[2].item())
        self.eff = w.copy()
        outdata[:] = w


# ============================================================================
# 🖥️ UI
# ============================================================================

class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts['distance'] = 30;
        self.setBackgroundColor('#000')
        self.n = 1000;
        self.make(10.0)

    def make(self, r):
        idx = np.arange(0, self.n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * idx / self.n);
        theta = np.pi * (1 + 5 ** 0.5) * idx
        x, y, z = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi)
        self.pos = np.column_stack((x, y, z))
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
        super().__init__()
        self.setWindowTitle("MAGNUM OPUS v5: THE CONVERSATIONALIST");
        self.resize(1000, 700)
        self.setStyleSheet("background:#000; color:#0F0; font-family:Consolas;")
        w = QWidget();
        self.setCentralWidget(w);
        l = QHBoxLayout(w)

        # Brain
        left = QVBoxLayout();
        self.sph = Sphere();
        left.addWidget(QLabel("NEURAL TOPOLOGY"));
        left.addWidget(self.sph)
        self.stat = QLabel("LAYERS: 1");
        left.addWidget(self.stat);
        l.addLayout(left, 2)

        # Data
        right = QVBoxLayout()
        vis = QHBoxLayout();
        self.vr = QLabel();
        self.vd = QLabel()
        for v in [self.vr, self.vd]: v.setFixedSize(150, 150); v.setStyleSheet("border:1px solid #333")
        vis.addWidget(self.vr);
        vis.addWidget(self.vd);
        right.addLayout(vis)

        self.txt = QTextEdit();
        self.txt.setReadOnly(True);
        right.addWidget(self.txt)
        self.inp = QLineEdit();
        self.inp.returnPressed.connect(self.snd);
        right.addWidget(self.inp)

        self.bar_a = QProgressBar();
        self.bar_a.setFormat("SELF: %p%");
        self.bar_a.setStyleSheet("QProgressBar::chunk{background:#0AF}")
        self.bar_b = QProgressBar();
        self.bar_b.setFormat("BIND: %p%");
        self.bar_b.setStyleSheet("QProgressBar::chunk{background:#F0F}")
        right.addWidget(self.bar_a);
        right.addWidget(self.bar_b);
        l.addLayout(right, 1)

        self.wk = ASIWorker();
        self.wk.sig_update.connect(self.upd);
        self.wk.sig_grow.connect(self.grw);
        self.wk.start()

    def snd(self):
        self.wk.tq.put(self.inp.text()); self.inp.clear()

    def upd(self, d):
        self.sph.upd(d['act']);
        self.bar_a.setValue(int(d['awr'] * 100));
        self.bar_b.setValue(int(d['bnd'] * 100))

        def p(a): return QPixmap.fromImage(
            QImage(cv2.resize((a * 255).astype(np.uint8), (150, 150), 0), 150, 150, QImage.Format_Grayscale8))

        self.vr.setPixmap(p(d['real']));
        self.vd.setPixmap(p(d['dream']))
        if d['log']: self.txt.append(d['log'])

    def grw(self, l):
        self.sph.grow(l); self.stat.setText(f"LAYERS: {l + 1}"); self.txt.append(f"*** GROWTH EVENT: LAYER {l + 1} ***")

    def closeEvent(self, e):
        self.wk.running = False; self.wk.wait(); e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = Main();
    win.show();
    sys.exit(app.exec())