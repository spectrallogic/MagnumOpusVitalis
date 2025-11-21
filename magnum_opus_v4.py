"""
Magnum Opus v4.5: Black Box Recorder (Hotfix)
================================================================
ARCHITECT: Alan Hourmand
STATUS: STABLE (Fixed IndexError)

FIXES:
- Fixed Tensor Indexing Error in Event Detection.
- Flattened voice vector before logic checks.
"""

import sys
import time
import queue
import random
import math
import cv2
import csv
import datetime
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
# 📝 DIAGNOSTIC LOGGER
# ============================================================================

class SessionLogger:
    def __init__(self, filename="brain_log.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.last_log = 0

        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time_Sec", "Event_Tag", "Loss_Entropy", "Layers",
                "Self_Awareness", "Binding_Score",
                "Voice_Tension", "Voice_Chirp", "Input_Vol"
            ])
        print(f"🔴 RECORDING STARTED: {self.filename}")

    def log(self, loss, layers, aware, bind, voice_tens, voice_chirp, vol, event=None):
        now = time.time()
        if (now - self.last_log > 1.0) or (event is not None):
            self.last_log = now
            elapsed = round(now - self.start_time, 2)
            tag = event if event else "-"

            row = [
                elapsed, tag, f"{loss:.4f}", layers,
                f"{aware:.2f}", f"{bind:.2f}",
                f"{voice_tens:.2f}", f"{voice_chirp:.2f}", f"{vol:.3f}"
            ]
            try:
                with open(self.filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except: pass

# ============================================================================
# 🧠 BRAIN
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

class BiologicalMultiSpeed(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dims = [8, 16, 32, 64, 96, 128, 192, 256]
        self.projections = nn.ModuleList([nn.Linear(input_dim, d) for d in self.dims])
        self.out_projections = nn.ModuleList([nn.Linear(d, input_dim) for d in self.dims])
        self.trust_weights = nn.Parameter(torch.tensor([5.0, 3.0, 1.0, 0.1, 0.01, 0.0, 0.0, 0.0]))

    def forward(self, x):
        speed_outputs = []
        for proj_in, proj_out in zip(self.projections, self.out_projections):
            thought = F.gelu(proj_in(x))
            result = proj_out(thought)
            speed_outputs.append(result)
        stacked = torch.stack(speed_outputs, dim=0)
        weights = F.softmax(self.trust_weights, dim=0)
        return torch.einsum('kbd,k->bd', stacked, weights)

class SubconsciousMind(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l0 = nn.Linear(dim, dim)
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, dim)

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1
        s0 = F.gelu(self.l0(x) + noise)
        s1 = torch.sigmoid(self.l1(s0)) * s0
        s2 = F.gelu(self.l2(s1))
        s3 = torch.sigmoid(self.l3(s2)) * s2
        return s3

class GrowingSynestheticBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_dim = 128

        self.eye = nn.Sequential(nn.Linear(32*32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text = nn.Embedding(1000, 128)

        self.binder = SynestheticBinder(64)
        self.multi_speed = BiologicalMultiSpeed(128)
        self.subconscious = SubconsciousMind(128)
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        self.vis_out = nn.Linear(128, 32*32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)

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
        v = self.eye(img)
        a_e = self.ear_ext(aud_ext)

        bound_signal, binding_score = self.binder(v, a_e)
        binding_injection = torch.cat([bound_signal, bound_signal], dim=1)

        combined = torch.cat([v, a_e], dim=1)
        combined = combined + (binding_injection * binding_score)

        a_s = self.ear_self(aud_slf)
        self_injection = self.self_proj(a_s)

        t = self.text(txt_idx).mean(dim=1)
        h = combined + t + self_injection

        h = h + self.multi_speed(h)
        intuition = self.subconscious(h)
        h = h + (intuition * 0.2)

        for layer in self.cortex:
            h = F.gelu(layer(h)) + h

        pred_img = torch.sigmoid(self.vis_out(h))
        voice = torch.sigmoid(self.voice_ctrl(h))
        aware = torch.sigmoid(self.mirror_check(h))

        return pred_img, voice, aware, binding_score, h

# ============================================================================
# 🔊 SYRINX
# ============================================================================

class SyrinxSynthesizer:
    def __init__(self):
        self.fs = 16000
        self.phase = 0
        self.freq = 60.0
        self.chirp_active = False
        self.chirp_t = 0

    def generate(self, frames, tension, chaos, chirp_prob):
        t = np.arange(frames) / self.fs
        target_f = 50 + (tension * 200)
        self.freq = 0.9 * self.freq + 0.1 * target_f
        mod = 2 + (chaos * 10)
        drone = np.sin(2*np.pi*self.freq*t + self.phase + mod*np.sin(2*np.pi*self.freq*2.5*t))
        self.phase += 2*np.pi*self.freq*(frames/self.fs)

        if not self.chirp_active and chirp_prob > 0.8 and random.random() < 0.1:
            self.chirp_active = True
            self.chirp_t = 0

        signal = drone * 0.2
        if self.chirp_active:
            cf = 800 + 500*np.sin(self.chirp_t * 20)
            signal += np.sin(2*np.pi*cf*t) * 0.3
            self.chirp_t += (frames/self.fs)
            if self.chirp_t > 0.2: self.chirp_active = False

        return signal.reshape(-1, 1).astype(np.float32)

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
        self.brain = GrowingSynestheticBrain().to(self.device)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.005)
        self.syrinx = SyrinxSynthesizer()
        self.logger = SessionLogger()

        self.audio_q = queue.Queue()
        self.text_q = queue.Queue()
        self.last_voice = torch.zeros(3)
        self.efference_copy = np.zeros((1024, 1), dtype=np.float32)

        self.retina_buffer = None
        self.echoic_buffer = None
        self.loss_history = []
        self.current_vol = 0.0

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.audio_cb)
        stream.start()
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32,32)).astype(np.float32) / 255.0
            if self.retina_buffer is None: self.retina_buffer = gray
            else: self.retina_buffer = (self.retina_buffer * 0.3) + (gray * 0.7)
            v_t = torch.from_numpy(self.retina_buffer).float().flatten().unsqueeze(0).to(self.device)

            try: raw_aud = self.audio_q.get_nowait()
            except: raw_aud = np.zeros((1024, 1), dtype=np.float32)

            if self.echoic_buffer is None: self.echoic_buffer = raw_aud
            else: self.echoic_buffer = (self.echoic_buffer * 0.2) + (raw_aud * 0.8)
            a_ext = torch.from_numpy(self.echoic_buffer).float().mean(1).unsqueeze(0).to(self.device)

            a_slf = torch.from_numpy(self.efference_copy).float().mean(1).unsqueeze(0).to(self.device)

            try: txt = self.text_q.get_nowait(); t_t = torch.tensor([[hash(txt)%1000]]).to(self.device); log=f"USER: {txt}"
            except: t_t = torch.tensor([[0]]).to(self.device); log=None

            # Brain
            self.opt.zero_grad()
            pred_img, voice, aware, binding, h = self.brain(v_t, a_ext, a_slf, t_t)

            loss = F.mse_loss(pred_img, v_t)
            loss.backward()
            self.opt.step()

            # [FIX] Update last_voice BEFORE using it for checks
            self.last_voice = voice.detach().cpu().flatten()

            # Event Detection
            event_tag = None
            if aware.item() > 0.8: event_tag = "SELF_RECOGNIZED"
            if binding.item() > 0.8: event_tag = "SENSES_BOUND"
            # Now we use self.last_voice which is guaranteed to be 1D
            if self.last_voice[2].item() > 0.8: event_tag = "ATTEMPT_COMM (CHIRP)"
            if log: event_tag = "USER_INPUT"

            # Growth
            l_val = loss.item()
            self.loss_history.append(l_val)
            if len(self.loss_history) > 50:
                avg_loss = sum(self.loss_history[-50:])/50
                if avg_loss > 0.08 and len(self.brain.cortex) < 10:
                    lvl = self.brain.grow()
                    self.sig_grow.emit(lvl)
                    event_tag = "GROWTH_EVENT"
                    self.loss_history = []
                else:
                    self.loss_history.pop(0)

            # LOG DATA
            self.logger.log(
                loss=l_val,
                layers=len(self.brain.cortex),
                aware=aware.item(),
                bind=binding.item(),
                voice_tens=self.last_voice[0].item(),
                voice_chirp=self.last_voice[2].item(),
                vol=self.current_vol,
                event=event_tag
            )

            if random.random() < 0.3:
                self.sig_update.emit({
                    'loss': l_val,
                    'act': h.detach().cpu().numpy().flatten(),
                    'voice': self.last_voice,
                    'aware': aware.item(),
                    'binding': binding.item(),
                    'real': self.retina_buffer,
                    'dream': pred_img.detach().cpu().numpy().reshape(32,32),
                    'chat': log
                })

            time.sleep(0.01)

        cap.release(); stream.stop()

    def audio_cb(self, indata, outdata, frames, time, status):
        self.current_vol = np.linalg.norm(indata)
        if self.current_vol > 0.02: self.audio_q.put(indata.copy())
        v = self.last_voice
        wave = self.syrinx.generate(frames, v[0].item(), v[1].item(), v[2].item())
        self.efference_copy = wave.copy()
        outdata[:] = wave

# ============================================================================
# 🖥️ UI
# ============================================================================

class ExpandingSphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts['distance'] = 30
        self.setBackgroundColor('#000')
        self.n_points = 1000
        self.make_sphere(10.0)

    def make_sphere(self, radius):
        indices = np.arange(0, self.n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/self.n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)
        self.pos = np.column_stack((x, y, z))
        self.colors = np.ones((self.n_points, 4)) * 0.3
        self.colors[:, 2] = 1.0

        try: self.scatter.setData(pos=self.pos, color=self.colors)
        except:
            self.scatter = gl.GLScatterPlotItem(pos=self.pos, color=self.colors, size=4, pxMode=True)
            self.addItem(self.scatter)

    def update_state(self, act):
        c = self.colors.copy()
        mask = np.tile(act, 10)[:self.n_points] > 0.5
        c[mask] = [1.0, 0.5, 0.0, 1.0]
        self.scatter.setData(color=c)
        self.scatter.rotate(0.2, 0, 0, 1)

    def grow_visuals(self, level):
        self.n_points += 500
        self.make_sphere(10.0 + (level * 2))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAGNUM OPUS v4.5: BLACK BOX (STABLE)")
        self.resize(1200, 800)
        self.setStyleSheet("background: #000; color: #0F0; font-family: Consolas;")

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left
        left = QWidget(); l_lay = QVBoxLayout(left)
        self.brain = ExpandingSphere()
        l_lay.addWidget(QLabel("CORTICAL TOPOLOGY"))
        l_lay.addWidget(self.brain)
        self.stats = QLabel("LAYERS: 1")
        l_lay.addWidget(self.stats)
        splitter.addWidget(left)

        # Right
        right = QWidget(); r_lay = QVBoxLayout(right)
        v_lay = QHBoxLayout()
        self.v_real = QLabel(); self.v_real.setFixedSize(200,200); self.v_real.setStyleSheet("border:1px solid #333")
        self.v_dream = QLabel(); self.v_dream.setFixedSize(200,200); self.v_dream.setStyleSheet("border:1px solid #333")
        v_lay.addWidget(self.v_real); v_lay.addWidget(self.v_dream)
        r_lay.addLayout(v_lay)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.inp = QLineEdit(); self.inp.returnPressed.connect(self.send)
        r_lay.addWidget(self.log); r_lay.addWidget(self.inp)

        # BARS
        self.aware = QProgressBar()
        self.aware.setFormat("SELF-AWARENESS: %p%")
        self.aware.setStyleSheet("QProgressBar::chunk{background: #00AAFF}")
        r_lay.addWidget(self.aware)

        self.binding = QProgressBar()
        self.binding.setFormat("A/V BINDING: %p%")
        self.binding.setStyleSheet("QProgressBar::chunk{background: #FF00FF}")
        r_lay.addWidget(self.binding)

        rec = QLabel("🔴 REC"); rec.setStyleSheet("color: red; font-weight: bold;")
        r_lay.addWidget(rec)

        splitter.addWidget(right)
        splitter.setSizes([800, 400])

        self.worker = ASIWorker()
        self.worker.sig_update.connect(self.upd)
        self.worker.sig_grow.connect(self.on_grow)
        self.worker.start()

    def send(self):
        self.worker.text_q.put(self.inp.text()); self.inp.clear()

    def upd(self, d):
        self.brain.update_state(d['act'])
        self.aware.setValue(int(d['aware']*100))
        self.binding.setValue(int(d['binding']*100))

        def pix(a): return QPixmap.fromImage(QImage(cv2.resize((a*255).astype(np.uint8),(200,200),interpolation=0),200,200,QImage.Format_Grayscale8))
        self.v_real.setPixmap(pix(d['real']))
        self.v_dream.setPixmap(pix(d['dream']))

        if d['chat']: self.log.append(d['chat'])

    def on_grow(self, level):
        self.brain.grow_visuals(level)
        self.stats.setText(f"LAYERS: {level+1} (GROWTH EVENT)")
        self.log.append(f"*** EXPANSION: LAYER {level+1} ADDED ***")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())