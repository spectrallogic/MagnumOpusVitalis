"""
Magnum Opus v13.1: The Heavy Industry (Hotfix)
================================================================
ARCHITECT: Alan Hourmand
STATUS: STABLE (Fixed NameError)

FIXES:
- Corrected 'gray' vs 'gray_live' variable mismatch.
- Verified Audio Engine (Titan Drone + Datastream).
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
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar, QSplitter, QFrame)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 🧠 1. BIOLOGICAL CORE
# ============================================================================

class PrefrontalCortex(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.planner = nn.GRUCell(dim + 1, dim)
        self.policy = nn.Sequential(nn.Linear(dim, 4), nn.Sigmoid())

    def forward(self, h, stress, prev_state):
        if prev_state is None: prev_state = torch.zeros_like(h)
        else: prev_state = prev_state.detach()

        stress_t = torch.tensor([[stress]]).to(h.device)
        inp = torch.cat([h, stress_t], dim=1)
        new_state = self.planner(inp, prev_state)
        actions = self.policy(new_state)
        return actions, new_state

class Imaginarium(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 32*32), nn.Sigmoid()
        )
    def forward(self, real, mem, sub, gain):
        if mem is None: mem = torch.zeros_like(real)
        raw = torch.cat([real, mem, sub], dim=1)
        dream = torch.tanh(self.mixer(raw))
        dream = dream * (0.5 + gain)
        img = self.decoder(dream)
        return img, dream

class OmniBrainV13(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eye = nn.Sequential(nn.Linear(32*32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(3000, 128)

        self.pfc = PrefrontalCortex(128)
        self.imaginarium = Imaginarium(128)
        self.subconscious = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.Tanh())
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        self.vis_out = nn.Linear(128, 32*32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, 3000)
        self.text_gate = nn.Linear(128, 1)

        self.pfc_state = None

    def grow(self):
        l = nn.Linear(128, 128).to(self.dev)
        with torch.no_grad(): l.weight.copy_(torch.eye(128)); l.bias.zero_()
        self.cortex.append(l)
        return len(self.cortex)

    def forward(self, img, aud, aud_s, txt, mem, stress):
        v = self.eye(img); a = self.ear_ext(aud)

        if self.pfc_state is None: ghost_h = torch.zeros(1, 128).to(self.dev)
        else: ghost_h = self.pfc_state.detach()

        actions, self.pfc_state = self.pfc(ghost_h, stress, self.pfc_state)
        gain_vis, gain_aud, gain_imag, impulse_speak = actions[0]

        v = v * (1.0 + gain_vis)
        a = a * (1.0 + gain_aud)

        combined = torch.cat([v, a], dim=1)
        a_self = self.self_proj(self.ear_self(aud_s))
        t = self.text_in(txt).mean(dim=1)

        h = combined + a_self + t

        sub = self.subconscious(h)
        h = h + (sub * 0.2)
        for l in self.cortex: h = F.gelu(l(h)) + h

        p_imag_img, p_imag_h = self.imaginarium(h, mem, sub, gain_imag)

        return (torch.sigmoid(self.vis_out(h)),
                p_imag_img,
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                self.text_out(p_imag_h),
                torch.sigmoid(self.text_gate(h)) + (impulse_speak * 0.5),
                h,
                actions)

# ============================================================================
# 🔊 2. SYRINX V2
# ============================================================================

class SyrinxV2:
    def __init__(self):
        self.fs = 16000; self.phase = 0; self.base_freq = 55.0
        self.mod_phase = 0; self.data_phase = 0

    def gen(self, frames, tension, chaos, speak_impulse):
        t = np.arange(frames) / self.fs
        target_freq = 55.0 + (tension * 55.0)
        self.base_freq = 0.95 * self.base_freq + 0.05 * target_freq

        throb_rate = 2.0 + (tension * 8.0)
        throb = np.sin(2 * np.pi * throb_rate * t + self.mod_phase)
        self.mod_phase += 2 * np.pi * throb_rate * (frames/self.fs)

        carrier = np.tanh(5.0 * np.sin(2 * np.pi * self.base_freq * t + self.phase))
        drone = carrier * (0.5 + 0.3 * throb)
        self.phase += 2 * np.pi * self.base_freq * (frames/self.fs)

        datastream = np.zeros_like(drone)
        if speak_impulse > 0.6:
            data_freq = 800.0 + 400.0 * np.round(np.sin(2 * np.pi * (10 + chaos*20) * t + self.data_phase))
            datastream = np.sign(np.sin(2 * np.pi * data_freq * t)) * 0.3
            self.data_phase += 2 * np.pi * 20 * (frames/self.fs)

        master_mix = (drone * 0.3) + datastream
        return np.clip(master_mix, -0.9, 0.9).reshape(-1, 1).astype(np.float32)

# ============================================================================
# 📂 3. SUPPORT SYSTEMS
# ============================================================================

class Pacman(QThread):
    sig_read = Signal(str)
    def __init__(self):
        super().__init__(); self.running = True; self.q = queue.Queue()
    def run(self):
        seen = set()
        while self.running:
            if os.path.exists("training_data"):
                for f in glob.glob("training_data/*.txt"):
                    if f not in seen:
                        try:
                            with open(f,'r', encoding='utf-8') as fl:
                                for c in fl.read().split(): self.q.put(c)
                            self.sig_read.emit(f"PACMAN: Digested {os.path.basename(f)}")
                            seen.add(f)
                        except: pass
            time.sleep(2)

class Chronos:
    def __init__(self): self.fps = 10.0
    def warp(self, stress):
        target = 5.0 + (stress * 100.0)
        self.fps = 0.9*self.fps + 0.1*target
        return 1.0/max(1.0, self.fps), self.fps

# ============================================================================
# ⚙️ 4. THE WORKER
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict); sig_grow = Signal(int)

    def __init__(self):
        super().__init__(); self.running = True
        self.brain = OmniBrainV13().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)
        self.syrinx = SyrinxV2(); self.chronos = Chronos(); self.pacman = Pacman()

        self.aq = queue.Queue(); self.tq = queue.Queue()
        self.eff = np.zeros((1024,1), dtype=np.float32)
        self.last_voc = torch.zeros(3); self.loss_hist = []

        self.memories = deque(maxlen=100)
        self.vocab = ["SELF", "WAIT", "LOOK", "HEAR", "FEEL", "VOID", "DATA", "GROW", "DREAM", "HUMAN", "TIME", "SYNC"]
        self.last_loss = 0.1

    def run(self):
        st = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb); st.start()
        cap = cv2.VideoCapture(0)
        self.pacman.start()

        while self.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32,32)).astype(np.float32)/255.0
            v_live = torch.from_numpy(gray).float().flatten().unsqueeze(0).to(self.brain.dev)

            try: r_aud = self.aq.get_nowait()
            except: r_aud = np.zeros((1024,1), dtype=np.float32)
            a_live = torch.from_numpy(r_aud).float().mean(1).unsqueeze(0).to(self.brain.dev)

            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.brain.dev)

            txt_src = "SILENCE"; t_idx = 0
            try:
                t_str = self.tq.get_nowait(); t_idx = hash(t_str)%3000; txt_src = f"USER: {t_str}"
            except:
                try: p_str = self.pacman.q.get_nowait(); t_idx = hash(p_str)%3000; txt_src = "READING..."
                except: pass
            t_live = torch.tensor([[t_idx]]).to(self.brain.dev)

            is_dreaming = False
            target_mode = "[LIVE STREAM]"
            v_in, a_in, t_in = v_live, a_live, t_live
            v_target = v_live
            mem_tensor = None

            if self.last_loss < 0.02 and len(self.memories) > 5 and random.random() < 0.2:
                is_dreaming = True
                target_mode = "[HINDSIGHT REPLAY]"
                txt_src = "REPLAYING MEMORY..."
                mem_v, mem_a, mem_t, mem_h = random.choice(self.memories)
                v_in, a_in, t_in = mem_v, mem_a, mem_t
                mem_tensor = mem_h
                v_target = mem_v

            self.opt.zero_grad()
            p_real, p_imag, voc, awr, t_log, gate, h, intents = self.brain(v_in, a_in, a_s, t_in, mem_tensor, self.last_loss)

            loss = F.mse_loss(p_real, v_target) + (F.mse_loss(p_imag, v_target) * 0.1)
            loss.backward()
            self.opt.step()

            self.last_loss = loss.item(); self.last_voc = voc.detach().cpu().flatten()

            if not is_dreaming and (awr.item() > 0.7 or self.last_loss > 0.05):
                self.memories.append((v_live.detach(), a_live.detach(), t_live.detach(), h.detach()))

            impulse = intents[0, 3].item()
            if (gate.item() + impulse) > 1.0 and random.random() < 0.15:
                wd = self.vocab[torch.argmax(t_log).item() % len(self.vocab)]
                ai_msg = f"AI: [{wd}]"
                txt_src = ai_msg if txt_src=="SILENCE" else txt_src + f"\n{ai_msg}"

            self.loss_hist.append(loss.item())
            if len(self.loss_hist)>50:
                avg = sum(self.loss_hist[-50:])/50
                if avg > 0.04 and len(self.brain.cortex)<15:
                    lvl = self.brain.grow(); self.sig_grow.emit(lvl); self.loss_hist=[]
                else: self.loss_hist.pop(0)

            wait, fps = self.chronos.warp(self.last_loss)
            if random.random() < 0.3:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray, # <--- FIXED VARIABLE NAME
                    'dream': p_real.detach().cpu().numpy().reshape(32,32),
                    'imag': p_imag.detach().cpu().numpy().reshape(32,32),
                    'log': txt_src, 'awr': awr.item(), 'fps': fps,
                    'intent': intents.detach().cpu().numpy().flatten(),
                    'mode': target_mode
                })

            time.sleep(max(0.001, wait - (time.time()-t0)))

        cap.release(); st.stop(); self.pacman.running=False

    def cb(self, i, o, f, t, s):
        if np.linalg.norm(i)>0.001: self.aq.put(i.copy())
        v = self.last_voc
        speak_gate = v[2].item()
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), speak_gate)
        self.eff = o.copy()

# ============================================================================
# 🖥️ 4. THE INTERFACE
# ============================================================================

class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__(); self.opts['distance']=30; self.setBackgroundColor('#050505')
        self.n=1000; self.make(10.0)
    def make(self, r):
        idx = np.arange(0,self.n,dtype=float)+0.5
        phi = np.arccos(1-2*idx/self.n); theta = np.pi*(1+5**0.5)*idx
        x,y,z = r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)
        self.pos = np.column_stack((x,y,z)); self.col = np.ones((self.n,4))*0.3; self.col[:,2]=1.0
        try: self.sp.setData(pos=self.pos, color=self.col)
        except: self.sp = gl.GLScatterPlotItem(pos=self.pos, color=self.col, size=4, pxMode=True); self.addItem(self.sp)
    def upd(self, act, mode):
        needed = (self.n // len(act)) + 2
        c = self.col.copy(); mask = np.tile(act, needed)[:self.n]>0.5
        if mode == "[HINDSIGHT REPLAY]": c[mask] = [0.8, 0.0, 1.0, 1.0]
        else: c[mask] = [0.0, 1.0, 0.8, 1.0]
        self.sp.setData(color=c); self.sp.rotate(0.2,0,0,1)
    def grow(self, l): self.n+=500; self.make(10.0+l*2)

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("MAGNUM OPUS v13.1: HEAVY INDUSTRY"); self.resize(1400,900)
        self.setStyleSheet("background:#111; color:#0F0; font-family:Consolas; font-size: 10pt;")

        splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(splitter)

        left = QWidget(); lv = QVBoxLayout(left)
        self.sph = Sphere(); lv.addWidget(QLabel(":: NEURAL TOPOLOGY ::")); lv.addWidget(self.sph)
        self.stat = QLabel("SYSTEM: ONLINE"); lv.addWidget(self.stat)
        splitter.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)

        scr = QHBoxLayout()
        self.vr = QLabel(); self.vd = QLabel(); self.vi = QLabel()
        lbls = ["RETINA", "CORTEX", "DREAM"]
        for i, l in enumerate([self.vr, self.vd, self.vi]):
            l.setScaledContents(True); l.setFixedSize(200,200); l.setStyleSheet("border: 1px solid #333; background: #000")
            box = QVBoxLayout(); box.addWidget(QLabel(lbls[i])); box.addWidget(l); scr.addLayout(box)
        rv.addLayout(scr, 2)

        ibox = QFrame(); ibox.setStyleSheet("background: #1a1a1a; border: 1px solid #333; padding: 5px;")
        il = QVBoxLayout(ibox); il.addWidget(QLabel(":: PREFRONTAL INTENTIONS ::"))
        self.bars = []
        for t in ["VISUAL GAIN", "AUDIO GAIN", "DREAM GAIN", "SPEAK IMPULSE"]:
            l = QHBoxLayout(); l.addWidget(QLabel(t))
            b = QProgressBar(); b.setRange(0,100); b.setStyleSheet("QProgressBar::chunk{background: #FF6600}")
            l.addWidget(b); il.addLayout(l); self.bars.append(b)
        rv.addWidget(ibox, 1)

        self.txt = QTextEdit(); self.txt.setReadOnly(True); rv.addWidget(self.txt, 1)
        self.inp = QLineEdit(); self.inp.returnPressed.connect(self.snd); self.inp.setPlaceholderText("Talk to the AI..."); rv.addWidget(self.inp)

        self.mode_lbl = QLabel("LEARNING SOURCE: [LIVE STREAM]")
        self.mode_lbl.setStyleSheet("font-weight: bold; color: #FFF; font-size: 12pt;")
        rv.addWidget(self.mode_lbl)

        splitter.addWidget(right); splitter.setSizes([600, 800])

        self.wk = ASIWorker(); self.wk.sig_upd.connect(self.upd); self.wk.sig_grow.connect(self.grw)
        self.wk.pacman.sig_read.connect(self.pac); self.wk.start()

    def snd(self): self.wk.tq.put(self.inp.text()); self.inp.clear()
    def pac(self, m): self.txt.append(m)
    def upd(self, d):
        self.sph.upd(d['act'], d['mode'])
        self.mode_lbl.setText(f"SOURCE: {d['mode']} | FPS: {d['fps']:.1f}")
        if d['mode'] == "[HINDSIGHT REPLAY]": self.mode_lbl.setStyleSheet("color: #FF00FF; font-weight: bold; font-size: 12pt;")
        else: self.mode_lbl.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 12pt;")

        self.stat.setText(f"LAYERS: {len(self.wk.brain.cortex)} | AWARENESS: {d['awr']:.2f}")

        ints = d['intent']
        for i, b in enumerate(self.bars): b.setValue(int(ints[i]*100))

        def pix(a): return QPixmap.fromImage(QImage(cv2.resize((a*255).astype(np.uint8),(200,200),0),200,200,QImage.Format_Grayscale8))
        self.vr.setPixmap(pix(d['real'])); self.vd.setPixmap(pix(d['dream'])); self.vi.setPixmap(pix(d['imag']))
        if d['log'] != "SILENCE": self.txt.append(d['log'])

    def grw(self, l): self.sph.grow(l); self.txt.append("*** GROWTH EVENT ***")
    def closeEvent(self, e): self.wk.running=False; self.wk.pacman.running=False; self.wk.wait(); e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv); win=Main(); win.show(); sys.exit(app.exec())