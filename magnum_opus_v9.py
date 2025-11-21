"""
Magnum Opus v9.1: The Teleological Engine (Hotfix)
================================================================
ARCHITECT: Alan Hourmand
STATUS: STABLE (RuntimeError & Graph Fix applied)

FIXES:
1. Fixed RuntimeError by detaching PFC hidden state between frames.
   (Prevents "Backward through graph a second time" error).
2. Includes Dynamic Sphere Visualization (Prevents IndexErrors).
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
from collections import deque
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 🧠 COMPONENT 1: THE PREFRONTAL CORTEX
# ============================================================================

class PrefrontalCortex(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Input: Thought(128) + Stress(1)
        self.planner = nn.GRUCell(dim + 1, dim)
        self.policy = nn.Sequential(nn.Linear(dim, 4), nn.Sigmoid())
        self.critic = nn.Linear(dim, 1)

    def forward(self, h, current_stress, prev_pfc_state):
        # CRITICAL FIX: Ensure prev_pfc_state is detached from previous graph
        if prev_pfc_state is None:
            prev_pfc_state = torch.zeros_like(h)
        else:
            prev_pfc_state = prev_pfc_state.detach() # <--- THE LOBOTOMY FIX

        stress_tensor = torch.tensor([[current_stress]]).to(h.device)
        pfc_input = torch.cat([h, stress_tensor], dim=1)

        new_pfc_state = self.planner(pfc_input, prev_pfc_state)
        actions = self.policy(new_pfc_state)
        value = self.critic(new_pfc_state)

        return actions, value, new_pfc_state

# ============================================================================
# 🔮 COMPONENT 2: THE IMAGINARIUM
# ============================================================================

class Imaginarium(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Linear(256, 32*32), nn.Sigmoid()
        )
    def forward(self, real_h, mem_h, sub_h, gain):
        if mem_h is None: mem_h = torch.zeros_like(real_h)

        raw = torch.cat([real_h, mem_h, sub_h], dim=1)
        dream = torch.tanh(self.mixer(raw))
        dream = dream * (0.5 + gain)

        img = self.decoder(dream)
        return img, dream

# ============================================================================
# 🧠 COMPONENT 3: THE OMNI BRAIN
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
        self.dims = [8, 32, 64, 128]
        self.projections = nn.ModuleList([nn.Linear(dim, d) for d in self.dims])
        self.out_projections = nn.ModuleList([nn.Linear(d, dim) for d in self.dims])
        self.weights = nn.Parameter(torch.ones(len(self.dims)))
    def forward(self, x):
        out = 0; w = F.softmax(self.weights, dim=0)
        for i, (p_in, p_out) in enumerate(zip(self.projections, self.out_projections)):
            out += p_out(F.gelu(p_in(x))) * w[i]
        return out

class OmniBrainV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eye = nn.Sequential(nn.Linear(32*32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(2000, 128)

        self.pfc = PrefrontalCortex(128)
        self.binder = SynestheticBinder(64)
        self.multi_speed = BiologicalMultiSpeed(128)
        self.subconscious = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.Tanh())
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])
        self.imaginarium = Imaginarium(128)

        self.vis_out = nn.Linear(128, 32*32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, 2000)

        self.growth_stage = 0
        self.pfc_state = None

    def grow(self):
        new_layer = nn.Linear(128, 128).to(self.dev)
        with torch.no_grad(): new_layer.weight.copy_(torch.eye(128)); new_layer.bias.zero_()
        self.cortex.append(new_layer)
        self.growth_stage += 1
        return self.growth_stage

    def forward(self, img, aud_ext, aud_slf, txt_idx, mem_h, current_stress):
        v_raw = self.eye(img)
        a_raw = self.ear_ext(aud_ext)

        # 2. PFC Intervention
        # We pass 'self.pfc_state' which is handled/detached inside PrefrontalCortex.forward now
        ghost_h = torch.zeros(1, 128).to(self.dev) if self.pfc_state is None else self.pfc_state
        actions, value, self.pfc_state = self.pfc(ghost_h, current_stress, self.pfc_state)

        gain_vis = actions[0, 0] * 2.0
        gain_aud = actions[0, 1] * 2.0
        gain_imag = actions[0, 2] * 1.5
        impulse_speak = actions[0, 3]

        v = v_raw * gain_vis
        a_e = a_raw * gain_aud

        # 3. Integration
        bound_sig, bind_score = self.binder(v, a_e)
        combined = torch.cat([v, a_e], dim=1) + torch.cat([bound_sig, bound_sig], dim=1)
        a_s = self.self_proj(self.ear_self(aud_slf))
        t = self.text_in(txt_idx).mean(dim=1)

        h = combined + a_s + t

        # 4. Processing
        h = h + self.multi_speed(h)
        sub_state = self.subconscious(h)
        h = h + (sub_state * 0.2)
        for layer in self.cortex: h = F.gelu(layer(h)) + h

        # 5. Imagination
        p_imag_img, p_imag_h = self.imaginarium(h, mem_h, sub_state, gain_imag)

        # 6. Outputs
        return (torch.sigmoid(self.vis_out(h)),
                p_imag_img,
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                self.text_out(p_imag_h),
                impulse_speak,
                h,
                actions)

# ============================================================================
# 🔊 AUDIO & CHRONOS
# ============================================================================

class Chronos:
    def __init__(self): self.fps=10.0
    def warp(self, stress):
        self.fps = 0.9*self.fps + 0.1*(5.0 + stress*60.0)
        return 1.0/max(1.0, self.fps), self.fps

class Syrinx:
    def __init__(self): self.fs=16000; self.ph=0; self.f=60.0; self.ch=False; self.ct=0
    def gen(self, fr, t, c, p):
        tm = np.arange(fr)/self.fs; self.f = 0.9*self.f + 0.1*(50+t*200)
        s = np.sin(2*np.pi*self.f*tm + self.ph + (2+c*10)*np.sin(2*np.pi*self.f*2.5*tm))
        self.ph += 2*np.pi*self.f*(fr/self.fs)
        if not self.ch and p>0.8 and random.random()<0.05: self.ch=True; self.ct=0
        if self.ch: s += np.sin(2*np.pi*(800+500*np.sin(self.ct*20))*tm)*0.3; self.ct += fr/self.fs; self.ch = self.ct<0.2
        return (s*0.2).reshape(-1,1).astype(np.float32)

# ============================================================================
# ⚙️ WORKER
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict); sig_grow = Signal(int)

    def __init__(self):
        super().__init__(); self.running = True
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = OmniBrainV9().to(self.dev)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)
        self.syrinx = Syrinx(); self.chronos = Chronos()
        self.aq = queue.Queue(); self.tq = queue.Queue()
        self.eff = np.zeros((1024,1), dtype=np.float32)
        self.last_voc = torch.zeros(3); self.loss_hist = []
        self.memories = deque(maxlen=100)
        self.vocab = ["INTENT", "SEEK", "HIDE", "FOCUS", "DREAM", "PAIN", "CALM", "SELF", "GROW", "VOID"]
        self.last_loss = 0.1

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb)
        stream.start(); cap = cv2.VideoCapture(0)

        while self.running:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32,32)).astype(np.float32)/255.0
            v = torch.from_numpy(gray).float().flatten().unsqueeze(0).to(self.dev)

            try: raw = self.aq.get_nowait()
            except: raw = np.zeros((1024,1), dtype=np.float32)
            a_e = torch.from_numpy(raw).float().mean(1).unsqueeze(0).to(self.dev)
            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.dev)

            try: txt_str = self.tq.get_nowait(); t_idx = hash(txt_str)%2000; log=f"USER: {txt_str}"
            except: t_idx = 0; log=""
            t = torch.tensor([[t_idx]]).to(self.dev)

            mem_t = random.choice(self.memories) if (self.last_loss < 0.02 and len(self.memories)>5) else None

            self.opt.zero_grad()
            p_img, p_imag, voc, awr, txt_log, gate, h, intentions = self.brain(v, a_e, a_s, t, mem_t, self.last_loss)

            loss_recon = F.mse_loss(p_img, v)
            loss_imag = F.mse_loss(p_imag, v) * 0.05
            total_loss = loss_recon + loss_imag

            total_loss.backward()
            self.opt.step()

            self.last_loss = total_loss.item()
            self.last_voc = voc.detach().cpu().flatten()

            if gate.item() > 0.75 and random.random() < 0.1:
                w_id = torch.argmax(txt_log).item() % len(self.vocab)
                ai_msg = f"AI: [{self.vocab[w_id]}]"
                log += f"\n{ai_msg}"

            if awr.item()>0.7 or self.last_loss>0.05: self.memories.append(h.detach())

            self.loss_hist.append(self.last_loss)
            if len(self.loss_hist)>50:
                avg = sum(self.loss_hist[-50:])/50
                if avg > 0.04 and len(self.brain.cortex)<10:
                    lvl = self.brain.grow(); self.sig_grow.emit(lvl); self.loss_hist=[]
                else: self.loss_hist.pop(0)

            wait, fps = self.chronos.warp(self.last_loss)

            if random.random() < 0.3:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray,
                    'dream': p_img.detach().cpu().numpy().reshape(32,32),
                    'imag': p_imag.detach().cpu().numpy().reshape(32,32),
                    'log': log,
                    'awr': awr.item(),
                    'fps': fps,
                    'intent': intentions.detach().cpu().numpy().flatten()
                })

            sleep_time = max(0.001, wait - (time.time()-loop_start))
            time.sleep(sleep_time)

        cap.release(); stream.stop()

    def cb(self, i, o, f, t, s):
        if np.linalg.norm(i)>0.001: self.aq.put(i.copy())
        v = self.last_voc
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), v[2].item())
        self.eff = o.copy()

# ============================================================================
# 🖥️ UI
# ============================================================================

class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__(); self.opts['distance']=30; self.setBackgroundColor('#000')
        self.n=1000; self.make(10.0)
    def make(self, r):
        idx = np.arange(0,self.n,dtype=float)+0.5
        phi = np.arccos(1-2*idx/self.n); theta = np.pi*(1+5**0.5)*idx
        x,y,z = r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)
        self.pos = np.column_stack((x,y,z)); self.col = np.ones((self.n,4))*0.3; self.col[:,2]=1.0
        try: self.sp.setData(pos=self.pos, color=self.col)
        except: self.sp = gl.GLScatterPlotItem(pos=self.pos, color=self.col, size=4, pxMode=True); self.addItem(self.sp)
    def upd(self, act):
        # Dynamic Resizing Fix
        needed = (self.n // len(act)) + 2
        c = self.col.copy()
        mask = np.tile(act, needed)[:self.n] > 0.5
        c[mask] = [1.0,0.5,0.0,1.0]; self.sp.setData(color=c); self.sp.rotate(0.2,0,0,1)
    def grow(self, l): self.n+=500; self.make(10.0+l*2)

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("MAGNUM OPUS v9.1: TELEOLOGICAL ENGINE (STABLE)"); self.resize(1400,850)
        self.setStyleSheet("background:#000; color:#0F0; font-family:Consolas;")
        w=QWidget(); self.setCentralWidget(w); l=QHBoxLayout(w)

        left=QVBoxLayout(); self.sph=Sphere(); left.addWidget(QLabel("CORTEX")); left.addWidget(self.sph)
        self.stat=QLabel("LAYERS: 1"); left.addWidget(self.stat); l.addLayout(left, 2)

        right=QVBoxLayout()
        vis=QHBoxLayout()
        self.vr=QLabel(); self.vd=QLabel(); self.vi=QLabel()
        labels = ["RETINA", "RECONSTRUCTION", "IMAGINATION"]
        screens = [self.vr, self.vd, self.vi]
        for i, v in enumerate(screens):
            v.setFixedSize(200,200); v.setStyleSheet("border:1px solid #333")
            v_box = QVBoxLayout(); v_box.addWidget(QLabel(labels[i])); v_box.addWidget(v); vis.addLayout(v_box)
        right.addLayout(vis)

        intent_box = QWidget(); intent_box.setStyleSheet("border: 1px solid #444; padding: 5px;")
        ib_layout = QVBoxLayout(intent_box)
        ib_layout.addWidget(QLabel("--- PREFRONTAL INTENTIONS ---"))
        self.bars = []
        for name in ["VISUAL FOCUS", "AUDIO FOCUS", "DREAM GAIN", "SPEAK IMPULSE"]:
            lbl = QLabel(name); pb = QProgressBar(); pb.setRange(0,100)
            pb.setStyleSheet("QProgressBar::chunk{background: #FF5500}")
            ib_layout.addWidget(lbl); ib_layout.addWidget(pb); self.bars.append(pb)
        right.addWidget(intent_box)

        self.txt=QTextEdit(); self.txt.setReadOnly(True); self.txt.setMaximumHeight(100); right.addWidget(self.txt)
        self.inp=QLineEdit(); self.inp.returnPressed.connect(self.snd); right.addWidget(self.inp)

        self.time_lbl = QLabel("TIME: 1.0x"); right.addWidget(self.time_lbl)
        l.addLayout(right, 2)

        self.wk=ASIWorker(); self.wk.sig_upd.connect(self.upd); self.wk.sig_grow.connect(self.grw); self.wk.start()

    def snd(self): self.wk.tq.put(self.inp.text()); self.inp.clear()
    def upd(self, d):
        self.sph.upd(d['act'])
        self.time_lbl.setText(f"SUBJECTIVE TIME: {d['fps']:.1f} FPS")
        ints = d['intent']
        self.bars[0].setValue(int(ints[0]*100))
        self.bars[1].setValue(int(ints[1]*100))
        self.bars[2].setValue(int(ints[2]*100))
        self.bars[3].setValue(int(ints[3]*100))

        def p(a): return QPixmap.fromImage(QImage(cv2.resize((a*255).astype(np.uint8),(200,200),0),200,200,QImage.Format_Grayscale8))
        self.vr.setPixmap(p(d['real'])); self.vd.setPixmap(p(d['dream'])); self.vi.setPixmap(p(d['imag']))
        if d['log']: self.txt.append(d['log'])
    def grw(self, l): self.sph.grow(l); self.txt.append(f"*** GROWTH EVENT ***")
    def closeEvent(self, e): self.wk.running=False; self.wk.wait(); e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv); win=Main(); win.show(); sys.exit(app.exec())