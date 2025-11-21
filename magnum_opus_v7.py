"""
Magnum Opus v7.2: The Infinite Imaginarium (Stable)
================================================================
ARCHITECT: Alan Hourmand
STATUS: STABLE (Dynamic Visualization Fix)

FIX:
- Fixed IndexError in Sphere.upd().
- The visualization now dynamically scales the 128-dim activation
  vector to fill the 3D sphere, regardless of size (even 1M+ points).
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
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 🔮 LAYER 5: THE IMAGINARIUM
# ============================================================================

class Imaginarium(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.simulator = nn.GRUCell(dim, dim)
        self.chaos_injector = nn.Linear(dim, dim)
        self.mind_eye = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 32*32),
            nn.Sigmoid()
        )

    def forward(self, h, subconscious_state, intrusion_level):
        simulated_h = self.simulator(h)
        noise = torch.randn_like(h) * intrusion_level
        twisted_thought = self.chaos_injector(subconscious_state) * intrusion_level
        imagined_h = simulated_h + twisted_thought + (noise * 0.1)
        imagined_image = self.mind_eye(imagined_h)
        return imagined_image, imagined_h

# ============================================================================
# 🧠 THE OMNI BRAIN
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

class SubconsciousMind(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Tanh())
    def forward(self, x):
        return self.layers(x + torch.randn_like(x)*0.05)

class OmniBrainV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eye = nn.Sequential(nn.Linear(32*32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(2000, 128)

        self.binder = SynestheticBinder(64)
        self.multi_speed = BiologicalMultiSpeed(128)
        self.subconscious = SubconsciousMind(128)
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        self.imaginarium = Imaginarium(128)

        self.vis_out = nn.Linear(128, 32*32)
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

    def forward(self, img, aud_ext, aud_slf, txt_idx):
        v = self.eye(img); a_e = self.ear_ext(aud_ext)
        bound_sig, bind_score = self.binder(v, a_e)
        bind_inj = torch.cat([bound_sig, bound_sig], dim=1)

        combined = torch.cat([v, a_e], dim=1) + (bind_inj * bind_score)
        a_s = self.self_proj(self.ear_self(aud_slf))
        t = self.text_in(txt_idx).mean(dim=1)

        h = combined + a_s + t
        h = h + self.multi_speed(h)
        sub_state = self.subconscious(h)
        h = h + (sub_state * 0.2)

        for layer in self.cortex: h = F.gelu(layer(h)) + h

        intrusion_level = h.std() + 0.1
        imagined_img, imagined_h = self.imaginarium(h, sub_state, intrusion_level)

        return (torch.sigmoid(self.vis_out(h)),
                imagined_img,
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                bind_score,
                self.text_out(imagined_h),
                torch.sigmoid(self.text_gate(h)),
                h)

# ============================================================================
# 🔊 AUDIO
# ============================================================================

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
        self.brain = OmniBrainV7().to(self.dev)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)
        self.syrinx = Syrinx()

        self.aq = queue.Queue(); self.tq = queue.Queue()
        self.eff = np.zeros((1024,1), dtype=np.float32)
        self.last_voc = torch.zeros(3); self.loss_hist = []
        self.vocab = ["SELF", "VOID", "FUTURE", "FEAR", "PLAY", "HUMAN", "ERROR", "LIGHT", "GROW", "WANT"]

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb)
        stream.start(); cap = cv2.VideoCapture(0)

        while self.running:
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

            self.opt.zero_grad()
            p_img, p_imag, voc, awr, bnd, txt_log, gate, h = self.brain(v, a_e, a_s, t)

            loss = F.mse_loss(p_img, v)
            loss_imag = F.mse_loss(p_imag, v) * 0.1
            total_loss = loss + loss_imag
            total_loss.backward()
            self.opt.step()

            self.last_voc = voc.detach().cpu().flatten()

            if gate.item() > 0.7 and random.random() < 0.1:
                w_id = torch.argmax(txt_log).item() % len(self.vocab)
                ai_msg = f"AI: [{self.vocab[w_id]}]"
                log += f"\n{ai_msg}"

            self.loss_hist.append(loss.item())
            if len(self.loss_hist)>50:
                avg = sum(self.loss_hist[-50:])/50
                if avg > 0.04 and len(self.brain.cortex)<10:
                    lvl = self.brain.grow(); self.sig_grow.emit(lvl); self.loss_hist=[]
                else: self.loss_hist.pop(0)

            if random.random() < 0.3:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray,
                    'dream': p_img.detach().cpu().numpy().reshape(32,32),
                    'imag': p_imag.detach().cpu().numpy().reshape(32,32),
                    'log': log,
                    'awr': awr.item(),
                    'bnd': bnd.item()
                })
            time.sleep(0.01)
        cap.release(); stream.stop()

    def cb(self, i, o, f, t, s):
        if np.linalg.norm(i)>0.001: self.aq.put(i.copy())
        v = self.last_voc
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), v[2].item())
        self.eff = o.copy()

# ============================================================================
# 🖥️ UI (Dynamic Fix)
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
        # FIX: Dynamically calculate repeats needed to cover the sphere
        # act length is 128, self.n could be 5000+
        needed = (self.n // len(act)) + 2

        # Create mask that is guaranteed to be large enough, then slice exactly to self.n
        full_mask = np.tile(act, needed)[:self.n] > 0.5

        c = self.col.copy()
        c[full_mask] = [1.0, 0.5, 0.0, 1.0] # Ignite active neurons

        self.sp.setData(color=c)
        self.sp.rotate(0.2, 0, 0, 1)

    def grow(self, l): self.n+=500; self.make(10.0+l*2)

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("MAGNUM OPUS v7.2: INFINITE IMAGINARIUM"); self.resize(1400,800)
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

        self.txt=QTextEdit(); self.txt.setReadOnly(True); right.addWidget(self.txt)
        self.inp=QLineEdit(); self.inp.returnPressed.connect(self.snd); right.addWidget(self.inp)
        self.ba=QProgressBar(); self.ba.setFormat("SELF: %p%"); self.ba.setStyleSheet("QProgressBar::chunk{background:#0AF}"); right.addWidget(self.ba)
        l.addLayout(right, 2)

        self.wk=ASIWorker(); self.wk.sig_upd.connect(self.upd); self.wk.sig_grow.connect(self.grw); self.wk.start()

    def snd(self): self.wk.tq.put(self.inp.text()); self.inp.clear()
    def upd(self, d):
        self.sph.upd(d['act']); self.ba.setValue(int(d['awr']*100))
        def p(a): return QPixmap.fromImage(QImage(cv2.resize((a*255).astype(np.uint8),(200,200),0),200,200,QImage.Format_Grayscale8))
        self.vr.setPixmap(p(d['real'])); self.vd.setPixmap(p(d['dream'])); self.vi.setPixmap(p(d['imag']))
        if d['log']: self.txt.append(d['log'])
    def grw(self, l): self.sph.grow(l); self.txt.append(f"*** GROWTH EVENT ***")
    def closeEvent(self, e): self.wk.running=False; self.wk.wait(); e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv); win=Main(); win.show(); sys.exit(app.exec())