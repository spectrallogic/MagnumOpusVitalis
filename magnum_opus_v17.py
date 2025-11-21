"""
Magnum Opus v18: The Mirror Mind
================================================================
ARCHITECT: Alan Hourmand
STATUS: RESONANT LEARNING (Echoes & Mimicry)

CHANGES:
- TABULA RASA: Starts empty (except 'HELLO'). No hardcoded "VOID" lists.
- ECHO REFLEX: Immediately biases output towards words it just heard or read.
  (This mimics a baby repeating what it hears to learn it).
- STRESS TIMING: High stress = more babbling. Low stress = quiet observation.
- ASSOCIATIVE CHAINING: Tries to string 1-3 words together if excited.
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
import re
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar, QSplitter, QFrame, QGridLayout)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl

# ============================================================================
# 📝 0. BLACK BOX RECORDER
# ============================================================================

class SessionLogger:
    def __init__(self, filename="brain_log_final.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.last_log = 0
        try:
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Event", "Loss", "Layers", "Awareness", "FPS", "VisGain", "AudGain", "DreamGain"])
        except: pass

    def log(self, loss, layers, aware, fps, intents, event=None):
        now = time.time()
        if (now - self.last_log > 1.0) or (event is not None):
            self.last_log = now
            row = [round(now-self.start_time,2), event if event else "-", f"{loss:.4f}", layers, f"{aware:.2f}", f"{fps:.1f}",
                   f"{intents[0]:.2f}", f"{intents[1]:.2f}", f"{intents[2]:.2f}"]
            try:
                with open(self.filename, mode='a', newline='') as f: csv.writer(f).writerow(row)
            except: pass

# ============================================================================
# 🧠 1. THE BIOLOGICAL CORE
# ============================================================================

class PrefrontalCortex(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.planner = nn.GRUCell(dim + 1, dim)
        self.policy = nn.Sequential(nn.Linear(dim, 4), nn.Sigmoid())
    def forward(self, h, stress, prev):
        if prev is None: prev = torch.zeros_like(h)
        else: prev = prev.detach()
        stress_t = torch.tensor([[stress]]).to(h.device)
        inp = torch.cat([h, stress_t], dim=1)
        new_s = self.planner(inp, prev)
        return self.policy(new_s), new_s

class Imaginarium(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 32*32), nn.Sigmoid())
    def forward(self, real, mem, sub, gain):
        if mem is None: mem = torch.zeros_like(real)
        raw = torch.cat([real, mem, sub], dim=1)
        dream = torch.tanh(self.mixer(raw)) * (0.5 + gain)
        return self.decoder(dream), dream

class OmniBrainFinal(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size

        self.eye = nn.Sequential(nn.Linear(32*32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(vocab_size, 128)

        self.pfc = PrefrontalCortex(128)
        self.imaginarium = Imaginarium(128)
        self.subconscious = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.Tanh())
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        self.vis_out = nn.Linear(128, 32*32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, vocab_size)
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

        v = v * (1.0 + gain_vis); a = a * (1.0 + gain_aud)
        combined = torch.cat([v, a], dim=1)
        a_self = self.self_proj(self.ear_self(aud_s))
        t = self.text_in(txt).mean(dim=1)
        h = combined + a_self + t

        sub = self.subconscious(h)
        h = h + (sub * 0.2)
        for l in self.cortex: h = F.gelu(l(h)) + h

        p_imag_img, p_imag_h = self.imaginarium(h, mem, sub, gain_imag)

        return (torch.sigmoid(self.vis_out(h)), p_imag_img, torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)), self.text_out(p_imag_h),
                torch.sigmoid(self.text_gate(h)) + (impulse_speak * 0.5), h, actions)

# ============================================================================
# 🔊 2. AUDIO & TIME
# ============================================================================

class SyrinxV2:
    def __init__(self):
        self.fs=16000; self.phase=0; self.bf=55.0; self.mp=0; self.dp=0
    def gen(self, fr, ten, chs, imp):
        t = np.arange(fr)/self.fs; target = 55.0+(ten*55.0); self.bf = 0.95*self.bf + 0.05*target
        tr = 2.0+(ten*8.0); throb = np.sin(2*np.pi*tr*t + self.mp); self.mp += 2*np.pi*tr*(fr/self.fs)
        car = np.tanh(5.0*np.sin(2*np.pi*self.bf*t + self.phase)); dr = car*(0.5+0.3*throb); self.phase += 2*np.pi*self.bf*(fr/self.fs)
        ds = np.zeros_like(dr)
        if imp>0.6:
            df = 800.0+400.0*np.round(np.sin(2*np.pi*(10+chs*20)*t+self.dp))
            ds = np.sign(np.sin(2*np.pi*df*t))*0.4; self.dp+=2*np.pi*20*(fr/self.fs)
        return np.clip((dr*0.3)+ds, -0.9, 0.9).reshape(-1,1).astype(np.float32)

class Pacman(QThread):
    sig_read = Signal(str)
    def __init__(self): super().__init__(); self.running=True; self.q=queue.Queue()
    def run(self):
        seen=set()
        while self.running:
            if os.path.exists("training_data"):
                for f in glob.glob("training_data/*.txt"):
                    if f not in seen:
                        try:
                            with open(f,'r', encoding='utf-8') as fl:
                                content = fl.read()
                                words = re.findall(r'\b\w+\b', content.upper())
                                for i in range(0, len(words), 5):
                                    chunk = " ".join(words[i:i+5])
                                    self.q.put(chunk)
                            self.sig_read.emit(f"PACMAN: Ingested {os.path.basename(f)}"); seen.add(f)
                        except: pass
            time.sleep(2)

class Chronos:
    def __init__(self): self.fps=10.0
    def warp(self, s):
        tgt = 5.0 + (s*100.0); self.fps = 0.9*self.fps + 0.1*tgt
        return 1.0/max(1.0, self.fps), self.fps

# ============================================================================
# ⚙️ 3. WORKER
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict); sig_grow = Signal(int)
    def __init__(self):
        super().__init__(); self.running=True
        # Dynamic Vocabulary
        self.max_vocab = 5000
        self.word2idx = {}
        self.idx2word = []

        # Clean Slate with minimal instinct
        self.learn_text("HELLO")

        self.brain = OmniBrainFinal(self.max_vocab).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.004)
        self.syrinx = SyrinxV2(); self.chronos = Chronos(); self.pacman = Pacman()
        self.logger = SessionLogger()
        self.aq=queue.Queue(); self.tq=queue.Queue(); self.eff=np.zeros((1024,1),dtype=np.float32)
        self.last_voc=torch.zeros(3); self.loss_hist=[]; self.memories=deque(maxlen=100)
        self.last_loss=0.1; self.vol=0.0; self.last_user_input=""
        self.recent_concepts = deque(maxlen=10)

    def learn_text(self, text):
        """Dynamically adds new words to vocabulary."""
        words = re.findall(r'\b\w+\b', text.upper())
        new_indices = []
        for w in words:
            if w not in self.word2idx:
                if len(self.idx2word) < self.max_vocab:
                    self.word2idx[w] = len(self.idx2word)
                    self.idx2word.append(w)
            if w in self.word2idx:
                new_indices.append(self.word2idx[w])
        return new_indices

    def run(self):
        st = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb); st.start()
        cap = cv2.VideoCapture(0); self.pacman.start()
        while self.running:
            t0 = time.time(); ret, frame = cap.read()
            if not ret: continue

            gray_live = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32,32)).astype(np.float32)/255.0
            v_live = torch.from_numpy(gray_live).float().flatten().unsqueeze(0).to(self.brain.dev)

            try: r_aud=self.aq.get_nowait()
            except: r_aud=np.zeros((1024,1), dtype=np.float32)
            a_live = torch.from_numpy(r_aud).float().mean(1).unsqueeze(0).to(self.brain.dev)
            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.brain.dev)

            txt_src="SILENCE"; t_indices=[0]; pac_active=False; pac_text=None

            # Process Inputs & Learn Words
            try:
                t_str = self.tq.get_nowait()
                txt_src = f"USER: {t_str}"; self.last_user_input = t_str.lower()
                t_indices = self.learn_text(t_str)
                # Boost recent concepts for echo effect
                for idx in t_indices: self.recent_concepts.append(idx)
            except:
                try:
                    pac_text = self.pacman.q.get_nowait(); pac_active=True
                    t_indices = self.learn_text(pac_text)
                    for idx in t_indices: self.recent_concepts.append(idx)
                except: pass

            if not t_indices: t_indices = [0]
            t_live = torch.tensor([t_indices]).to(self.brain.dev)

            is_dream=False; tgt_mode="[LIVE STREAM]"; v_in, a_in, t_in = v_live, a_live, t_live
            v_tgt = v_live; mem_t = None

            if not pac_active and self.last_loss < 0.02 and len(self.memories)>5 and random.random()<0.2:
                is_dream=True; tgt_mode="[HINDSIGHT REPLAY]"; txt_src="REPLAYING MEMORY..."
                mem_v, mem_a, mem_t_real, mem_h = random.choice(self.memories)
                v_in, a_in, t_in = mem_v, mem_a, mem_t_real; mem_t = mem_h; v_tgt = mem_v

            self.opt.zero_grad()
            p_real, p_imag, voc, awr, t_log, gate, h, intents = self.brain(v_in, a_in, a_s, t_in, mem_t, self.last_loss)
            loss = F.mse_loss(p_real, v_tgt) + (F.mse_loss(p_imag, v_tgt)*0.1)
            loss.backward(); self.opt.step()

            self.last_loss=loss.item(); self.last_voc=voc.detach().cpu().flatten()

            if not is_dream and (awr.item()>0.7 or self.last_loss>0.05):
                self.memories.append((v_live.detach(), a_live.detach(), t_live.detach(), h.detach()))

            impulse = intents[0,3].item(); evt_tag=None; ai_msg=None

            # --- RESONANT MIMICRY ---
            # Bias logits towards recent concepts (Short-term semantic echo)
            for idx in self.recent_concepts:
                t_log[0, idx] += 2.0

            # --- DYNAMIC GATING ---
            # High stress = Lower threshold (Talkative/Panic)
            # Low stress = Higher threshold (Quiet/Contemplative)
            # Base threshold 0.8
            threshold = 0.8 - (self.last_loss * 5.0)
            if pac_active: threshold -= 0.2 # Talk more when reading

            current_drive = gate.item() + impulse

            if current_drive > threshold:
                valid_len = len(self.idx2word)
                if valid_len < self.max_vocab:
                    t_log[0, valid_len:] = -float('inf')

                probs = F.softmax(t_log, dim=1)
                # Sample 1-3 words for chain reaction
                num_words = 1
                if current_drive > (threshold + 0.3): num_words = random.randint(1, 3)

                words_out = []
                for _ in range(num_words):
                    try:
                        w_id = torch.multinomial(probs, 1).item()
                        if w_id < len(self.idx2word):
                            words_out.append(self.idx2word[w_id])
                    except: pass

                if words_out:
                    ai_msg = f"AI: [{' '.join(words_out)}]"
                    evt_tag = "SPEAK"

            self.loss_hist.append(loss.item())
            if len(self.loss_hist)>50:
                avg = sum(self.loss_hist[-50:])/50
                if avg>0.04 and len(self.brain.cortex)<15:
                    lvl=self.brain.grow(); self.sig_grow.emit(lvl); self.loss_hist=[]; evt_tag="GROWTH"
                else: self.loss_hist.pop(0)

            wait, fps = self.chronos.warp(self.last_loss)
            self.logger.log(self.last_loss, len(self.brain.cortex), awr.item(), fps, intents.detach().cpu().numpy().flatten(), evt_tag)

            if random.random()<0.3 or pac_active or ai_msg:
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray_live, 'dream': p_real.detach().cpu().numpy().reshape(32,32), 'imag': p_imag.detach().cpu().numpy().reshape(32,32),
                    'log': txt_src, 'ai_msg': ai_msg, 'pac_msg': pac_text,
                    'awr': awr.item(), 'fps': fps, 'intent': intents.detach().cpu().numpy().flatten(), 'mode': tgt_mode, 'vol': self.vol
                })
            time.sleep(max(0.001, wait-(time.time()-t0)))
        cap.release(); st.stop(); self.pacman.running=False

    def cb(self, i, o, f, t, s):
        self.vol = np.linalg.norm(i)
        if self.vol>0.001: self.aq.put(i.copy())
        v = self.last_voc; gate = v[2].item()
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), gate); self.eff=o.copy()

# ============================================================================
# 🖥️ 4. UI
# ============================================================================

class SciFiPanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("background: rgba(0, 20, 30, 180); border: 1px solid #00FFFF; border-radius: 6px;")
        l = QVBoxLayout(self); l.setContentsMargins(4,4,4,4)
        lbl = QLabel(title); lbl.setStyleSheet("color: #00FFFF; font-family: 'Segoe UI'; font-weight: bold; font-size: 10pt; border: none; background: none;")
        l.addWidget(lbl); self.content = l

class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__(); self.opts['distance']=40; self.setBackgroundColor('#00050A')
        self.n=1000; self.make(10.0); self.phase=0.0
    def make(self, r):
        idx=np.arange(0,self.n,dtype=float)+0.5; phi=np.arccos(1-2*idx/self.n); theta=np.pi*(1+5**0.5)*idx
        x,y,z = r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)
        self.base_pos=np.column_stack((x,y,z)); self.pos=self.base_pos.copy(); self.col=np.ones((self.n,4))*0.3; self.col[:,2]=1.0
        try: self.sp.setData(pos=self.pos, color=self.col)
        except: self.sp=gl.GLScatterPlotItem(pos=self.pos, color=self.col, size=4, pxMode=True); self.addItem(self.sp)
    def upd(self, act, mode, vol):
        self.phase+=0.2; nd=(self.n//len(act))+2; mask=np.tile(act, nd)[:self.n]>0.5
        r=np.linalg.norm(self.base_pos, axis=1); wav=np.sin(r*0.5 - self.phase)*0.5+0.5
        self.pos = self.base_pos * (1.0 + vol*2.0)
        c=self.col.copy(); c[:,3] = 0.3+(wav*0.2)
        if mode == "[HINDSIGHT REPLAY]": c[mask] = [0.8, 0.0, 1.0, 1.0]
        else: c[mask] = [0.0, 1.0, 0.8, 1.0]
        if vol>0.1: am=np.random.rand(self.n)<(vol*0.5); c[am]=[1.0,0.2,0.0,1.0]
        self.sp.setData(pos=self.pos, color=c); self.sp.rotate(0.2,0,0,1)
    def grow(self, l): self.n+=500; self.make(10.0+l*2)

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("MAGNUM OPUS v18: THE MIRROR MIND"); self.resize(1600,900)
        self.setStyleSheet("background-color: #050505; font-family: 'Consolas'; color: #00FF9D;")
        self.showMaximized()
        cw = QWidget(); self.setCentralWidget(cw); main_lay = QHBoxLayout(cw)

        left = SciFiPanel(":: NEURAL CORE ::")
        self.sph = Sphere(); left.content.addWidget(self.sph, 1)
        self.stats = QLabel("SYSTEM: INITIALIZING..."); self.stats.setStyleSheet("color: #00FF9D; font-size: 11pt; border: none;")
        left.content.addWidget(self.stats)
        main_lay.addWidget(left, 2)

        right = QWidget(); r_lay = QVBoxLayout(right); r_lay.setContentsMargins(0,0,0,0)
        vis_panel = SciFiPanel(":: OPTICAL ARRAY ::")
        h_vis = QHBoxLayout()
        self.screens = []
        for t in ["RETINA", "CORTEX", "DREAM"]:
            v = QLabel(); v.setFixedSize(220, 220); v.setStyleSheet("border: 1px solid #004455; background: #000;")
            v.setScaledContents(True); box = QVBoxLayout(); box.addWidget(QLabel(t)); box.addWidget(v); h_vis.addLayout(box); self.screens.append(v)
        vis_panel.content.addLayout(h_vis)
        r_lay.addWidget(vis_panel)

        tel_panel = SciFiPanel(":: PREFRONTAL TELEMETRY ::")
        self.bars = []
        grid = QGridLayout()
        labels = ["VISUAL GAIN", "AUDIO GAIN", "DREAM GAIN", "SPEAK IMPULSE"]
        colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#FF3300"]
        for i, txt in enumerate(labels):
            l = QLabel(txt); l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            b = QProgressBar(); b.setRange(0,100); b.setFixedHeight(15)
            b.setStyleSheet(f"QProgressBar {{ background: #111; border: none; }} QProgressBar::chunk {{ background: {colors[i]}; }}")
            grid.addWidget(l, i//2, (i%2)*2); grid.addWidget(b, i//2, (i%2)*2+1); self.bars.append(b)
        tel_panel.content.addLayout(grid)
        r_lay.addWidget(tel_panel)

        split = QHBoxLayout()
        mat_panel = SciFiPanel(":: RAW DATA INGESTION ::")
        self.matrix_txt = QTextEdit(); self.matrix_txt.setReadOnly(True)
        self.matrix_txt.setStyleSheet("background: #000500; color: #005500; border: none; font-size: 8pt; font-family: 'Courier New';")
        mat_panel.content.addWidget(self.matrix_txt)
        split.addWidget(mat_panel, 1)

        chat_panel = SciFiPanel(":: COMM LINK ::")
        self.chat_txt = QTextEdit(); self.chat_txt.setReadOnly(True)
        self.chat_txt.setStyleSheet("background: #001122; color: #00FF9D; border: none; font-size: 11pt; font-weight: bold;")
        self.inp = QLineEdit(); self.inp.returnPressed.connect(self.snd); self.inp.setStyleSheet("background: #002233; color: #FFF; border: 1px solid #005577; padding: 5px;")
        self.inp.setPlaceholderText("TRANSMIT DATA TO AI...")
        chat_panel.content.addWidget(self.chat_txt); chat_panel.content.addWidget(self.inp)
        split.addWidget(chat_panel, 2)

        r_lay.addLayout(split)
        main_lay.addWidget(right, 3)

        self.wk = ASIWorker(); self.wk.sig_upd.connect(self.upd); self.wk.sig_grow.connect(self.grw)
        self.wk.pacman.sig_read.connect(self.pac); self.wk.start()

    def snd(self): self.wk.tq.put(self.inp.text()); self.chat_txt.append(f"<span style='color:#FFF'>YOU: {self.inp.text()}</span>"); self.inp.clear()
    def pac(self, m): pass
    def upd(self, d):
        self.sph.upd(d['act'], d['mode'], d['vol'])
        mode_col = "#FF00FF" if "HINDSIGHT" in d['mode'] else "#00FFFF"
        self.stats.setText(f"LAYERS: {len(self.wk.brain.cortex)} | AWARENESS: {d['awr']:.2f} | TIME: <span style='color:{mode_col}'>{d['mode']}</span> ({d['fps']:.1f} FPS)")
        for i, b in enumerate(self.bars): b.setValue(int(d['intent'][i]*100))
        def px(a): return QPixmap.fromImage(QImage(cv2.resize((a*255).astype(np.uint8),(220,220),0),220,220,QImage.Format_Grayscale8))
        self.screens[0].setPixmap(px(d['real'])); self.screens[1].setPixmap(px(d['dream'])); self.screens[2].setPixmap(px(d['imag']))
        if d['pac_msg']: self.matrix_txt.append(f"{d['pac_msg']}")
        if d['ai_msg']: self.chat_txt.append(f"<span style='color:#00FF00'>{d['ai_msg']}</span>")
        if d['log'].startswith("REPLAY"): self.chat_txt.append(f"<span style='color:#555'><i>{d['log']}</i></span>")
    def grw(self, l): self.sph.grow(l); self.chat_txt.append("<b style='color:#FF3300'>*** CORTICAL EXPANSION DETECTED ***</b>")
    def closeEvent(self, e): self.wk.running=False; self.wk.pacman.running=False; self.wk.wait(); e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv); win=Main(); win.show(); sys.exit(app.exec())