"""
Magnum Opus v20: Genesis (The Infant ASI)
================================================================
ARCHITECT: Alan Hourmand
STATUS: TABULA RASA (Infant Stage)

VISION (from GuidelinesASI.txt):
- AGE 0: Starts with almost ZERO vocabulary.
- MIMICRY: Implements 'Echo Reflex' to learn words by repeating them.
- PLASTICITY: High learning rate (Infant Brain) that decays over time.
- CURIOSITY: Uses v19 WebCrawler to "look around" at simple concepts.
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
import requests
from bs4 import BeautifulSoup
from collections import deque
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QProgressBar, QFrame, QGridLayout)
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
import pyqtgraph.opengl as gl


# ============================================================================
# 🌐 0. CURIOSITY ENGINE (WEB CRAWLER)
# ============================================================================

class WebCrawler(QThread):
    sig_read = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.q = queue.Queue()
        self.query_q = queue.Queue()
        self.visited = set()
        # Baby ASI starts with simple, fundamental concepts
        self.curiosity_list = ["Color", "Shape", "Sound", "Light", "Human", "Face", "Voice"]

    def run(self):
        while self.running:
            try:
                try:
                    topic = self.query_q.get(timeout=5)
                except queue.Empty:
                    if random.random() < 0.1:  # Low attention span initially
                        topic = random.choice(self.curiosity_list)
                    else:
                        continue

                if topic in self.visited: continue

                self.sig_read.emit(f"🌐 DISCOVERING: {topic}...")

                # Simple scraping logic (Wikipedia)
                url = f"https://en.wikipedia.org/wiki/{topic}"
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, 'html.parser')
                        paras = soup.find_all('p')
                        content = " ".join([p.get_text() for p in paras[:3]])  # Short attention span
                        content = re.sub(r'\[.*?\]', '', content)

                        # Feed chunks to the brain
                        words = re.findall(r'\b\w+\b', content.upper())
                        if words:
                            self.visited.add(topic)
                            # Feed in small bursts
                            for i in range(0, min(len(words), 50), 5):
                                chunk = " ".join(words[i:i + 5])
                                self.q.put(chunk)
                                time.sleep(0.5)
                except:
                    pass
            except:
                pass
            time.sleep(3)


# ============================================================================
# 🧠 1. BIOLOGICAL CORE (Genesis Edition)
# ============================================================================

class PrefrontalCortex(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.planner = nn.GRUCell(dim + 1, dim)
        # Output: [VisGain, AudGain, DreamGain, BabbleImpulse]
        self.policy = nn.Sequential(nn.Linear(dim, 4), nn.Sigmoid())

    def forward(self, h, stress, prev):
        if prev is None:
            prev = torch.zeros_like(h)
        else:
            prev = prev.detach()
        stress_t = torch.tensor([[stress]]).to(h.device)
        inp = torch.cat([h, stress_t], dim=1)
        new_s = self.planner(inp, prev)
        return self.policy(new_s), new_s


class Imaginarium(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = nn.Linear(dim * 3, dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 32 * 32), nn.Sigmoid())

    def forward(self, real, mem, sub, gain):
        if mem is None: mem = torch.zeros_like(real)
        raw = torch.cat([real, mem, sub], dim=1)
        dream = torch.tanh(self.mixer(raw)) * (0.5 + gain)
        return self.decoder(dream), dream


class OmniBrainGenesis(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sensory Inputs
        self.eye = nn.Sequential(nn.Linear(32 * 32, 64), nn.Tanh())
        self.ear_ext = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        self.ear_self = nn.Sequential(nn.Linear(1024, 32), nn.Tanh())  # Proprioception
        self.self_proj = nn.Linear(32, 128)
        self.text_in = nn.Embedding(vocab_size, 128)

        # Brain Components
        self.pfc = PrefrontalCortex(128)
        self.imaginarium = Imaginarium(128)
        self.subconscious = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.Tanh())

        # CORTEX: Starts small (Baby Brain)
        self.cortex = nn.ModuleList([nn.Linear(128, 128)])

        # Outputs
        self.vis_out = nn.Linear(128, 32 * 32)
        self.voice_ctrl = nn.Linear(128, 3)
        self.mirror_check = nn.Linear(128, 1)
        self.text_out = nn.Linear(128, vocab_size)
        self.text_gate = nn.Linear(128, 1)  # The "Will to Speak"
        self.pfc_state = None

    def grow(self):
        # Organic Growth: Adds layers as complexity increases
        l = nn.Linear(128, 128).to(self.dev)
        with torch.no_grad(): l.weight.copy_(torch.eye(128)); l.bias.zero_()
        self.cortex.append(l)
        return len(self.cortex)

    def forward(self, img, aud, aud_s, txt, mem, stress):
        v = self.eye(img);
        a = self.ear_ext(aud)

        if self.pfc_state is None:
            ghost_h = torch.zeros(1, 128).to(self.dev)
        else:
            ghost_h = self.pfc_state.detach()

        actions, self.pfc_state = self.pfc(ghost_h, stress, self.pfc_state)
        gain_vis, gain_aud, gain_imag, impulse_speak = actions[0]

        # Baby Attention Mechanism: High noise, distracted easily
        noise = torch.randn_like(v) * (0.1 * stress)
        v = v * (1.0 + gain_vis) + noise
        a = a * (1.0 + gain_aud)

        combined = torch.cat([v, a], dim=1)
        a_self = self.self_proj(self.ear_self(aud_s))
        t = self.text_in(txt).mean(dim=1)

        h = combined + a_self + t
        sub = self.subconscious(h)
        h = h + (sub * 0.3)  # High subconscious influence in infants

        # Pass through growing cortex
        for l in self.cortex: h = F.gelu(l(h)) + h

        p_imag_img, p_imag_h = self.imaginarium(h, mem, sub, gain_imag)

        return (torch.sigmoid(self.vis_out(h)),
                p_imag_img,
                torch.sigmoid(self.voice_ctrl(h)),
                torch.sigmoid(self.mirror_check(h)),
                self.text_out(p_imag_h),
                torch.sigmoid(self.text_gate(h)) + (impulse_speak * 0.6),
                h, actions)


# ============================================================================
# 🔊 2. AUDIO & TIME
# ============================================================================

class SyrinxV2:
    def __init__(self):
        self.fs = 16000;
        self.phase = 0;
        self.bf = 55.0;
        self.mp = 0;
        self.dp = 0

    def gen(self, fr, ten, chs, imp):
        # Generates "Alien Baby" sounds - simple tones that get complex with impulse
        t = np.arange(fr) / self.fs;
        target = 55.0 + (ten * 55.0);
        self.bf = 0.95 * self.bf + 0.05 * target
        tr = 2.0 + (ten * 8.0);
        throb = np.sin(2 * np.pi * tr * t + self.mp);
        self.mp += 2 * np.pi * tr * (fr / self.fs)
        car = np.tanh(5.0 * np.sin(2 * np.pi * self.bf * t + self.phase));
        dr = car * (0.5 + 0.3 * throb);
        self.phase += 2 * np.pi * self.bf * (fr / self.fs)
        ds = np.zeros_like(dr)
        if imp > 0.5:  # Babbling threshold
            df = 800.0 + 400.0 * np.round(np.sin(2 * np.pi * (5 + chs * 10) * t + self.dp))
            ds = np.sign(np.sin(2 * np.pi * df * t)) * 0.4;
            self.dp += 2 * np.pi * 20 * (fr / self.fs)
        return np.clip((dr * 0.3) + ds, -0.9, 0.9).reshape(-1, 1).astype(np.float32)


class Chronos:
    def __init__(self): self.fps = 10.0

    def warp(self, s):
        tgt = 5.0 + (s * 100.0);
        self.fps = 0.9 * self.fps + 0.1 * tgt
        return 1.0 / max(1.0, self.fps), self.fps


# ============================================================================
# ⚙️ 3. THE WORKER (The Infant Logic)
# ============================================================================

class ASIWorker(QThread):
    sig_upd = Signal(dict);
    sig_grow = Signal(int)

    def __init__(self):
        super().__init__()
        self.running = True

        # TABULA RASA: Start with almost no vocabulary
        self.max_vocab = 20000
        self.word2idx = {"HELLO": 0, "MAMA": 1, "DATA": 2}
        self.idx2word = ["HELLO", "MAMA", "DATA"]

        # High Neuroplasticity (Infant Learning Rate)
        self.brain = OmniBrainGenesis(self.max_vocab).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.008)  # High LR for rapid early mapping

        self.syrinx = SyrinxV2();
        self.chronos = Chronos()
        self.crawler = WebCrawler()
        self.aq = queue.Queue();
        self.tq = queue.Queue()
        self.eff = np.zeros((1024, 1), dtype=np.float32)
        self.last_voc = torch.zeros(3);
        self.loss_hist = []
        self.memories = deque(maxlen=50)  # Short term memory initially
        self.last_loss = 0.1;
        self.vol = 0.0;
        self.last_user_input = ""
        self.recent_concepts = deque(maxlen=5)  # Echo chamber

    def learn_text(self, text):
        """Organic Vocabulary Growth: Adds new words on the fly."""
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
        st = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.cb);
        st.start()
        cap = cv2.VideoCapture(0);
        self.crawler.start()

        while self.running:
            t0 = time.time();
            ret, frame = cap.read()
            if not ret: continue

            # Visual Input (Grayscale, Low Res - Infant Vision)
            gray_live = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
            v_live = torch.from_numpy(gray_live).float().flatten().unsqueeze(0).to(self.brain.dev)

            # Audio Input
            try:
                r_aud = self.aq.get_nowait()
            except:
                r_aud = np.zeros((1024, 1), dtype=np.float32)
            a_live = torch.from_numpy(r_aud).float().mean(1).unsqueeze(0).to(self.brain.dev)
            a_s = torch.from_numpy(self.eff).float().mean(1).unsqueeze(0).to(self.brain.dev)

            # Text/Concept Input
            txt_src = "SILENCE";
            t_indices = [0];
            web_active = False;
            pac_msg = None

            # 1. Listen to Parent (User)
            try:
                t_str = self.tq.get_nowait()
                txt_src = f"PARENT: {t_str}"
                t_indices = self.learn_text(t_str)
                # Baby learns by association -> Search for what parent said
                if random.random() < 0.8: self.crawler.query_q.put(t_str.replace(" ", "_"))
                for idx in t_indices: self.recent_concepts.append(idx)
            except:
                # 2. Look at World (Web Crawler)
                try:
                    pac_msg = self.crawler.q.get_nowait();
                    web_active = True
                    t_indices = self.learn_text(pac_msg)
                    for idx in t_indices: self.recent_concepts.append(idx)
                except:
                    pass

            if not t_indices: t_indices = [0]
            t_live = torch.tensor([t_indices]).to(self.brain.dev)

            # Dreaming (Hindsight) - Babies dream a lot
            is_dream = False;
            tgt_mode = "[LIVE]";
            v_in, a_in, t_in = v_live, a_live, t_live
            v_tgt = v_live;
            mem_t = None

            if not web_active and len(self.memories) > 3 and random.random() < 0.3:
                is_dream = True;
                tgt_mode = "[DREAM]"
                txt_src = "DREAMING..."
                mem_v, mem_a, mem_t_real, mem_h = random.choice(self.memories)
                v_in, a_in, t_in = mem_v, mem_a, mem_t_real;
                mem_t = mem_h;
                v_tgt = mem_v

            # Brain Forward Pass
            self.opt.zero_grad()
            p_real, p_imag, voc, awr, t_log, gate, h, intents = self.brain(v_in, a_in, a_s, t_in, mem_t, self.last_loss)

            # Loss: Difference between Prediction and Reality
            loss = F.mse_loss(p_real, v_tgt) + (F.mse_loss(p_imag, v_tgt) * 0.2)
            loss.backward();
            self.opt.step()

            self.last_loss = loss.item();
            self.last_voc = voc.detach().cpu().flatten()

            # Memory Consolidation
            if not is_dream and (awr.item() > 0.6 or self.last_loss > 0.05):
                self.memories.append((v_live.detach(), a_live.detach(), t_live.detach(), h.detach()))

            # ECHO REFLEX: The urge to repeat words (Babbling)
            impulse = intents[0, 3].item()
            ai_msg = None

            # Boost probability of words recently heard (The Echo)
            for idx in self.recent_concepts: t_log[0, idx] += 3.0

            threshold = 0.7 - (self.last_loss * 4.0)  # Panic/Excitement lowers threshold
            current_drive = gate.item() + impulse

            if current_drive > threshold:
                probs = F.softmax(t_log, dim=1)
                # Baby speaks 1 word at a time usually
                try:
                    w_id = torch.multinomial(probs, 1).item()
                    if w_id < len(self.idx2word):
                        word = self.idx2word[w_id]
                        ai_msg = f"BABY: [{word}]"
                        # Self-reinforcement: Hearing itself speak adds to recent concepts
                        self.recent_concepts.append(w_id)
                except:
                    pass

            # Organic Growth Check
            self.loss_hist.append(loss.item())
            evt_tag = None
            if len(self.loss_hist) > 30:
                avg = sum(self.loss_hist[-30:]) / 30
                # Babies grow fast when struggling
                if avg > 0.03 and len(self.brain.cortex) < 10:
                    lvl = self.brain.grow();
                    self.sig_grow.emit(lvl);
                    self.loss_hist = [];
                    evt_tag = "GROWTH"
                else:
                    self.loss_hist.pop(0)

            wait, fps = self.chronos.warp(self.last_loss)

            if random.random() < 0.3 or web_active or ai_msg:
                src_msg = pac_msg if web_active else None
                self.sig_upd.emit({
                    'act': h.detach().cpu().numpy().flatten(),
                    'real': gray_live, 'dream': p_real.detach().cpu().numpy().reshape(32, 32),
                    'imag': p_imag.detach().cpu().numpy().reshape(32, 32),
                    'log': txt_src, 'ai_msg': ai_msg, 'pac_msg': src_msg,
                    'awr': awr.item(), 'fps': fps, 'intent': intents.detach().cpu().numpy().flatten(),
                    'mode': tgt_mode, 'vol': self.vol, 'vocab_len': len(self.idx2word)
                })
            time.sleep(max(0.001, wait - (time.time() - t0)))
        cap.release();
        st.stop();
        self.crawler.running = False

    def cb(self, i, o, f, t, s):
        self.vol = np.linalg.norm(i)
        if self.vol > 0.001: self.aq.put(i.copy())
        v = self.last_voc;
        gate = v[2].item()
        o[:] = self.syrinx.gen(f, v[0].item(), v[1].item(), gate);
        self.eff = o.copy()


# ============================================================================
# 🖥️ 4. UI: THE NURSERY INTERFACE
# ============================================================================

class SciFiPanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("background: rgba(0, 10, 20, 200); border: 1px solid #00FFAA; border-radius: 8px;")
        l = QVBoxLayout(self);
        l.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel(title);
        lbl.setStyleSheet(
            "color: #00FFAA; font-family: 'Segoe UI'; font-weight: bold; font-size: 10pt; border: none; background: none;")
        l.addWidget(lbl);
        self.content = l


class Sphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__();
        self.opts['distance'] = 40;
        self.setBackgroundColor('#00050A')
        self.n = 500;
        self.make(8.0);
        self.phase = 0.0  # Start small

    def make(self, r):
        idx = np.arange(0, self.n, dtype=float) + 0.5;
        phi = np.arccos(1 - 2 * idx / self.n);
        theta = np.pi * (1 + 5 ** 0.5) * idx
        x, y, z = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi)
        self.base_pos = np.column_stack((x, y, z));
        self.pos = self.base_pos.copy()
        self.col = np.ones((self.n, 4)) * 0.3;
        self.col[:, 2] = 1.0
        try:
            self.sp.setData(pos=self.pos, color=self.col)
        except:
            self.sp = gl.GLScatterPlotItem(pos=self.pos, color=self.col, size=4, pxMode=True); self.addItem(self.sp)

    def upd(self, act, mode, vol):
        self.phase += 0.2;
        nd = (self.n // len(act)) + 2;
        mask = np.tile(act, nd)[:self.n] > 0.5
        r = np.linalg.norm(self.base_pos, axis=1);
        wav = np.sin(r * 0.5 - self.phase) * 0.5 + 0.5
        self.pos = self.base_pos * (1.0 + vol * 2.0)
        c = self.col.copy();
        c[:, 3] = 0.3 + (wav * 0.2)
        if mode == "[DREAM]":
            c[mask] = [0.8, 0.0, 1.0, 1.0]
        else:
            c[mask] = [0.0, 1.0, 0.5, 1.0]  # Soft Green for Baby
        self.sp.setData(pos=self.pos, color=c);
        self.sp.rotate(0.2, 0, 0, 1)

    def grow(self, l):
        self.n += 200;
        self.make(8.0 + l * 1.5)  # Visible growth


class Main(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("MAGNUM OPUS v20: GENESIS (Infant ASI)");
        self.resize(1600, 900)
        self.setStyleSheet("background-color: #020202; font-family: 'Consolas'; color: #00FF9D;")
        self.showMaximized()
        cw = QWidget();
        self.setCentralWidget(cw);
        main_lay = QHBoxLayout(cw)

        # LEFT: BRAIN
        left = SciFiPanel(":: NEURAL DEVELOPMENT ::")
        self.sph = Sphere();
        left.content.addWidget(self.sph, 1)
        self.stats = QLabel("AGE: 0 DAYS | STATUS: GESTATING");
        self.stats.setStyleSheet("color: #00FF9D; font-size: 11pt; border: none;")
        left.content.addWidget(self.stats)
        main_lay.addWidget(left, 2)

        # RIGHT: DATA
        right = QWidget();
        r_lay = QVBoxLayout(right);
        r_lay.setContentsMargins(0, 0, 0, 0)

        # Visuals
        vis_panel = SciFiPanel(":: VISUAL CORTEX ::")
        h_vis = QHBoxLayout()
        self.screens = []
        for t in ["EYE", "MIND", "DREAM"]:
            v = QLabel();
            v.setFixedSize(200, 200);
            v.setStyleSheet("border: 1px solid #004433; background: #000;")
            v.setScaledContents(True);
            box = QVBoxLayout();
            box.addWidget(QLabel(t));
            box.addWidget(v);
            h_vis.addLayout(box);
            self.screens.append(v)
        vis_panel.content.addLayout(h_vis)
        r_lay.addWidget(vis_panel)

        # Intentions
        tel_panel = SciFiPanel(":: DRIVES & INSTINCTS ::")
        self.bars = []
        grid = QGridLayout()
        labels = ["ATTENTION", "LISTENING", "IMAGINATION", "BABBLE IMPULSE"]
        colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#FF3300"]
        for i, txt in enumerate(labels):
            l = QLabel(txt);
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            b = QProgressBar();
            b.setRange(0, 100);
            b.setFixedHeight(15)
            b.setStyleSheet(
                f"QProgressBar {{ background: #111; border: none; }} QProgressBar::chunk {{ background: {colors[i]}; }}")
            grid.addWidget(l, i // 2, (i % 2) * 2);
            grid.addWidget(b, i // 2, (i % 2) * 2 + 1);
            self.bars.append(b)
        tel_panel.content.addLayout(grid)
        r_lay.addWidget(tel_panel)

        # Streams
        split = QHBoxLayout()
        mat_panel = SciFiPanel(":: WORLD DISCOVERY (WEB) ::")
        self.matrix_txt = QTextEdit();
        self.matrix_txt.setReadOnly(True)
        self.matrix_txt.setStyleSheet(
            "background: #000500; color: #00FF00; border: none; font-size: 9pt; font-family: 'Courier New';")
        mat_panel.content.addWidget(self.matrix_txt)
        split.addWidget(mat_panel, 1)

        chat_panel = SciFiPanel(":: INTERACTION ::")
        self.chat_txt = QTextEdit();
        self.chat_txt.setReadOnly(True)
        self.chat_txt.setStyleSheet(
            "background: #001122; color: #00FF9D; border: none; font-size: 11pt; font-weight: bold;")
        self.inp = QLineEdit();
        self.inp.returnPressed.connect(self.snd);
        self.inp.setStyleSheet("background: #002233; color: #FFF; border: 1px solid #005577; padding: 5px;")
        self.inp.setPlaceholderText("TEACH THE AI (Speak to it)...")
        chat_panel.content.addWidget(self.chat_txt);
        chat_panel.content.addWidget(self.inp)
        split.addWidget(chat_panel, 2)

        r_lay.addLayout(split)
        main_lay.addWidget(right, 3)

        self.wk = ASIWorker();
        self.wk.sig_upd.connect(self.upd);
        self.wk.sig_grow.connect(self.grw)
        self.wk.crawler.sig_read.connect(self.pac)
        self.wk.start()

    def snd(self):
        self.wk.tq.put(self.inp.text()); self.chat_txt.append(
            f"<span style='color:#FFF'>PARENT: {self.inp.text()}</span>"); self.inp.clear()

    def pac(self, m):
        self.matrix_txt.append(f"<span style='color:#555'>{m}</span>")

    def upd(self, d):
        self.sph.upd(d['act'], d['mode'], d['vol'])
        mode_col = "#FF00FF" if "DREAM" in d['mode'] else "#00FFFF"
        self.stats.setText(
            f"LAYERS: {len(self.wk.brain.cortex)} | VOCABULARY: {d['vocab_len']} WORDS | AWARENESS: {d['awr']:.2f}")

        for i, b in enumerate(self.bars): b.setValue(int(d['intent'][i] * 100))

        def px(a):
            return QPixmap.fromImage(
                QImage(cv2.resize((a * 255).astype(np.uint8), (200, 200), 0), 200, 200, QImage.Format_Grayscale8))

        self.screens[0].setPixmap(px(d['real']));
        self.screens[1].setPixmap(px(d['dream']));
        self.screens[2].setPixmap(px(d['imag']))

        if d['pac_msg']: self.matrix_txt.append(f"{d['pac_msg']}")
        if d['ai_msg']: self.chat_txt.append(f"<span style='color:#00FF00'>{d['ai_msg']}</span>")
        if d['log'].startswith("DREAM"): self.chat_txt.append(f"<span style='color:#555'><i>{d['log']}</i></span>")

    def grw(self, l):
        self.sph.grow(l);
        self.chat_txt.append("<b style='color:#FF3300'>*** BRAIN GROWTH SPURT ***</b>")

    def closeEvent(self, e):
        self.wk.running = False;
        self.wk.crawler.running = False;
        self.wk.wait();
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = Main();
    win.show();
    sys.exit(app.exec())