"""
Magnum Opus v3.6: The Architect Edition
================================================================
ARCHITECT: Alan Hourmand
STATUS: ONLINE (Spatial Topology + Resizable UI)

CHANGELOG:
- UI: Implemented QSplitter (Draggable resize).
- BRAIN: Neurons are now clustered by function (Vision=Top, Audio=Bottom).
- AUDIO: Added Visual Voice Meter to confirm sound generation.
- RES: Defaulted to Widescreen 1400x900.
"""

import sys
import time
import queue
import random
import math
import cv2
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, QSplitter, QProgressBar)
from PySide6.QtCore import QTimer, Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPalette
import pyqtgraph.opengl as gl

# ============================================================================
# 🔊 AUDIO ENGINE (FM Synthesis)
# ============================================================================

class DroneSynthesizer:
    def __init__(self):
        self.sample_rate = 16000
        self.phase = 0
        self.base_freq = 60.0
        self.target_freq = 60.0
        self.mod_index = 5.0
        self.talking_gate = 0.0

    def generate(self, frames, loss, activation_intensity, talking_gate):
        t = np.arange(frames) / self.sample_rate

        # Dynamic Frequency based on Entropy (Loss)
        self.target_freq = 50.0 + (loss * 150)
        self.base_freq = 0.9 * self.base_freq + 0.1 * self.target_freq

        # Modulator for texture
        mod_freq = self.base_freq * (2.0 + activation_intensity)

        # FM Synthesis
        carrier = np.sin(2 * np.pi * self.base_freq * t + self.phase +
                         (self.mod_index * activation_intensity) * np.sin(2 * np.pi * mod_freq * t))

        # Voice Overlay (High pitch chirps if "Talking")
        if talking_gate > 0.6:
            voice_mod = np.sin(2 * np.pi * (800 + random.randint(-100, 100)) * t) * 0.2
            carrier += voice_mod

        self.phase += 2 * np.pi * self.base_freq * (frames / self.sample_rate)

        # Master Volume
        return carrier.reshape(-1, 1).astype(np.float32) * 0.2

# ============================================================================
# 🧠 SPATIAL BRAIN (Functionally Mapped)
# ============================================================================

class FunctionalBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # We split the hidden dimension into functional blocks
        self.dim_vis = 64
        self.dim_aud = 64
        self.dim_total = 128

        # Encoders
        self.eye = nn.Sequential(nn.Linear(32*32, self.dim_vis), nn.LayerNorm(self.dim_vis), nn.Tanh())
        self.ear = nn.Sequential(nn.Linear(1024, self.dim_aud), nn.LayerNorm(self.dim_aud), nn.Tanh())
        self.text = nn.Embedding(1000, self.dim_total)

        # Associative Core (The "Corpus Callosum")
        self.core = nn.GRUCell(self.dim_total, self.dim_total)
        self.hidden = None

        # Decoders
        self.vis_out = nn.Linear(self.dim_total, 32*32)
        self.voice_gate = nn.Linear(self.dim_total, 1) # Should I speak?

    def forward(self, img, audio, text_idx, h):
        if h is None: h = torch.zeros(img.size(0), self.dim_total).to(img.device)

        # 1. Encode Senses
        v_emb = self.eye(img)     # [1, 64]
        a_emb = self.ear(audio)   # [1, 64]

        # Concatenate to form full sensory experience
        # Vision goes to first half, Audio to second half
        sensory = torch.cat([v_emb, a_emb], dim=1) # [1, 128]

        # Add Text context (global)
        t_emb = self.text(text_idx).mean(dim=1)
        sensory = sensory + t_emb

        # 2. Process
        new_h = self.core(sensory, h)

        # 3. Decode
        img_pred = torch.sigmoid(self.vis_out(new_h))
        talking = torch.sigmoid(self.voice_gate(new_h))

        return img_pred, talking, new_h

# ============================================================================
# ⚙️ WORKER THREAD
# ============================================================================

class ASIWorker(QThread):
    sig_update = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = FunctionalBrain().to(self.device)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.005)
        self.synth = DroneSynthesizer()

        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.hidden_state = None

        # Stats for audio thread
        self.last_loss = 0.0
        self.last_act = 0.0
        self.last_talk = 0.0

    def audio_callback(self, indata, outdata, frames, time, status):
        # Input
        if np.linalg.norm(indata) > 0.05:
            self.audio_queue.put(indata.copy())

        # Output (Synthesized Drone)
        wave = self.synth.generate(frames, self.last_loss, self.last_act, self.last_talk)
        outdata[:] = wave

    def run(self):
        stream = sd.Stream(channels=1, samplerate=16000, blocksize=1024, callback=self.audio_callback)
        stream.start()
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret: continue

            # --- PREP ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32))
            vis_t = torch.from_numpy(small).float().flatten().unsqueeze(0).to(self.device) / 255.0

            try: aud_data = self.audio_queue.get_nowait()
            except: aud_data = np.zeros((1024,1))
            aud_t = torch.from_numpy(aud_data).float().mean(axis=1).unsqueeze(0).to(self.device)

            try:
                txt = self.text_queue.get_nowait()
                txt_t = torch.tensor([[hash(txt) % 1000]]).to(self.device)
                chat_out = f"USER: {txt}"
            except:
                txt_t = torch.tensor([[0]]).to(self.device)
                chat_out = None

            # --- THINK ---
            self.optimizer.zero_grad()
            pred_img, talking, self.hidden_state = self.brain(vis_t, aud_t, txt_t, self.hidden_state)
            self.hidden_state = self.hidden_state.detach()

            loss = F.mse_loss(pred_img, vis_t)
            loss.backward()
            self.optimizer.step()

            # --- SYNC AUDIO & UI ---
            self.last_loss = loss.item()
            self.last_act = self.hidden_state.abs().mean().item()
            self.last_talk = talking.item()

            if random.random() < 0.3: # 30 FPS UI update
                activations = self.hidden_state.cpu().numpy().flatten()
                data = {
                    'loss': self.last_loss,
                    'act': activations,
                    'talk': self.last_talk,
                    'real': small,
                    'dream': pred_img.detach().cpu().numpy().reshape(32,32),
                    'chat': chat_out
                }
                self.sig_update.emit(data)

            time.sleep(0.015)

        cap.release()
        stream.stop()

# ============================================================================
# 🖥️ UI: THE ARCHITECT INTERFACE
# ============================================================================

class SpatialSphere(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts['distance'] = 35
        self.opts['fov'] = 50 # cinematic fov
        self.setWindowTitle('Neural Sphere')
        self.setBackgroundColor('#050505')

        self.n_points = 2000

        # Generate Sphere
        indices = np.arange(0, self.n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/self.n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = 10 * np.cos(theta) * np.sin(phi), 10 * np.sin(theta) * np.sin(phi), 10 * np.cos(phi)
        self.pos = np.column_stack((x, y, z))

        # Define Regions
        self.mask_vis = z > 2   # Top
        self.mask_aud = z < -2  # Bottom
        self.mask_core = (z >= -2) & (z <= 2) # Middle

        self.colors = np.zeros((self.n_points, 4))
        self.colors[:, 3] = 0.4 # Alpha

        # Default Region Colors
        self.colors[self.mask_vis] = [0.0, 1.0, 1.0, 0.4]  # Cyan (Vision)
        self.colors[self.mask_aud] = [1.0, 0.0, 1.0, 0.4]  # Magenta (Audio)
        self.colors[self.mask_core] = [1.0, 0.8, 0.0, 0.4] # Gold (Core)

        self.scatter = gl.GLScatterPlotItem(pos=self.pos, color=self.colors, size=4, pxMode=True)
        self.addItem(self.scatter)

    def update_state(self, activations):
        # activations is size 128.
        # 0-63 = Vision, 64-127 = Audio.

        vis_act = activations[:64]
        aud_act = activations[64:]

        # Map intensity to alpha/brightness
        vis_mean = vis_act.mean()
        aud_mean = aud_act.mean()

        # Reset
        self.colors[self.mask_vis] = [0.0, 1.0, 1.0, 0.2]
        self.colors[self.mask_aud] = [1.0, 0.0, 1.0, 0.2]
        self.colors[self.mask_core] = [1.0, 0.8, 0.0, 0.1]

        # Ignite active neurons
        # We map the 64 activation values randomly across the thousands of points in that region
        # for a "sparkling" effect.

        if vis_mean > 0.1:
            # Light up random top nodes
            self.colors[self.mask_vis] = [0.5, 1.0, 1.0, 0.8]

        if aud_mean > 0.1:
            # Light up random bottom nodes
            self.colors[self.mask_aud] = [1.0, 0.5, 1.0, 0.8]

        self.scatter.setData(color=self.colors)
        self.scatter.rotate(0.2, 0, 0, 1) # Slow rotation

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAGNUM OPUS v3.6: ARCHITECT")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #000; }
            QLabel { color: #CCC; font-family: 'Segoe UI'; font-size: 10pt; }
            QTextEdit { background-color: #111; color: #0F0; border: none; font-family: Consolas; }
            QLineEdit { background-color: #222; color: #FFF; border: 1px solid #444; padding: 5px; }
            QProgressBar { border: 1px solid #444; background: #111; text-align: center; }
            QProgressBar::chunk { background-color: #00FF9D; }
        """)

        # SPLITTER (The anti-squish mechanism)
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # --- LEFT PANEL (THE BRAIN) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)

        self.brain_viz = SpatialSphere()
        left_layout.addWidget(self.brain_viz)

        # Overlay Stats
        stats_box = QHBoxLayout()
        self.status_lbl = QLabel("SYSTEM: ONLINE")
        self.voice_bar = QProgressBar()
        self.voice_bar.setRange(0, 100)
        self.voice_bar.setTextVisible(False)
        self.voice_bar.setFixedWidth(100)
        self.voice_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0055; }")

        stats_box.addWidget(QLabel("VOICE AMP:"))
        stats_box.addWidget(self.voice_bar)
        stats_box.addStretch()
        stats_box.addWidget(self.status_lbl)

        left_widget_ctrl = QWidget()
        left_widget_ctrl.setLayout(stats_box)
        left_widget_ctrl.setMaximumHeight(50)
        left_layout.addWidget(left_widget_ctrl)

        splitter.addWidget(left_widget)

        # --- RIGHT PANEL (DATA) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Vision Area
        vis_layout = QHBoxLayout()
        self.vis_real = QLabel()
        self.vis_dream = QLabel()
        for l in [self.vis_real, self.vis_dream]:
            l.setFixedSize(200, 200)
            l.setStyleSheet("background: #050505; border: 1px solid #333;")
            l.setScaledContents(True)

        vis_layout.addWidget(self.vis_real)
        vis_layout.addWidget(self.vis_dream)
        right_layout.addLayout(vis_layout)
        right_layout.addWidget(QLabel("OPTICAL FEED // PREDICTIVE MODEL"))

        # Chat Area
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.inp = QLineEdit(); self.inp.setPlaceholderText("Inject Linguistic Data...")
        self.inp.returnPressed.connect(self.send_text)

        right_layout.addWidget(self.log)
        right_layout.addWidget(self.inp)

        splitter.addWidget(right_widget)

        # Set Splitter Ratios (70% Brain, 30% Data)
        splitter.setSizes([900, 400])

        # Worker
        self.worker = ASIWorker()
        self.worker.sig_update.connect(self.update_ui)
        self.worker.start()

    def send_text(self):
        t = self.inp.text()
        if t:
            self.worker.text_queue.put(t)
            self.inp.clear()

    def update_ui(self, data):
        # Update Brain
        self.brain_viz.update_state(data['act'])

        # Update Voice Meter
        amp = int(data['talk'] * 100)
        self.voice_bar.setValue(amp)

        # Update Status
        loss = data['loss']
        self.status_lbl.setText(f"ENTROPY: {loss:.4f} | NEURONS: 2000")

        # Update Vision
        def np2pix(arr):
            arr = (arr * 255).astype(np.uint8)
            # Upscale for UI
            arr = cv2.resize(arr, (200, 200), interpolation=cv2.INTER_NEAREST)
            return QPixmap.fromImage(QImage(arr.data, 200, 200, QImage.Format_Grayscale8))

        self.vis_real.setPixmap(np2pix(data['real']))
        self.vis_dream.setPixmap(np2pix(data['dream']))

        if data['chat']:
            self.log.append(data['chat'])

    def closeEvent(self, event):
        self.worker.running = False
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())