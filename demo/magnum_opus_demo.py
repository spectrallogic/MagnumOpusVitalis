"""
MagnumOpusVitalis Demo
======================
The full experience: neural sphere visualization, emotional audio, real-time learning.

This is the "living AI" demo - the artistic expression of the core algorithm.
For pure algorithm, see core/magnum_opus_core.py

Author: Alan Hourmand
"""

import sys
import os
import time
import queue
import threading
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import core algorithm
from magnum_opus_core import MagnumOpusCore, GrowthConfig, ComputationalEnergy

# ============================================================================
# OPTIONAL IMPORTS (graceful degradation)
# ============================================================================

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                   QHBoxLayout, QTextEdit, QLineEdit, QLabel,
                                   QProgressBar, QSplitter, QFrame)
    from PySide6.QtCore import QThread, Signal, Qt, QTimer
    from PySide6.QtGui import QFont, QColor, QImage, QPixmap

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    print("PySide6 not available - running in console mode")

try:
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("pyqtgraph not available - no 3D visualization")

try:
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not available - no audio output")

try:
    import cv2

    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("OpenCV not available - no camera input")


# ============================================================================
# VOCABULARY (Tabula Rasa)
# ============================================================================

class TabulaRasaVocabulary:
    """
    A vocabulary that starts EMPTY and learns words from experience.
    No pre-training, no cheating.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.word_to_idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts: Dict[str, int] = {}
        self.next_idx = 2

    def add_word(self, word: str) -> int:
        """Add a word to vocabulary, return its index"""
        word = word.upper().strip()
        if not word:
            return 1  # UNK

        if word in self.word_to_idx:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
            return self.word_to_idx[word]

        if self.next_idx >= self.max_size:
            return 1  # UNK - vocab full

        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.word_counts[word] = 1
        self.next_idx += 1
        return idx

    def encode(self, text: str) -> List[int]:
        """Encode text to indices"""
        words = text.upper().split()
        return [self.add_word(w) for w in words]

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text"""
        return " ".join(self.idx_to_word.get(i, "<UNK>") for i in indices)

    def __len__(self):
        return self.next_idx


# ============================================================================
# EMOTIONAL AUDIO SYNTHESIS (SYRINX)
# ============================================================================

class EmotionalSyrinx:
    """
    Audio synthesizer with emotional layers:
    1. Thinking drone (55-110Hz) - always present
    2. Confusion/crying (300-600Hz warble) - triggered by stress
    3. Speech tones (800Hz+) - when speaking
    4. Growth event (one-shot gong)
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.phase = 0.0
        self.cry_phase = 0.0

        # State
        self.tension = 0.5  # 0-1, drives thinking drone
        self.stress = 0.0  # 0-1, drives crying
        self.speak_drive = 0.0  # 0-1, drives speech tones
        self.cry_suppression = 0.0  # Learned suppression

        # Growth event flag
        self.growth_pending = False

    def trigger_growth(self):
        """Signal that growth just occurred"""
        self.growth_pending = True

    def generate_chunk(self, num_samples: int = 1024) -> np.ndarray:
        """Generate audio chunk based on current emotional state"""
        t = np.arange(num_samples) / self.sr
        audio = np.zeros(num_samples, dtype=np.float32)

        # Layer 1: Thinking drone (always present)
        drone_freq = 55 + self.tension * 55  # 55-110Hz
        drone = np.sin(2 * np.pi * drone_freq * t + self.phase) * 0.1
        audio += drone * (0.3 + self.tension * 0.3)

        # Layer 2: Crying/confusion (stress-triggered, suppressible)
        effective_stress = self.stress * (1.0 - self.cry_suppression * 0.8)
        if effective_stress > 0.3:
            cry_freq = 300 + 300 * np.sin(self.cry_phase * 5)  # Warbling
            cry = np.sin(2 * np.pi * cry_freq * t + self.cry_phase) * effective_stress * 0.15
            audio += cry
            self.cry_phase += 0.1

        # Layer 3: Speech tones
        if self.speak_drive > 0.5:
            speech_freq = 800 + self.speak_drive * 400
            speech = np.sin(2 * np.pi * speech_freq * t) * self.speak_drive * 0.2
            audio += speech

        # Layer 4: Growth event (one-shot gong)
        if self.growth_pending:
            gong_freq = 220
            gong = np.sin(2 * np.pi * gong_freq * t) * np.exp(-t * 3) * 0.5
            audio += gong
            self.growth_pending = False

        # Update phase
        self.phase += 2 * np.pi * drone_freq * num_samples / self.sr
        self.phase %= 2 * np.pi

        # Soft clip
        audio = np.tanh(audio * 2) * 0.5

        return audio


# ============================================================================
# NEURAL SPHERE VISUALIZATION
# ============================================================================

class NeuralSphere:
    """
    3D visualization of neural activity as a pulsating sphere.

    - Points on sphere represent neural clusters
    - Colors show activation regions (cyan=vision, magenta=audio, gold=core)
    - Waves propagate based on internal state
    - Grows physically when model grows
    """

    def __init__(self, initial_points: int = 500):
        self.num_points = initial_points
        self.base_radius = 1.0
        self.current_radius = 1.0

        # Generate Fibonacci lattice sphere
        self.positions = self._fibonacci_sphere(initial_points)
        self.colors = self._initial_colors(initial_points)

        # Animation state
        self.time = 0.0
        self.wave_offset = 0.0
        self.breath_phase = 0.0

        # Activation channels
        self.learning_wave = 0.0  # Cyan
        self.stress_wave = 0.0  # Red
        self.growth_wave = 0.0  # Purple
        self.speak_wave = 0.0  # All colors sync

    def _fibonacci_sphere(self, n: int) -> np.ndarray:
        """Generate evenly distributed points using golden ratio"""
        indices = np.arange(n, dtype=float)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        y = 1 - (indices / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)

        theta = phi * indices

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        return np.column_stack([x, y, z])

    def _initial_colors(self, n: int) -> np.ndarray:
        """Initial colors: top=cyan (vision), bottom=magenta (audio), middle=gold (core)"""
        colors = np.zeros((n, 4), dtype=np.float32)

        for i in range(n):
            y = self.positions[i, 1]

            if y > 0.3:  # Top - vision (cyan)
                colors[i] = [0.0, 0.8, 1.0, 0.8]
            elif y < -0.3:  # Bottom - audio (magenta)
                colors[i] = [1.0, 0.0, 0.8, 0.8]
            else:  # Middle - core (gold)
                colors[i] = [1.0, 0.8, 0.0, 0.8]

        return colors

    def update(self, dt: float, loss: float, stress: float, speaking: bool, grew: bool):
        """Update sphere state based on neural activity"""
        self.time += dt

        # Update waves
        self.learning_wave = max(0, self.learning_wave - dt * 0.5)
        self.stress_wave = stress * 0.8 + self.stress_wave * 0.2
        self.speak_wave = 1.0 if speaking else max(0, self.speak_wave - dt * 2)

        if grew:
            self.growth_wave = 1.0
            self.grow_points(50)  # Add points on growth
        else:
            self.growth_wave = max(0, self.growth_wave - dt * 0.3)

        if loss < 0.5:  # Good learning
            self.learning_wave = min(1.0, self.learning_wave + dt * 2)

        # Breathing (expands when speaking)
        target_radius = 1.0 + self.speak_wave * 0.1
        self.current_radius += (target_radius - self.current_radius) * dt * 5

        # Wave propagation
        self.wave_offset += dt * 2

    def grow_points(self, additional: int):
        """Add more points to sphere (called on growth events)"""
        new_total = self.num_points + additional
        new_positions = self._fibonacci_sphere(new_total)
        new_colors = self._initial_colors(new_total)

        self.positions = new_positions
        self.colors = new_colors
        self.num_points = new_total

    def get_render_data(self) -> tuple:
        """Get current positions and colors for rendering"""
        # Apply radius and waves
        positions = self.positions.copy() * self.current_radius
        colors = self.colors.copy()

        # Apply wave effects to colors
        for i in range(self.num_points):
            y = self.positions[i, 1]
            wave_phase = y * 3 + self.wave_offset
            wave_intensity = (np.sin(wave_phase) + 1) * 0.5

            # Learning wave (adds cyan)
            colors[i, 1] += self.learning_wave * wave_intensity * 0.3
            colors[i, 2] += self.learning_wave * wave_intensity * 0.3

            # Stress wave (adds red)
            colors[i, 0] += self.stress_wave * wave_intensity * 0.5

            # Growth wave (adds purple/white flash)
            colors[i] += self.growth_wave * 0.5

        # Clamp colors
        colors = np.clip(colors, 0, 1)

        return positions, colors


# ============================================================================
# MAIN BRAIN WORKER (runs in separate thread)
# ============================================================================

class BrainWorker(QThread if PYSIDE_AVAILABLE else threading.Thread):
    """
    Main processing loop - runs the brain continuously.
    Emits signals for UI updates.
    """

    if PYSIDE_AVAILABLE:
        sig_update = Signal(dict)
        sig_growth = Signal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.text_queue = queue.Queue()

        # Core systems
        self.vocab = TabulaRasaVocabulary()
        self.energy = ComputationalEnergy()
        self.syrinx = EmotionalSyrinx() if AUDIO_AVAILABLE else None

        # Brain (using core algorithm)
        config = GrowthConfig(
            initial_dim=64,
            max_dim=256,
            max_layers=12,
            plateau_patience=150,
            lr_boost_attempts=2
        )
        self.brain = MagnumOpusCore(vocab_size=self.vocab.max_size, config=config)
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=0.002)

        # State
        self.last_loss = 0.5
        self.current_stress = 0.0
        self.frame_count = 0
        self.last_time = time.time()

        # Camera (optional)
        self.camera = None
        if CAMERA_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            except:
                self.camera = None

        # Audio output (optional)
        self.audio_stream = None
        if AUDIO_AVAILABLE:
            try:
                self.audio_stream = sd.OutputStream(
                    samplerate=22050,
                    channels=1,
                    dtype='float32',
                    blocksize=1024,
                    callback=self._audio_callback
                )
                self.audio_stream.start()
            except:
                self.audio_stream = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio output"""
        if self.syrinx:
            chunk = self.syrinx.generate_chunk(frames)
            outdata[:, 0] = chunk
        else:
            outdata.fill(0)

    def queue_text(self, text: str):
        """Queue text for processing"""
        self.text_queue.put(text)

    def run(self):
        """Main processing loop"""
        while self.running:
            try:
                self._process_step()
            except Exception as e:
                print(f"[BRAIN] Error: {e}")
            time.sleep(0.033)  # ~30 FPS

    def _process_step(self):
        """Single processing step"""
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Get camera frame (if available)
        vision_input = np.zeros((1, 1024), dtype=np.float32)
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (32, 32))
                vision_input = small.flatten().astype(np.float32) / 255.0
                vision_input = vision_input.reshape(1, -1)

        # Get text input (if any)
        text_input = ""
        ai_response = None

        try:
            text_input = self.text_queue.get_nowait()
        except queue.Empty:
            pass

        # Process text
        if text_input:
            indices = self.vocab.encode(text_input)
            if len(indices) > 1:
                # Train on input
                x = torch.tensor([indices[:-1]], dtype=torch.long)
                y = torch.tensor([indices[1:]], dtype=torch.long)

                metrics = self.brain.training_step(x, y, self.optimizer, base_lr=0.002)
                self.last_loss = metrics['loss']

                # Check for growth
                if metrics['grew']:
                    if PYSIDE_AVAILABLE:
                        self.sig_growth.emit(metrics['layers'])
                    if self.syrinx:
                        self.syrinx.trigger_growth()

                # Maybe generate response
                if self.energy.can_afford(10000) and len(self.vocab) > 20:
                    # Simple response: sample from vocabulary
                    with torch.no_grad():
                        outputs = self.brain(x)
                        logits = outputs['logits'][0, -1, :len(self.vocab)]
                        probs = F.softmax(logits / 0.8, dim=-1)
                        sampled = torch.multinomial(probs, 3)
                        words = [self.vocab.idx_to_word.get(i.item(), "") for i in sampled]
                        ai_response = " ".join(w for w in words if w)

        # Update stress
        self.current_stress = min(1.0, self.last_loss)

        # Update syrinx
        if self.syrinx:
            self.syrinx.tension = self.last_loss
            self.syrinx.stress = self.current_stress
            self.syrinx.speak_drive = 0.8 if ai_response else 0.0

        # Calculate FPS
        fps = 1.0 / dt if dt > 0 else 30.0

        # Emit update
        if PYSIDE_AVAILABLE:
            self.sig_update.emit({
                'loss': self.last_loss,
                'stress': self.current_stress,
                'energy': self.energy.utilization,
                'vocab_size': len(self.vocab),
                'layers': len(self.brain.cortex),
                'params': self.brain.count_parameters(),
                'fps': fps,
                'ai_response': ai_response,
                'speaking': ai_response is not None
            })

    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.camera:
            self.camera.release()
        if self.audio_stream:
            self.audio_stream.stop()


# ============================================================================
# MAIN WINDOW (Qt)
# ============================================================================

if PYSIDE_AVAILABLE and PYQTGRAPH_AVAILABLE:

    class NeuralOrbWidget(gl.GLViewWidget):
        """3D neural sphere visualization"""

        def __init__(self):
            super().__init__()
            self.setCameraPosition(distance=4)
            self.setBackgroundColor('#050505')

            # Create sphere
            self.sphere = NeuralSphere(500)
            positions, colors = self.sphere.get_render_data()

            self.scatter = gl.GLScatterPlotItem(
                pos=positions,
                color=colors,
                size=3,
                pxMode=True
            )
            self.addItem(self.scatter)

            # Animation timer
            self.timer = QTimer()
            self.timer.timeout.connect(self._animate)
            self.timer.start(33)  # 30 FPS

            # State
            self.loss = 0.5
            self.stress = 0.0
            self.speaking = False

        def _animate(self):
            """Animation tick"""
            self.sphere.update(0.033, self.loss, self.stress, self.speaking, False)
            positions, colors = self.sphere.get_render_data()
            self.scatter.setData(pos=positions, color=colors)

        def trigger_growth(self):
            """Called when brain grows"""
            self.sphere.growth_wave = 1.0
            self.sphere.grow_points(30)

        def update_state(self, loss: float, stress: float, speaking: bool):
            """Update visualization state"""
            self.loss = loss
            self.stress = stress
            self.speaking = speaking


    class MainWindow(QMainWindow):
        """Main application window"""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("MagnumOpusVitalis - Living Intelligence")
            self.setGeometry(100, 100, 1400, 900)
            self.setStyleSheet("background-color: #0a0a0a; color: #00ffff;")

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)

            # Left panel: Neural orb
            left_panel = QVBoxLayout()

            self.orb = NeuralOrbWidget()
            self.orb.setMinimumSize(600, 600)
            left_panel.addWidget(self.orb)

            # Stats
            self.stats_label = QLabel("Initializing...")
            self.stats_label.setStyleSheet("color: #00ffff; font-family: monospace;")
            left_panel.addWidget(self.stats_label)

            layout.addLayout(left_panel, 2)

            # Right panel: Controls and chat
            right_panel = QVBoxLayout()

            # Energy bar
            energy_label = QLabel("ENERGY")
            energy_label.setStyleSheet("color: #ffaa00;")
            right_panel.addWidget(energy_label)

            self.energy_bar = QProgressBar()
            self.energy_bar.setStyleSheet("""
                QProgressBar { background-color: #1a1a1a; border: 1px solid #333; }
                QProgressBar::chunk { background-color: #ffaa00; }
            """)
            self.energy_bar.setValue(100)
            right_panel.addWidget(self.energy_bar)

            # Chat display
            chat_label = QLabel("NEURAL LINK")
            chat_label.setStyleSheet("color: #00ff00;")
            right_panel.addWidget(chat_label)

            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            self.chat_display.setStyleSheet("""
                QTextEdit { 
                    background-color: #0a0a0a; 
                    color: #00ff00; 
                    border: 1px solid #00ff00;
                    font-family: monospace;
                }
            """)
            right_panel.addWidget(self.chat_display)

            # Input field
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Type to communicate...")
            self.input_field.setStyleSheet("""
                QLineEdit { 
                    background-color: #1a1a1a; 
                    color: #00ffff; 
                    border: 1px solid #00ffff;
                    padding: 8px;
                    font-family: monospace;
                }
            """)
            self.input_field.returnPressed.connect(self._on_input)
            right_panel.addWidget(self.input_field)

            layout.addLayout(right_panel, 1)

            # Brain worker
            self.worker = BrainWorker()
            self.worker.sig_update.connect(self._on_update)
            self.worker.sig_growth.connect(self._on_growth)
            self.worker.start()

            self.chat_display.append("<span style='color:#555'>System initialized. Type to teach me.</span>")

        def _on_input(self):
            """Handle user input"""
            text = self.input_field.text().strip()
            if text:
                self.chat_display.append(f"<span style='color:#00ffff'>YOU: {text}</span>")
                self.worker.queue_text(text)
                self.input_field.clear()

        def _on_update(self, data: dict):
            """Handle brain update"""
            # Update stats
            self.stats_label.setText(
                f"Loss: {data['loss']:.4f} | Vocab: {data['vocab_size']} | "
                f"Layers: {data['layers']} | Params: {data['params']:,} | FPS: {data['fps']:.0f}"
            )

            # Update energy bar
            self.energy_bar.setValue(int((1 - data['energy']) * 100))

            # Update orb
            self.orb.update_state(data['loss'], data['stress'], data['speaking'])

            # Show AI response
            if data.get('ai_response'):
                self.chat_display.append(
                    f"<span style='color:#00ff00'>AI: {data['ai_response']}</span>"
                )

        def _on_growth(self, layers: int):
            """Handle growth event"""
            self.orb.trigger_growth()
            self.chat_display.append(
                f"<b style='color:#ff3300'>*** NEURAL EXPANSION *** Layers: {layers}</b>"
            )

        def closeEvent(self, event):
            """Clean shutdown"""
            self.worker.stop()
            self.worker.wait()
            event.accept()


# ============================================================================
# CONSOLE MODE (fallback)
# ============================================================================

def run_console_mode():
    """Run in console if Qt not available"""
    print("=" * 60)
    print("MagnumOpusVitalis - Console Mode")
    print("=" * 60)
    print("Type text to teach the AI. Type 'quit' to exit.")
    print()

    vocab = TabulaRasaVocabulary()
    config = GrowthConfig(initial_dim=64, max_dim=256, max_layers=8)
    brain = MagnumOpusCore(vocab_size=vocab.max_size, config=config)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=0.002)

    while True:
        try:
            text = input("YOU: ").strip()
            if text.lower() == 'quit':
                break

            if not text:
                continue

            # Encode and train
            indices = vocab.encode(text)
            if len(indices) > 1:
                x = torch.tensor([indices[:-1]], dtype=torch.long)
                y = torch.tensor([indices[1:]], dtype=torch.long)

                metrics = brain.training_step(x, y, optimizer, base_lr=0.002)

                print(f"     [Loss: {metrics['loss']:.4f}, Vocab: {len(vocab)}, "
                      f"Layers: {metrics['layers']}, Params: {metrics['params']:,}]")

                if metrics['grew']:
                    print("     *** GROWTH EVENT ***")

                # Generate response
                with torch.no_grad():
                    outputs = brain(x)
                    logits = outputs['logits'][0, -1, :len(vocab)]
                    probs = F.softmax(logits / 0.8, dim=-1)
                    sampled = torch.multinomial(probs, min(3, len(vocab)))
                    words = [vocab.idx_to_word.get(i.item(), "") for i in sampled]
                    response = " ".join(w for w in words if w)
                    if response:
                        print(f"AI:  {response}")

            print()

        except KeyboardInterrupt:
            break

    print("\nShutting down...")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if PYSIDE_AVAILABLE and PYQTGRAPH_AVAILABLE:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        run_console_mode()


if __name__ == "__main__":
    print("=" * 70)
    print("  MAGNUMOPUSVITALIS DEMO")
    print("  'A seed that grows, not a machine that thinks.'")
    print("=" * 70)
    print()
    main()