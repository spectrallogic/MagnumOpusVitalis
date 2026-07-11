"""
PresenceSensor — is someone here with me?

Cheap heuristics only (Haar face detection + a speechiness measure), and
they feed nothing but the social drive — no safety-relevant decision
depends on them. False positives (a photo, a TV) are acceptable noise in
a hunger signal.
"""

import numpy as np

try:
    import cv2
    _CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:  # noqa: BLE001 — degraded but alive
    cv2 = None
    _CASCADE = None


class PresenceSensor:
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.last = {"face": False, "faces_n": 0, "speechiness": 0.0,
                     "audio_rms": 0.0}

    def measure(self, frame_rgb, pcm_f32, self_heard: float = 0.0) -> dict:
        faces_n = 0
        if frame_rgb is not None and _CASCADE is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = _CASCADE.detectMultiScale(gray, 1.2, 4, minSize=(20, 20))
            faces_n = len(faces)

        speechiness = 0.0
        rms = 0.0
        if pcm_f32 is not None and len(pcm_f32) > 1024:
            x = pcm_f32 - pcm_f32.mean()
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            if rms > 1e-4:
                spec = np.abs(np.fft.rfft(x))
                freqs = np.fft.rfftfreq(len(x), 1.0 / self.sr)
                band = spec[(freqs > 300) & (freqs < 3000)].sum()
                total = spec.sum() + 1e-9
                band_ratio = float(band / total)
                # syllabic modulation: energy of the 3–8 Hz amplitude envelope
                env = np.abs(x)
                hop = max(1, self.sr // 100)
                env = env[: len(env) - len(env) % hop].reshape(-1, hop).mean(1)
                if len(env) > 16:
                    espec = np.abs(np.fft.rfft(env - env.mean()))
                    efreq = np.fft.rfftfreq(len(env), hop / self.sr)
                    mod = float(espec[(efreq > 3) & (efreq < 8)].sum()
                                / (espec.sum() + 1e-9))
                else:
                    mod = 0.0
                speechiness = min(1.0, band_ratio * 1.4) * min(1.0, mod * 4.0)
                # If it's mostly hearing itself, that's not company.
                speechiness *= max(0.0, 1.0 - self_heard)

        self.last = {
            "face": faces_n > 0,
            "faces_n": int(faces_n),
            "speechiness": round(float(speechiness), 4),
            "audio_rms": round(rms, 5),
        }
        return self.last
