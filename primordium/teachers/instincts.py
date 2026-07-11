"""
Instincts — inherited leanings, honestly labeled.

Newborns arrive with a few perceptual biases (top-heavy face-like
patterns, alarm at looming, unease in darkness). Primordium gets the
same head start without hand-coding feelings about the REAL world:
each instinct is a PROCEDURAL probe set — synthetic images we can print
and inspect — whose teacher features, projected through the organism's
own scaffold, define a direction ("needle") in its embedding space.
The ValenceCompass stores these with provenance="innate" at reduced
weight, and RETIRES them once enough lived affects exist or stage 2
arrives. Unlike biology, every inherited fear here knows its origin
and has a expiry date.
"""

from typing import Dict

import numpy as np


def _canvas(rng, res: int, lum: float, noise: float = 6.0) -> np.ndarray:
    base = np.full((res, res, 3), lum, dtype=np.float32)
    base += rng.normal(0, noise, size=(res, res, 3)).astype(np.float32)
    return base


def probe_face_like(rng, res: int) -> np.ndarray:
    """Two dark dots above a light blob — the top-heavy pattern newborns
    track from the first hours."""
    img = _canvas(rng, res, 105.0)
    cx = res // 2 + int(rng.integers(-6, 7))
    cy = res // 2 + int(rng.integers(-4, 5))
    yy, xx = np.ogrid[:res, :res]
    rx, ry = res * 0.28, res * 0.38
    blob = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1.0
    img[blob] = 170.0 + float(rng.normal(0, 8))
    for sx in (-1, 1):                                    # the two dots
        ex = cx + sx * int(res * 0.11) + int(rng.integers(-2, 3))
        ey = cy - int(res * 0.12) + int(rng.integers(-2, 3))
        dot = ((xx - ex) ** 2 + (yy - ey) ** 2) < (res * 0.045) ** 2
        img[dot] = 30.0
    my = cy + int(res * 0.16)
    mouth = (np.abs(yy - my) < res * 0.02) & (np.abs(xx - cx) < res * 0.09)
    img[mouth] = 60.0
    return np.clip(img, 0, 255).astype(np.uint8)


def probe_looming(rng, res: int) -> np.ndarray:
    """A dark disc filling the visual field — the geometry of approach."""
    img = _canvas(rng, res, 130.0)
    cx = res // 2 + int(rng.integers(-5, 6))
    cy = res // 2 + int(rng.integers(-5, 6))
    rad = res * (0.36 + float(rng.uniform(0, 0.12)))
    yy, xx = np.ogrid[:res, :res]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < rad * rad
    img[disc] = 18.0 + float(rng.normal(0, 4))
    return np.clip(img, 0, 255).astype(np.uint8)


def probe_darkness(rng, res: int) -> np.ndarray:
    """Almost nothing to see."""
    return np.clip(_canvas(rng, res, 12.0, noise=4.0), 0, 255).astype(np.uint8)


def probe_open_bright(rng, res: int) -> np.ndarray:
    """Bright open field, light from above — the safe direction."""
    grad = np.linspace(210.0, 120.0, res, dtype=np.float32)
    img = np.repeat(grad[:, None], res, axis=1)[..., None].repeat(3, axis=2)
    img = img + rng.normal(0, 5, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


PROBES = {
    "face_like": probe_face_like,
    "looming": probe_looming,
    "darkness": probe_darkness,
    "open_bright": probe_open_bright,
}


def probe_frames(name: str, n: int, res: int, seed: int = 7) -> np.ndarray:
    """n jittered instances of one probe, deterministic per (name, seed)."""
    rng = np.random.default_rng(seed + sum(name.encode()))
    return np.stack([PROBES[name](rng, res) for _ in range(n)])


def all_probe_frames(res: int, n: int, seed: int = 7
                     ) -> Dict[str, np.ndarray]:
    return {name: probe_frames(name, n, res, seed) for name in PROBES}
