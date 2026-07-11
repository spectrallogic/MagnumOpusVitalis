"""
Lodestar — a star to steer by, never a place to arrive.

Frozen pretrained weights, used INDIRECTLY: the organism never runs
inference through them to act, think, or speak. A vision teacher (the
DaViT tower of a local Florence-2 snapshot) occasionally looks at the
same frame the organism saw and hands back one feature vector; a small
scaffold aligns the organism's own embedding toward it, with a weight
that ANNEALS to zero as its own predictions improve and hard-cuts at
stage 2. Like inherited fear of snakes: a leaning inherited from
another lineage's experience, dissolving under lived experience.
The falsifiable contract lives in the tests: once annealed, removing
the teacher must not change prediction loss.

Everything loads local_files_only — no downloads, nothing leaves.
"""

import time
from typing import Optional

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VisionTeacher:
    """Frozen DaViT (Florence-2 vision tower + projection), fp16, no_grad.

    `embed_frame` maps one RGB frame to a pooled feature (feat_dim,).
    `unload` frees the weights the moment the organism outgrows them.
    """

    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self._full = None            # fallback keeps the whole Florence-2
        self.feat_dim = cfg.lodestar_feat_dim
        self.load_error = ""
        self.last_ms = 0.0
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        name = self.cfg.lodestar_model
        try:                          # vision tower alone (preferred)
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module)
            vis_cfg = AutoConfig.from_pretrained(
                name, trust_remote_code=True,
                local_files_only=True).vision_config
            cls = get_class_from_dynamic_module(
                "modeling_florence2.Florence2VisionModelWithProjection", name)
            # the snapshot's remote code predates transformers' sdpa
            # dispatch; eager attention is correct and fast enough (~40ms)
            cls._supports_sdpa = False
            self.model = cls.from_pretrained(
                name, config=vis_cfg, trust_remote_code=True,
                local_files_only=True, torch_dtype=self.dtype,
                attn_implementation="eager").to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            return
        except Exception as e:  # noqa: BLE001
            self.load_error = f"vision tower: {str(e)[:200]}"
        try:                          # whole model, use its image encoder
            from transformers import AutoModelForCausalLM
            self._full = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True, local_files_only=True,
                torch_dtype=self.dtype).to(self.device).eval()
            for p in self._full.parameters():
                p.requires_grad_(False)
        except Exception as e:  # noqa: BLE001
            self.load_error += f" | full model: {str(e)[:200]}"

    @property
    def ok(self) -> bool:
        return self.model is not None or self._full is not None

    def unload(self) -> None:
        """Stage 2: the star is no longer needed. Free the memory."""
        self.model = None
        self._full = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(n, H, W, 3) uint8 RGB -> (n, 3, 768, 768) normalized."""
        out = np.empty((len(frames), 768, 768, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            big = cv2.resize(f, (768, 768), interpolation=cv2.INTER_LINEAR)
            out[i] = (big.astype(np.float32) / 255.0 - IMAGENET_MEAN) \
                / IMAGENET_STD
        return torch.from_numpy(out).permute(0, 3, 1, 2)

    @torch.no_grad()
    def _forward(self, px: torch.Tensor) -> Optional[torch.Tensor]:
        px = px.to(self.device, dtype=self.dtype)
        if self.model is not None:
            out = self.model(pixel_values=px)
            if isinstance(out, torch.Tensor):
                feat = out                       # (n, tokens, 1024)
            else:
                feat = getattr(out, "image_embeds", None)
                if feat is None:
                    feat = getattr(out, "last_hidden_state", None)
                if feat is None and isinstance(out, (tuple, list)) and out:
                    feat = out[0]
        elif self._full is not None:
            feat = self._full._encode_image(px)  # noqa: SLF001
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
        else:
            return None
        if feat is None:
            return None
        if feat.dim() == 3:                      # (n, tokens, D) -> pool
            feat = feat.mean(1)
        if feat.shape[-1] != self.feat_dim:
            self.load_error = (f"feature dim {feat.shape[-1]} != "
                               f"{self.feat_dim}")
            self.unload()
            return None
        return feat.float().cpu()

    def embed_frame(self, frame_rgb: np.ndarray) -> Optional[torch.Tensor]:
        """One frame -> (feat_dim,) float32 cpu, timed honestly."""
        if not self.ok or frame_rgb is None:
            return None
        t0 = time.monotonic()
        feat = self._forward(self._preprocess(frame_rgb[None]))
        self.last_ms = (time.monotonic() - t0) * 1000.0
        return feat[0] if feat is not None else None

    def embed_batch(self, frames: np.ndarray,
                    chunk: int = 8) -> Optional[torch.Tensor]:
        """(n, H, W, 3) -> (n, feat_dim); chunked to bound VRAM."""
        if not self.ok or len(frames) == 0:
            return None
        outs = []
        for i in range(0, len(frames), chunk):
            f = self._forward(self._preprocess(frames[i:i + chunk]))
            if f is None:
                return None
            outs.append(f)
        return torch.cat(outs, 0)


class TextTeacher:
    """Frozen MiniLM on CPU: each incoming message becomes ONE quiet
    384-d target the Wordstream's latents lean toward — annealed away
    with lived language (word_tau_msgs). Never on a decision path."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tok = None
        self.load_error = ""
        try:
            from transformers import AutoModel, AutoTokenizer
            name = cfg.text_teacher_model
            self.tok = AutoTokenizer.from_pretrained(
                name, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                name, local_files_only=True).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        except Exception as e:  # noqa: BLE001
            self.load_error = str(e)[:200]

    @property
    def ok(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not self.ok or not text.strip():
            return None
        enc = self.tok(text[:512], return_tensors="pt",
                       truncation=True, max_length=128)
        out = self.model(**enc).last_hidden_state          # (1, T, 384)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        v = torch.nn.functional.normalize(v, dim=-1)[0]
        return v.numpy().astype(np.float32)
