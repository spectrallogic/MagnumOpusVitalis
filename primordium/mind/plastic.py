"""
Plastic variants of engine regions — born empty, grown into.

The engine's Limbic and SubconsciousStack cache their emotion vectors at
construction; Primordium is born affectless and DISCOVERS its affects
from lived experience hours later. These subclasses add a hot-swap that
installs discovered directions under the regions' own locks. The parent
classes in magnum_opus_v2 are never modified.
"""

from typing import Dict, Optional

import torch

from magnum_opus_v2.regions.limbic import Limbic
from magnum_opus_v2.regions.subconscious import SubconsciousStack
from magnum_opus_v2._dynamics import MultiSpeedEmotionalState


class MoodField(Limbic):
    def set_vectors(self, vectors: Dict[str, torch.Tensor],
                    configs: Optional[dict] = None) -> None:
        """Install discovered affect directions (and their measured
        dynamics). Emotional state restarts at the new baselines — the
        moment of first feeling is a birth, not an edit."""
        with self._lock:
            self._emo_vecs = {
                n: v.detach().float().to(self.device)
                for n, v in vectors.items()
            }
            self._state = MultiSpeedEmotionalState(
                emotion_names=list(vectors.keys()),
                configs=configs, interactions={},
            )


class Undertow(SubconsciousStack):
    def set_affect_vectors(self, vectors: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self._emo_vecs = {
                n: v.detach().float().to(self.device)
                for n, v in vectors.items()
            }
