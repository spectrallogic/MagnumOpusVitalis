"""
MindsEyeSampler — the subconscious drinks from the imagined stream.

Every tick the core predicts the next moment; those predicted latents
are its live understanding of reality. This sampler exposes a rolling
ring of them (plus dream pools) as L0 material for the SubconsciousStack:
the sea of bits now contains the organism's OWN generated reality, and
the resonance filters decide which imagined fragments surface as
intrusive thoughts.

The ring always runs — a mind's eye is not optional equipment. The UI
toggle only controls whether we RENDER the stream for human eyes and
ears (and rendering costs energy).
"""

import threading
from collections import deque
from typing import List, Optional

import torch

from magnum_opus_v2.regions.subconscious import Candidate, CandidateSampler


class MindsEyeRing:
    def __init__(self, maxlen: int = 64):
        self.ring: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, vec: torch.Tensor, kind: str = "prediction") -> None:
        v = vec.detach().float().cpu()
        if float(v.norm()) < 1e-6:
            return
        with self._lock:
            self.ring.append((v, kind))

    def sample(self, n: int) -> List[Candidate]:
        with self._lock:
            items = list(self.ring)
        if not items:
            return []
        idx = torch.randint(0, len(items), (min(n, len(items)),))
        out = []
        for i in idx.tolist():
            v, kind = items[i]
            out.append(Candidate(vec=v, source=f"minds_eye:{kind}",
                                 confidence=0.8, meta={"kind": kind}))
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self.ring)


class MindsEyeSampler(CandidateSampler):
    name = "minds_eye"

    def __init__(self, ring: MindsEyeRing):
        self.ring = ring

    def sample(self, n: int, bias: Optional[torch.Tensor] = None):
        cands = self.ring.sample(n)
        if bias is None or not cands:
            return cands
        # lightly prefer imagined fragments that resonate with the bias
        b = bias.detach().float().cpu()
        bn = float(b.norm())
        if bn < 1e-6:
            return cands
        b = b / bn
        scored = sorted(
            cands,
            key=lambda c: -float(torch.dot(
                c.vec / (c.vec.norm() + 1e-8), b)))
        return scored[: max(1, n)]
