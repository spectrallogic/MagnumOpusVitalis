"""
Brain regions for v2. Each region implements the Region protocol and runs
on one of the four clocks (flow / perception / expensive / slow).
"""

from magnum_opus_v2.regions.limbic import Limbic
from magnum_opus_v2.regions.temporal import Temporal
from magnum_opus_v2.regions.subconscious import (
    SubconsciousStack, Candidate, CandidateSampler,
    NoiseSampler, TokenEmbeddingSampler, MemorySampler,
)
from magnum_opus_v2.regions.memory import Memory, FalseMemoryConfabulator
from magnum_opus_v2.regions.salience import Salience
from magnum_opus_v2.regions.executive import Executive
from magnum_opus_v2.regions.default_mode import DefaultMode
from magnum_opus_v2.regions.knowledge_sparks import (
    KnowledgeSparks, KnowledgeSparksFire,
)
from magnum_opus_v2.regions.speculative import (
    SpeculativeFutures, SpeculativePenumbra,
)
from magnum_opus_v2.regions.abstraction import AbstractionLadder
from magnum_opus_v2.regions.self_model import SelfModel
from magnum_opus_v2.regions.consolidation import Consolidation
from magnum_opus_v2.regions.situation import SituationModel

__all__ = [
    "Limbic", "Temporal",
    "SubconsciousStack", "Candidate", "CandidateSampler",
    "NoiseSampler", "TokenEmbeddingSampler", "MemorySampler",
    "Memory", "FalseMemoryConfabulator",
    "Salience", "Executive",
    "DefaultMode", "KnowledgeSparks", "KnowledgeSparksFire",
    "SpeculativeFutures", "SpeculativePenumbra",
    "AbstractionLadder", "SelfModel", "Consolidation", "SituationModel",
]
