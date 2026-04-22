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

__all__ = [
    "Limbic", "Temporal",
    "SubconsciousStack", "Candidate", "CandidateSampler",
    "NoiseSampler", "TokenEmbeddingSampler", "MemorySampler",
    "Memory", "FalseMemoryConfabulator",
    "Salience", "Executive",
    "DefaultMode", "KnowledgeSparks", "KnowledgeSparksFire",
]
