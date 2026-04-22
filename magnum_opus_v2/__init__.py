"""
Magnum Opus Vitalis v2 — substrate-shaped runtime.

Greenfield rebuild around a shared continuous latent state (LatentBus) with
attractor dynamics, populated by uniform brain regions running on a
multi-rate clock so the system never stops processing.

v1 (magnum_opus/) stays runnable for A/B comparison.

Public API will grow as regions land. Substrate-only entry points first:
    from magnum_opus_v2 import LatentBus, FlowRunner, V2Config, NoOpRegion
"""

__version__ = "2.0.0-substrate"

from magnum_opus_v2.config import V2Config, BusConfig, ClockConfig
from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.flow import FlowRunner
from magnum_opus_v2.region import Region, NoOpRegion, PerturbationRegion
from magnum_opus_v2.steering_hook import SteeringHook, BusSteeringDriver
from magnum_opus_v2.neuromod import NeuromodState, NeuromodulatorRegion
from magnum_opus_v2.engine import V2Engine
from magnum_opus_v2.regions import (
    Limbic, Temporal, SubconsciousStack, Candidate, CandidateSampler,
    NoiseSampler, TokenEmbeddingSampler, MemorySampler,
    Memory, FalseMemoryConfabulator,
    Salience, Executive,
    DefaultMode, KnowledgeSparks, KnowledgeSparksFire,
)

__all__ = [
    "V2Engine",
    "V2Config", "BusConfig", "ClockConfig",
    "LatentBus", "FlowRunner",
    "Region", "NoOpRegion", "PerturbationRegion",
    "SteeringHook", "BusSteeringDriver",
    "NeuromodState", "NeuromodulatorRegion",
    "Limbic", "Temporal",
    "SubconsciousStack", "Candidate", "CandidateSampler",
    "NoiseSampler", "TokenEmbeddingSampler", "MemorySampler",
    "Memory", "FalseMemoryConfabulator",
    "Salience", "Executive",
    "DefaultMode", "KnowledgeSparks", "KnowledgeSparksFire",
]
