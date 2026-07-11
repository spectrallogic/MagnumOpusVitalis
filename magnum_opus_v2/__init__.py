"""
Magnum Opus Vitalis — substrate-shaped runtime.

A shared continuous latent state (LatentBus) with attractor dynamics, populated
by uniform brain regions running on a multi-rate clock so the system never
stops processing. Steers a frozen LLM at inference time via direction vectors
extracted from its own latent geometry.

Quick start:
    from magnum_opus_v2 import V2Engine, load_model, load_profile

    model, tokenizer, device = load_model("gpt2")
    profile = load_profile("gpt2")  # or create_profile("gpt2")
    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
    engine.start()
    print(engine.converse("Hello, how are you?"))
    engine.stop()
"""

__version__ = "2.0.0"

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
    SpeculativeFutures, SpeculativePenumbra,
    AbstractionLadder, SelfModel,
)
from magnum_opus_v2.loader import load_model
from magnum_opus_v2.profile import (
    ModelProfile, ProfileMetadata, Baseline,
    create_profile, load_profile, save_profile,
    list_profiles, profile_exists, delete_profile,
)
from magnum_opus_v2.extraction import extract_vectors, extract_hidden_states
from magnum_opus_v2.prompts import EMOTION_PROMPT_PAIRS, TEMPORAL_PROMPT_PAIRS

__all__ = [
    # Engine
    "V2Engine",
    "V2Config", "BusConfig", "ClockConfig",
    # Substrate
    "LatentBus", "FlowRunner",
    "Region", "NoOpRegion", "PerturbationRegion",
    "SteeringHook", "BusSteeringDriver",
    "NeuromodState", "NeuromodulatorRegion",
    # Regions
    "Limbic", "Temporal",
    "SubconsciousStack", "Candidate", "CandidateSampler",
    "NoiseSampler", "TokenEmbeddingSampler", "MemorySampler",
    "Memory", "FalseMemoryConfabulator",
    "Salience", "Executive",
    "DefaultMode", "KnowledgeSparks", "KnowledgeSparksFire",
    "SpeculativeFutures", "SpeculativePenumbra",
    "AbstractionLadder", "SelfModel",
    # Model loading / profiles / extraction
    "load_model",
    "ModelProfile", "ProfileMetadata", "Baseline",
    "create_profile", "load_profile", "save_profile",
    "list_profiles", "profile_exists", "delete_profile",
    "extract_vectors", "extract_hidden_states",
    "EMOTION_PROMPT_PAIRS", "TEMPORAL_PROMPT_PAIRS",
]
