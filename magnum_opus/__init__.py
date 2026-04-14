"""
Magnum Opus Vitalis - Live Latent Space Engine
================================================
A stateful engine that steers a frozen LLM's latent space
using biological dynamics, temporal awareness, reconstructive
memory, subconscious exploration, and dream consolidation.

Usage with a saved profile (recommended):
    from magnum_opus import MagnumOpusEngine, load_model, load_profile

    # First time: python -m magnum_opus.profile create gpt2
    model, tokenizer, device = load_model("gpt2")
    profile = load_profile("gpt2")
    engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)

Or extract vectors on the fly:
    from magnum_opus import MagnumOpusEngine, load_model, extract_vectors

    model, tokenizer, device = load_model("gpt2")
    vectors = extract_vectors(model, tokenizer, target_layer=6, device=device)
    engine = MagnumOpusEngine(model, tokenizer, vectors, device=device)

Then:
    response = engine.converse("Hello, how are you?")
    engine.dream()
    engine.save("my_engine_state.json")

Author: Alan Hourmand (April 2026)
Repository: https://github.com/spectrallogic/MagnumOpusVitalis
"""

__version__ = "1.1.0"

from magnum_opus.engine import MagnumOpusEngine, measure_projections
from magnum_opus.extraction import extract_vectors, extract_hidden_states
from magnum_opus.loader import load_model
from magnum_opus.profile import ModelProfile, create_profile, load_profile, list_profiles
