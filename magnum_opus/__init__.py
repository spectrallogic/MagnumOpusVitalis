"""
Magnum Opus Vitalis - Live Latent Space Engine
================================================
A stateful engine that steers a frozen LLM's latent space
using biological dynamics, temporal awareness, reconstructive
memory, subconscious exploration, and dream consolidation.

Usage:
    from magnum_opus import MagnumOpusEngine, load_model, extract_vectors

    model, tokenizer, device = load_model("gpt2")
    vectors = extract_vectors(model, tokenizer, target_layer=6, device=device)
    engine = MagnumOpusEngine(model, tokenizer, vectors, device=device)

    # Conversation
    response = engine.converse("Hello, how are you?")
    print(response)

    # The engine tracks emotional state, time, memory, subconscious goals
    print(engine.status())

    # Run a dream cycle during idle time
    engine.dream()

    # Save state for next session
    engine.save("my_engine_state.json")

Author: Alan Hourmand (April 2026)
Repository: https://github.com/spectrallogic/MagnumOpusVitalis
"""

__version__ = "1.0.0"

from magnum_opus.engine import MagnumOpusEngine
from magnum_opus.extraction import extract_vectors, extract_hidden_states
from magnum_opus.loader import load_model
