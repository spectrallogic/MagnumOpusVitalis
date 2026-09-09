"""
Emotion and temporal vector extraction via contrastive activation differencing.
Directions are hypotheses about model features, validated by intervention.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from magnum_opus_v2.prompts import EMOTION_PROMPT_PAIRS, TEMPORAL_PROMPT_PAIRS
from magnum_opus_v2.steering_hook import SteeringHook


def read_block_output(model, inputs: dict, target_layer: int) -> torch.Tensor:
    """Read the exact output boundary used by SteeringHook (zero-based block).

    HF hidden_states[0] is the embedding output; the final entry may also
    include final normalization. A block hook avoids both ambiguities.
    Callers using an engine must hold its model lock and disable steering.
    """
    layers = SteeringHook._layer_list(model)
    if not 0 <= target_layer < len(layers):
        raise ValueError("target_layer is outside the model's transformer blocks")
    captured = []

    def capture(module, args, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        captured.append(hidden.detach().float().clone())

    handle = layers[target_layer].register_forward_hook(capture)
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
        model.train(training)
    if len(captured) != 1:
        raise ValueError("Expected one target-block execution per measurement")
    hidden = captured[0]
    mask = inputs.get("attention_mask")
    if mask is None:
        return hidden.mean(dim=1).squeeze(0)
    weights = mask.to(hidden.device, hidden.dtype).unsqueeze(-1)
    return ((hidden * weights).sum(dim=1) /
            weights.sum(dim=1).clamp_min(1)).squeeze(0)


def extract_hidden_states(
    model, tokenizer, prompts: List[str], target_layer: int, device: str,
) -> torch.Tensor:
    """Extract mean hidden states at a target layer, averaged across prompts."""
    all_states = []
    if not prompts:
        raise ValueError("At least one prompt is required for extraction")
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128,
        ).to(device)
        mean_state = read_block_output(model, inputs, target_layer)
        all_states.append(mean_state)
    return torch.stack(all_states).mean(dim=0)


def extract_vectors(
    model, tokenizer, target_layer: int, device: str,
    emotion_pairs: Optional[Dict] = None,
    temporal_pairs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Extract emotion and temporal direction vectors using contrastive activation
    differencing: vector = mean(positive_activations) - mean(negative_activations).

    Returns dict mapping vector names to normalized direction vectors.
    """
    if emotion_pairs is None:
        emotion_pairs = EMOTION_PROMPT_PAIRS
    if temporal_pairs is None:
        temporal_pairs = TEMPORAL_PROMPT_PAIRS

    vectors = {}

    if verbose:
        print(f"\n  Extracting vectors at layer {target_layer}...")

    for name, pairs in emotion_pairs.items():
        if verbose:
            print(f"    {name}...", end=" ", flush=True)
        pos = extract_hidden_states(model, tokenizer, pairs["positive"], target_layer, device)
        neg = extract_hidden_states(model, tokenizer, pairs["negative"], target_layer, device)
        direction = pos - neg
        raw_norm = direction.norm().item()
        direction = _normalize_direction(direction, name)
        vectors[name] = direction
        if verbose:
            print(f"(norm={raw_norm:.3f})")

    for name, pairs in temporal_pairs.items():
        key = f"temporal_{name}"
        if verbose:
            print(f"    {key}...", end=" ", flush=True)
        pos = extract_hidden_states(model, tokenizer, pairs["positive"], target_layer, device)
        neg = extract_hidden_states(model, tokenizer, pairs["negative"], target_layer, device)
        direction = pos - neg
        direction = _normalize_direction(direction, key)
        vectors[key] = direction
        if verbose:
            print("done")

    if verbose:
        print(f"\n  Extracted {len(vectors)} vectors total.")
        _print_similarity_matrix(vectors)

    return vectors


def _normalize_direction(direction: torch.Tensor, name: str) -> torch.Tensor:
    norm = direction.norm()
    if not torch.isfinite(direction).all() or not torch.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Contrast for {name!r} has no finite nonzero direction")
    return direction / norm


def _print_similarity_matrix(vectors: Dict[str, torch.Tensor]):
    """Print cosine similarity between all extracted vectors."""
    names = list(vectors.keys())
    if len(names) > 12:
        names = [n for n in names if not n.startswith("temporal_")] + \
                [n for n in names if n.startswith("temporal_")]

    print(f"\n  Cosine similarities ({len(names)} vectors):")
    header = "              " + "".join(f"{n[:8]:>10}" for n in names)
    print(header)
    for n1 in names:
        row = f"  {n1[:10]:>12}"
        for n2 in names:
            sim = F.cosine_similarity(
                vectors[n1].unsqueeze(0), vectors[n2].unsqueeze(0),
            ).item()
            row += f"{sim:>10.3f}"
        print(row)
