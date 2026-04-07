"""
Emotion and temporal vector extraction via contrastive activation differencing.
Following Anthropic's methodology adapted for any HuggingFace causal LM.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from magnum_opus.config import EMOTION_PROMPT_PAIRS, TEMPORAL_PROMPT_PAIRS


def extract_hidden_states(
    model, tokenizer, prompts: List[str], target_layer: int, device: str,
) -> torch.Tensor:
    """Extract mean hidden states at a target layer, averaged across prompts."""
    all_states = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[target_layer]
        mean_state = hidden.mean(dim=1).squeeze(0).float()
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

    # Emotion vectors
    for name, pairs in emotion_pairs.items():
        if verbose:
            print(f"    {name}...", end=" ", flush=True)
        pos = extract_hidden_states(model, tokenizer, pairs["positive"], target_layer, device)
        neg = extract_hidden_states(model, tokenizer, pairs["negative"], target_layer, device)
        direction = pos - neg
        raw_norm = direction.norm().item()
        direction = direction / direction.norm()
        vectors[name] = direction
        if verbose:
            print(f"(norm={raw_norm:.3f})")

    # Temporal vectors
    for name, pairs in temporal_pairs.items():
        key = f"temporal_{name}"
        if verbose:
            print(f"    {key}...", end=" ", flush=True)
        pos = extract_hidden_states(model, tokenizer, pairs["positive"], target_layer, device)
        neg = extract_hidden_states(model, tokenizer, pairs["negative"], target_layer, device)
        direction = pos - neg
        direction = direction / direction.norm()
        vectors[key] = direction
        if verbose:
            print("done")

    if verbose:
        print(f"\n  Extracted {len(vectors)} vectors total.")
        _print_similarity_matrix(vectors)

    return vectors


def _print_similarity_matrix(vectors: Dict[str, torch.Tensor]):
    """Print cosine similarity between all extracted vectors."""
    names = list(vectors.keys())
    if len(names) > 12:
        # Truncate for readability
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
