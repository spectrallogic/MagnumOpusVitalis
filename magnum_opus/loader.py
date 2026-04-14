"""Model loading utilities. Supports any HuggingFace causal LM."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "gpt2", device: str = None):
    """
    Load a HuggingFace causal language model.

    Args:
        model_name: Any HuggingFace model ID (gpt2, gpt2-medium, meta-llama/Llama-3-8B, etc.)
        device: "cpu", "cuda", "mps", or None for auto-detect

    Returns:
        (model, tokenizer, device_str)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"  Loading '{model_name}' on {device}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load with appropriate precision for the device
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
        ).to(device)

    model.eval()

    # Detect architecture
    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError(f"Cannot detect layer count for model {model_name}")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_layers} layers, {hidden_dim}d hidden, {param_count:.1f}M params")

    return model, tokenizer, device
