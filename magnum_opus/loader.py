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

    # Load with appropriate precision for the device.
    # First try pure GPU in float16; if that OOMs, use explicit max_memory
    # spill to CPU RAM (not "meta" or disk — custom hooks break on meta
    # device tensors). Last resort is pure CPU.
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                raise
            torch.cuda.empty_cache()
            # Find free VRAM and leave ~1GB headroom for activations
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            gpu_budget_gib = max(2, int(free_bytes / (1024 ** 3)) - 1)
            print(f"  Model too large for VRAM alone — spilling to CPU RAM "
                  f"(GPU budget: {gpu_budget_gib}GiB)")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={0: f"{gpu_budget_gib}GiB", "cpu": "48GiB"},
                    trust_remote_code=True,
                )
            except Exception:
                print(f"  Spill load failed — falling back to pure CPU")
                torch.cuda.empty_cache()
                device = "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).to(device)
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
