"""Load local Transformers causal LMs with access to their transformer blocks."""

import gc

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from magnum_opus_v2.model_sources import validate_model_source


def load_model(model_name: str = "gpt2", device: str = None, *,
               local_files_only: bool = False, trust_remote_code: bool = False):
    """Return (model, tokenizer, device); the launcher defaults to local files only.

    Supported hooks require a recognized transformer block list. A local server's
    chat API and GGUF files cannot provide these hooks. Custom model code is
    opt-in. CPU fallback uses one complete model, avoiding unsupported sharding.
    """
    model_name = validate_model_source(model_name)
    if device in (None, "auto"):
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                  else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation. Use --device cpu or install a CUDA-enabled PyTorch build.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")

    print(f"  Loading '{model_name}' on {device}...", flush=True)
    options = dict(local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(model_name, **options)
    if getattr(config, "quantization_config", None):
        raise ValueError("Pre-quantized checkpoints need a separately verified hook/device adapter. Use the standard Transformers weights for this launcher.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("This tokenizer has neither a padding nor an EOS token; it needs a tokenizer adapter.")
        tokenizer.pad_token = tokenizer.eos_token

    # Keep the reference before .to(): an OOM may have moved only some layers.
    # Release that object before reloading so two model copies cannot stay live.
    model = None
    cpu_fallback = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            **options,
        )
        if getattr(model, "is_quantized", False):
            raise ValueError("Pre-quantized checkpoints need a separately verified hook/device adapter. Use the standard Transformers weights for this launcher.")
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        model = None
        cpu_fallback = True
    if cpu_fallback:
        # Exit the exception handler first: its traceback can retain the old model.
        gc.collect()
        torch.cuda.empty_cache()
        print("  Model exceeds available VRAM. Loading on CPU; responses will be slower.", flush=True)
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **options)

    model.eval()
    from magnum_opus_v2.steering_hook import SteeringHook
    try:
        layers = SteeringHook._layer_list(model)
    except ValueError as exc:
        raise ValueError(
            f"{type(model).__name__} needs a Vitalis block adapter. Currently supported "
            "layouts include GPT-2, Llama/Qwen/Mistral, GPT-NeoX, and OPT transformer blocks."
        ) from exc
    if not layers:
        raise ValueError("The model has no transformer blocks.")
    hidden_dim = model.get_input_embeddings().weight.shape[1]
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {len(layers)} layers, {hidden_dim}d hidden, {param_count:.1f}M params")
    return model, tokenizer, device
