"""Local model discovery and portable names for model-specific artifacts."""

import hashlib
import json
import os
import re
from pathlib import Path


def canonical_model_source(source: str) -> str:
    """Resolve folders once so relative paths cannot alias profiles across cwd's."""
    path = Path(source).expanduser()
    return os.path.normcase(str(path.resolve())) if path.is_dir() else source


def model_storage_key(source: str) -> str:
    """Keep existing Hub profile names; hash filesystem paths for Windows safety."""
    if Path(source).is_absolute() or "\\" in source or re.match(r"^[A-Za-z]:", source):
        leaf = re.split(r"[/\\]", source.rstrip("/\\"))[-1]
        leaf = re.sub(r"[^A-Za-z0-9_.-]", "_", leaf)[:48] or "model"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return f"local--{leaf}--{digest}"
    # Hub names are repo IDs, not filesystem paths. No traversal or drive names.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", source.replace("/", "--")).strip(".") or "model"


def validate_model_source(source: str) -> str:
    source = source.strip()
    if not source:
        raise ValueError("Choose a model ID or a local Transformers model folder.")
    path = Path(source).expanduser()
    if source.lower().endswith((".gguf", ".ggml")) or source.startswith(("http://", "https://", "ollama:")):
        raise ValueError(
            "GGUF files and Ollama/LM Studio API endpoints are not supported by this "
            "engine yet. Vitalis needs direct access to transformer activations. "
            "Choose the model's Transformers checkpoint folder (config.json, tokenizer, "
            "and .safetensors or pytorch .bin weights)."
        )
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise ValueError(f"{path} has no config.json. Choose a Transformers checkpoint folder.")
        if not has_weights(path):
            raise ValueError(f"{path} has no Transformers weights. GGUF-only folders are not supported.")
        return canonical_model_source(source)
    if path.exists() or path.is_absolute() or source.startswith((".", "~")) or "\\" in source or ":" in source:
        raise ValueError(f"Not a local Transformers model folder: {source}. Use a folder or a Hub ID such as Qwen/Qwen2.5-3B-Instruct.")
    return source


def has_weights(path: Path) -> bool:
    """Require all shards when an index is present; ignore half-downloaded caches."""
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / index_name
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                if shards and all((path / name).is_file() for name in shards):
                    return True
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def discover_cached_models(cache_dir=None) -> list[dict]:
    """Inspect the configured HF cache only; never download or scan home folders."""
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_dir = HF_HUB_CACHE
    found = []
    for repo in sorted(Path(cache_dir).glob("models--*")):
        repo_id = repo.name[len("models--"):].replace("--", "/")
        for config_path in sorted(repo.glob("snapshots/*/config.json")):
            snapshot = config_path.parent
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                # Exclude obvious encoders, vision-only models, and incomplete downloads.
                architectures = config.get("architectures", [])
                if not any("CausalLM" in arch or arch == "GPT2LMHeadModel" for arch in architectures):
                    continue
                if not has_weights(snapshot):
                    continue
            except (OSError, ValueError, TypeError):
                continue
            found.append({"label": f"{repo_id} ({snapshot.name[:8]})",
                          "source": canonical_model_source(str(snapshot)),
                          "model_type": config.get("model_type", "unknown")})
    return found
