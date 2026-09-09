"""One startup path for the local dashboard and its command-line diagnostic."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch

from magnum_opus_v2.extraction import read_block_output
from magnum_opus_v2.model_sources import (
    canonical_model_source, discover_cached_models, model_storage_key, validate_model_source,
)
from magnum_opus_v2.profile import PROFILES_DIR, create_profile, load_profile, save_profile
from magnum_opus_v2.steering_hook import SteeringHook


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Run Vitalis with a local Transformers model. No LLM training required.",
        epilog="With no --model, choose a cached checkpoint or enter a model folder. "
               "GGUF/Ollama APIs require an engine adapter and are not supported yet.",
    )
    parser.add_argument("--model", help="Local Transformers folder or Hugging Face model ID")
    parser.add_argument("--list-models", action="store_true", help="List cached causal LMs without loading weights")
    parser.add_argument("--download", action="store_true", help="Allow downloading an explicitly selected Hub model; default is offline")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--check", action="store_true", help="Calibrate, generate a short reply, verify live state, and exit")
    parser.add_argument("--rebuild-profile", action="store_true", help="Recalibrate; archive the previous profile")
    parser.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR)
    parser.add_argument("--profile-path", type=Path, help="Use an explicit compatible profile directory")
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)  # old scripts still work
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow executing custom code supplied by the selected model")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="Print the dashboard URL without opening a browser")
    parser.add_argument("--resume", action="store_true", help="Resume saved runtime state and save it on exit")
    return parser


def choose_model(source=None, *, interactive=None, input_fn=None):
    if source:
        return validate_model_source(source)
    choices = discover_cached_models()
    print("\n  Cached Transformers models:")
    for i, candidate in enumerate(choices, 1):
        print(f"    {i}. {candidate['label']}")
    if not choices:
        print("    None found. You can enter a local model folder.")
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError("Specify --model MODEL_OR_FOLDER in a non-interactive shell. Use --list-models to see cached checkpoints.")
    read = input_fn or input
    while True:
        answer = read("\n  Model number, folder, or Hub ID (q to quit): ").strip()
        if answer.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if answer.isdecimal():
            index = int(answer) - 1
            if 0 <= index < len(choices):
                return choices[index]["source"]
            print("  Choose a number from the list.")
            continue
        try:
            return validate_model_source(answer.strip('\"'))
        except ValueError as exc:
            print(f"  {exc}")


def model_fingerprint(source, model):
    """Revision/config identity plus local file stats; no costly weight rehash."""
    payload = {"config": model.config.to_dict(), "source": source,
               "revision": getattr(model.config, "_commit_hash", None)}
    folder = Path(source)
    if folder.is_dir():
        payload["files"] = [
            (str(p.name), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model", ".txt")
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_model_boundary(model, tokenizer, device):
    """Check the actual read/write boundary before expensive profile extraction."""
    layers = SteeringHook._layer_list(model)
    layer = len(layers) // 2
    inputs = tokenizer("Hello.", return_tensors="pt").to(device)
    hidden = read_block_output(model, inputs, layer)
    dim = model.get_input_embeddings().weight.shape[1]
    if hidden.shape != (dim,) or not torch.isfinite(hidden).all():
        raise ValueError("The model's block output is incompatible with Vitalis; a model adapter is required.")
    # A real nonzero intervention must reach that exact output and keep logits finite.
    hook = SteeringHook().attach(model, layer)
    # Stay above rounding at large residual coordinates in float16/bfloat16.
    epsilon = torch.finfo(model.get_input_embeddings().weight.dtype).eps
    magnitude = max(0.1, 8 * epsilon * float(hidden.abs().max()))
    delta = torch.ones_like(hidden) * magnitude
    try:
        with hook.isolated(delta), torch.no_grad():
            changed = read_block_output(model, inputs, layer)
            output = model(**inputs)
        if not torch.allclose(changed - hidden, delta, atol=magnitude / 4, rtol=0):
            raise ValueError("The model did not accept the activation intervention at the measured block.")
        if not torch.isfinite(output.logits).all():
            raise ValueError("Activation steering produced nonfinite logits on this model.")
    finally:
        hook.detach()
    print("  Transformer activation check passed.")


def validate_profile(profile, source, model):
    profile.validate_activation_site()
    if canonical_model_source(profile.model_name) != source:
        raise ValueError("The selected profile belongs to a different model. Omit --profile-path to calibrate automatically.")
    layers = SteeringHook._layer_list(model)
    if (profile.hidden_dim != model.get_input_embeddings().weight.shape[1]
            or profile.metadata.n_layers != len(layers)
            or not 0 <= profile.target_layer < len(layers)):
        raise ValueError("Profile dimensions/layer do not match the loaded model.")
    if not profile.vectors or any(v.shape != (profile.hidden_dim,) or not torch.isfinite(v).all()
                                  or v.norm() < 1e-8 for v in profile.vectors.values()):
        raise ValueError("The profile contains invalid activation vectors.")


def prepare_profile(source, loaded_model, *, profiles_dir=PROFILES_DIR,
                    profile_path=None, rebuild=False):
    """Reuse compatible calibration, or extract using the already loaded model.

    Extraction completes in a temporary folder before any existing profile is
    touched. Previous calibration files are copied to .archive before replacement.
    """
    model, tokenizer, device = loaded_model
    fingerprint = model_fingerprint(source, model)
    if profile_path is not None:
        if rebuild:
            raise ValueError("Use --rebuild-profile without --profile-path.")
        if not (Path(profile_path) / "metadata.json").is_file():
            raise ValueError("--profile-path must point to a saved profile containing metadata.json.")
        profile = load_profile(profile_path)
        validate_profile(profile, source, model)
        if profile.metadata.model_fingerprint and profile.metadata.model_fingerprint != fingerprint:
            raise ValueError("This profile was calibrated against different model files. Omit --profile-path to recalibrate.")
        return profile, False

    root = Path(profiles_dir).resolve()
    destination = root / model_storage_key(source)
    if not rebuild and destination.is_dir():
        try:
            profile = load_profile(destination)
            validate_profile(profile, source, model)
            if profile.metadata.model_fingerprint != fingerprint:
                raise ValueError("Model calibration identity needs to be refreshed.")
            print(f"  Reusing profile: {destination}")
            return profile, False
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, EOFError) as exc:
            print(f"  Preparing a fresh profile: {exc}")

    root.mkdir(parents=True, exist_ok=True)
    # Disallow symlinked profile destinations that could overwrite another directory.
    destination.resolve().relative_to(root)
    print("  First-run calibration: extracting directions and fitting controller dynamics.")
    print("  The LLM weights stay unchanged. Later starts reuse this profile.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".calibrating-", dir=root) as temp:
        profile = create_profile(source, profiles_dir=Path(temp), loaded_model=loaded_model,
                                 model_fingerprint=fingerprint, verbose=False)
        validate_profile(profile, source, model)
        if destination.exists():
            archive = root / ".archive" / (destination.name + "-" + uuid.uuid4().hex[:12])
            archive.resolve().relative_to(root)
            shutil.copytree(destination, archive)
            print(f"  Previous profile archived at: {archive}")
        save_profile(profile, root)
    print(f"  Profile ready: {destination}")
    return profile, True


def check_engine(engine):
    """A bounded CLI probe of real generation and a concurrently running flow clock."""
    engine.start()
    try:
        before = engine.snapshot()["flow_metrics"]["flow"]["ticks"]
        reply = engine.converse("Hello. Say one short sentence.", max_new_tokens=16)
        deadline = time.monotonic() + 2
        snapshot = engine.snapshot()
        while snapshot["flow_metrics"]["flow"]["ticks"] <= before and time.monotonic() < deadline:
            time.sleep(0.05)
            snapshot = engine.snapshot()
        if not reply.strip():
            raise RuntimeError("The model generated no text during the startup check.")
        if snapshot["flow_metrics"]["flow"]["ticks"] <= before:
            raise RuntimeError("The background flow clock did not advance.")
        if not torch.isfinite(engine.bus.state).all():
            raise RuntimeError("The runtime state contains nonfinite values.")
        print(f"\n  Model reply: {reply!r}")
        print("  CHECK PASSED: generation, activation access, and background state work.")
        print("  This checks operation; it does not measure response quality.")
    finally:
        engine.stop()
