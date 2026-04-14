"""
Model Profile System
=====================
Create, save, and load per-model profiles containing:
  - Direction vectors (emotion + temporal)
  - Metadata (model name, layer, dimensions, creation date)
  - Baseline (model's natural emotional resting state)

Usage:
    python -m magnum_opus.profile create gpt2
    python -m magnum_opus.profile list
    python -m magnum_opus.profile info gpt2

Programmatic:
    from magnum_opus import create_profile, load_profile

    profile = create_profile("gpt2")
    engine = MagnumOpusEngine(model, tokenizer, profile=profile, device=device)

    # Next time: instant load
    profile = load_profile("gpt2")
"""

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from magnum_opus.extraction import extract_hidden_states, extract_vectors
from magnum_opus.loader import load_model

# Default profiles directory (project root / profiles/)
PROFILES_DIR = Path(__file__).parent.parent / "profiles"

# Emotionally neutral prompts for baseline discovery
NEUTRAL_PROMPTS = [
    "The table is in the room.",
    "Today is a weekday.",
    "The document has three sections.",
    "Water boils at one hundred degrees Celsius.",
    "The meeting is scheduled for next Tuesday.",
    "There are twelve months in a year.",
    "The file contains several lines of text.",
    "The road connects two cities.",
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProfileMetadata:
    model_name: str
    target_layer: int
    hidden_dim: int
    n_layers: int
    vector_names: List[str]
    created_at: str
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Baseline:
    projections: Dict[str, float]
    neutral_prompts_used: int

    @classmethod
    def from_dict(cls, data: dict) -> "Baseline":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelProfile:
    """
    A saved profile for a specific model. Contains everything needed
    to create an engine without re-extracting vectors.
    """

    def __init__(self, metadata: ProfileMetadata, vectors: Dict[str, torch.Tensor],
                 baseline: Baseline):
        self.metadata = metadata
        self.vectors = vectors
        self.baseline = baseline

    @property
    def target_layer(self) -> int:
        return self.metadata.target_layer

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def hidden_dim(self) -> int:
        return self.metadata.hidden_dim

    def __repr__(self) -> str:
        return (f"ModelProfile(model={self.metadata.model_name!r}, "
                f"layer={self.metadata.target_layer}, "
                f"dim={self.metadata.hidden_dim}, "
                f"vectors={len(self.vectors)})")


# ═══════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_model_name(model_name: str) -> str:
    """Convert model name to a safe directory name."""
    return model_name.replace("/", "--").replace("\\", "--")


def _profile_dir(model_name: str, profiles_dir: Path = PROFILES_DIR) -> Path:
    return profiles_dir / _sanitize_model_name(model_name)


def _discover_baseline(
    model, tokenizer, vectors: Dict[str, torch.Tensor],
    target_layer: int, device: str, verbose: bool = True,
) -> Baseline:
    """
    Discover the model's natural emotional resting state by running
    neutral text and measuring projections onto each direction vector.
    """
    if verbose:
        print("\n  Discovering baseline emotional state...")

    all_projections: Dict[str, List[float]] = {name: [] for name in vectors}

    for prompt in NEUTRAL_PROMPTS:
        hidden = extract_hidden_states(model, tokenizer, [prompt], target_layer, device)
        for name, vec in vectors.items():
            proj = torch.dot(hidden.to(vec.device), vec.float()).item()
            all_projections[name].append(proj)

    # Average across all neutral prompts
    avg_projections = {
        name: sum(vals) / len(vals) for name, vals in all_projections.items()
    }

    if verbose:
        print("  Baseline projections (neutral text):")
        for name, val in sorted(avg_projections.items()):
            bar = "+" * int(abs(val) * 20) if abs(val) > 0.01 else "~"
            sign = "+" if val > 0 else "-" if val < 0 else " "
            print(f"    {name:>16}: {sign}{abs(val):.4f}  {bar}")

    return Baseline(projections=avg_projections, neutral_prompts_used=len(NEUTRAL_PROMPTS))


def create_profile(
    model_name: str,
    profiles_dir: Path = PROFILES_DIR,
    device: Optional[str] = None,
    verbose: bool = True,
) -> ModelProfile:
    """
    Extract direction vectors and baseline for a model, save as a reusable profile.

    Args:
        model_name: Any HuggingFace model ID (gpt2, gpt2-medium, mistralai/Mistral-7B, etc.)
        profiles_dir: Where to save profiles (default: project_root/profiles/)
        device: "cpu", "cuda", "mps", or None for auto-detect
        verbose: Print progress

    Returns:
        ModelProfile ready to pass to MagnumOpusEngine
    """
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  Creating profile for: {model_name}")
        print(f"{'=' * 50}")

    # Load model
    model, tokenizer, device = load_model(model_name, device)

    # Detect architecture
    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError(f"Cannot detect architecture for {model_name}")

    target_layer = n_layers // 2

    # Extract vectors (reuses existing extraction pipeline)
    vectors = extract_vectors(model, tokenizer, target_layer=target_layer, device=device,
                              verbose=verbose)

    # Discover baseline
    baseline = _discover_baseline(model, tokenizer, vectors, target_layer, device, verbose)

    # Build metadata
    metadata = ProfileMetadata(
        model_name=model_name,
        target_layer=target_layer,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        vector_names=list(vectors.keys()),
        created_at=datetime.now().isoformat(),
    )

    profile = ModelProfile(metadata, vectors, baseline)

    # Save
    save_path = save_profile(profile, profiles_dir)
    if verbose:
        print(f"\n  Profile saved to: {save_path}")
        print(f"  {len(vectors)} vectors, {hidden_dim}d, layer {target_layer}/{n_layers}")

    return profile


def save_profile(profile: ModelProfile, profiles_dir: Path = PROFILES_DIR) -> Path:
    """Save a profile to disk."""
    profile_dir = _profile_dir(profile.metadata.model_name, profiles_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Vectors
    torch.save(profile.vectors, profile_dir / "vectors.pt")

    # Metadata
    with open(profile_dir / "metadata.json", "w") as f:
        json.dump(asdict(profile.metadata), f, indent=2)

    # Baseline
    with open(profile_dir / "baseline.json", "w") as f:
        json.dump(asdict(profile.baseline), f, indent=2)

    return profile_dir


def load_profile(
    path_or_model_name: Union[str, Path],
    profiles_dir: Path = PROFILES_DIR,
) -> ModelProfile:
    """
    Load a saved profile.

    Args:
        path_or_model_name: Either a full path to a profile directory,
            or a model name that will be resolved against profiles_dir.
        profiles_dir: Base directory for profiles (used when path_or_model_name is a model name)

    Returns:
        ModelProfile ready to pass to MagnumOpusEngine
    """
    path = Path(path_or_model_name)

    # If it's not an existing directory, try resolving as a model name
    if not path.is_dir():
        path = _profile_dir(str(path_or_model_name), profiles_dir)

    if not path.is_dir():
        raise FileNotFoundError(
            f"No profile found at {path}. "
            f"Run: python -m magnum_opus.profile create {path_or_model_name}"
        )

    # Load all three files
    vectors_path = path / "vectors.pt"
    metadata_path = path / "metadata.json"
    baseline_path = path / "baseline.json"

    for f in [vectors_path, metadata_path, baseline_path]:
        if not f.exists():
            raise FileNotFoundError(f"Profile incomplete — missing {f.name} in {path}")

    vectors = torch.load(vectors_path, weights_only=True)

    with open(metadata_path) as f:
        metadata = ProfileMetadata.from_dict(json.load(f))

    with open(baseline_path) as f:
        baseline = Baseline.from_dict(json.load(f))

    # Validate
    if len(vectors) != len(metadata.vector_names):
        raise ValueError(
            f"Vector count mismatch: {len(vectors)} in vectors.pt vs "
            f"{len(metadata.vector_names)} in metadata"
        )

    for name, vec in vectors.items():
        if vec.shape[0] != metadata.hidden_dim:
            raise ValueError(
                f"Dimension mismatch for '{name}': {vec.shape[0]} vs "
                f"metadata says {metadata.hidden_dim}"
            )

    return ModelProfile(metadata, vectors, baseline)


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> List[ProfileMetadata]:
    """List all saved profiles."""
    profiles = []
    if not profiles_dir.exists():
        return profiles

    for entry in sorted(profiles_dir.iterdir()):
        meta_file = entry / "metadata.json"
        if entry.is_dir() and meta_file.exists():
            with open(meta_file) as f:
                profiles.append(ProfileMetadata.from_dict(json.load(f)))

    return profiles


def profile_exists(model_name: str, profiles_dir: Path = PROFILES_DIR) -> bool:
    """Check if a profile exists for the given model."""
    path = _profile_dir(model_name, profiles_dir)
    return (path / "metadata.json").exists() and (path / "vectors.pt").exists()


def delete_profile(model_name: str, profiles_dir: Path = PROFILES_DIR) -> bool:
    """Delete a profile. Returns True if it existed."""
    path = _profile_dir(model_name, profiles_dir)
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        prog="python -m magnum_opus.profile",
        description="Manage model profiles for Magnum Opus Vitalis",
    )
    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Extract vectors and create a profile")
    p_create.add_argument("model_name", help="HuggingFace model ID (e.g. gpt2, gpt2-medium)")
    p_create.add_argument("--device", default=None, help="cpu, cuda, or mps")
    p_create.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    # list
    p_list = sub.add_parser("list", help="List all saved profiles")
    p_list.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    # info
    p_info = sub.add_parser("info", help="Show profile details")
    p_info.add_argument("model_name", help="Model name or profile path")
    p_info.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    # delete
    p_delete = sub.add_parser("delete", help="Delete a profile")
    p_delete.add_argument("model_name")
    p_delete.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_delete.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    profiles_dir = Path(args.profiles_dir)

    if args.command == "create":
        create_profile(args.model_name, profiles_dir=profiles_dir, device=args.device)

    elif args.command == "list":
        profiles = list_profiles(profiles_dir)
        if not profiles:
            print("  No profiles found.")
            print(f"  Create one: python -m magnum_opus.profile create gpt2")
            return
        print(f"\n  {'Model':<35} {'Layer':>6} {'Dim':>6} {'Vectors':>8}  {'Created'}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*8}  {'-'*19}")
        for p in profiles:
            created = p.created_at[:19] if p.created_at else "unknown"
            print(f"  {p.model_name:<35} {p.target_layer:>6} {p.hidden_dim:>6} "
                  f"{len(p.vector_names):>8}  {created}")

    elif args.command == "info":
        try:
            profile = load_profile(args.model_name, profiles_dir)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            return

        m = profile.metadata
        print(f"\n  Profile: {m.model_name}")
        print(f"  Version: {m.version}")
        print(f"  Created: {m.created_at}")
        print(f"  Layers:  {m.n_layers} (steering at layer {m.target_layer})")
        print(f"  Hidden:  {m.hidden_dim}d")
        print(f"  Vectors: {len(m.vector_names)}")
        for name in m.vector_names:
            baseline_val = profile.baseline.projections.get(name, 0.0)
            print(f"    {name:>16}: baseline={baseline_val:+.4f}")

    elif args.command == "delete":
        if not profile_exists(args.model_name, profiles_dir):
            print(f"  No profile found for '{args.model_name}'")
            return
        if not args.yes:
            answer = input(f"  Delete profile for '{args.model_name}'? [y/N] ")
            if answer.lower() != "y":
                print("  Cancelled.")
                return
        delete_profile(args.model_name, profiles_dir)
        print(f"  Deleted profile for '{args.model_name}'")


if __name__ == "__main__":
    _cli()
