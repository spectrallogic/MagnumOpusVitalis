"""
Model Profile System
=====================
Create, save, and load per-model profiles containing:
  - Direction vectors (emotion + temporal)
  - Metadata (model name, layer, dimensions, creation date)
  - Baseline (model's natural emotional resting state)

Usage:
    python -m magnum_opus_v2.profile create gpt2
    python -m magnum_opus_v2.profile list
    python -m magnum_opus_v2.profile info gpt2

Programmatic:
    from magnum_opus_v2 import create_profile, load_profile, V2Engine, load_model

    model, tokenizer, device = load_model("gpt2")
    profile = create_profile("gpt2")
    engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from magnum_opus_v2.extraction import extract_hidden_states, extract_vectors
from magnum_opus_v2.loader import load_model
from magnum_opus_v2.model_sources import canonical_model_source, model_storage_key

PROFILES_DIR = Path(__file__).parent.parent / "profiles"

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


@dataclass
class ProfileMetadata:
    model_name: str
    target_layer: int
    hidden_dim: int
    n_layers: int
    vector_names: List[str]
    created_at: str
    version: str = "2.0"
    activation_site: str = "block_output"
    model_fingerprint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileMetadata":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields.setdefault("version", "1.0")
        fields.setdefault("activation_site", "legacy_hidden_states_index")
        return cls(**fields)


@dataclass
class Baseline:
    projections: Dict[str, float]
    neutral_prompts_used: int

    @classmethod
    def from_dict(cls, data: dict) -> "Baseline":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelProfile:
    """A saved profile for a specific model."""

    def __init__(self, metadata: ProfileMetadata, vectors: Dict[str, torch.Tensor],
                 baseline: Baseline, dynamics: Optional[dict] = None):
        self.metadata = metadata
        self.vectors = vectors
        self.baseline = baseline
        # The Mirror (M1): emotion dynamics fitted from the model's own
        # implied human trajectories. None = fall back to hand-authored
        # constants in _dynamics.py.
        self.dynamics = dynamics

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

    def validate_activation_site(self) -> None:
        if self.metadata.activation_site != "block_output":
            raise ValueError(
                "Legacy profile uses a different activation boundary. Re-extract "
                f"with: python -m magnum_opus_v2.profile create {self.model_name}. "
                "Start fresh runtime state; old latent memories use the old boundary."
            )

    def signature(self) -> str:
        """Bind runtime checkpoints to this calibration, including actual vectors."""
        digest = hashlib.sha256(json.dumps(
            {"metadata": asdict(self.metadata), "baseline": asdict(self.baseline),
             "dynamics": self.dynamics}, sort_keys=True,
        ).encode("utf-8"))
        for name, vector in sorted(self.vectors.items()):
            digest.update(name.encode("utf-8"))
            digest.update(vector.detach().cpu().float().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _sanitize_model_name(model_name: str) -> str:
    return model_storage_key(canonical_model_source(model_name))


def _profile_dir(model_name: str, profiles_dir: Path = PROFILES_DIR) -> Path:
    return profiles_dir / _sanitize_model_name(model_name)


def _discover_baseline(
    model, tokenizer, vectors: Dict[str, torch.Tensor],
    target_layer: int, device: str, verbose: bool = True,
) -> Baseline:
    if verbose:
        print("\n  Discovering baseline emotional state...")

    all_projections: Dict[str, List[float]] = {name: [] for name in vectors}

    for prompt in NEUTRAL_PROMPTS:
        hidden = extract_hidden_states(model, tokenizer, [prompt], target_layer, device)
        for name, vec in vectors.items():
            proj = torch.dot(hidden.to(vec.device), vec.float()).item()
            all_projections[name].append(proj)

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
    *,
    loaded_model=None,
    model_fingerprint: Optional[str] = None,
) -> ModelProfile:
    """Calibrate once; pass (model, tokenizer, device) to avoid a second model load."""
    model_name = canonical_model_source(model_name)
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  Creating profile for: {model_name}")
        print(f"{'=' * 50}")

    model, tokenizer, device = (loaded_model if loaded_model is not None
                                else load_model(model_name, device))

    from magnum_opus_v2.steering_hook import SteeringHook
    n_layers = len(SteeringHook._layer_list(model))
    hidden_dim = model.get_input_embeddings().weight.shape[1]

    target_layer = n_layers // 2

    vectors = extract_vectors(model, tokenizer, target_layer=target_layer, device=device,
                              verbose=verbose)

    baseline = _discover_baseline(model, tokenizer, vectors, target_layer, device, verbose)

    # The Mirror: fit emotion dynamics (onset/decay/baseline + interactions)
    # from the model's own implied emotional trajectories. This is where the
    # engine's temperament comes FROM THE LLM. There is no authored fallback:
    # if fitting fails, affect is flat/neutral (EmotionConfig()), never a
    # hand-written personality.
    from magnum_opus_v2.mirror import extract_dynamics
    try:
        dynamics = extract_dynamics(
            model, tokenizer, vectors, baseline.projections,
            target_layer, device, verbose=verbose,
        )
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  Mirror extraction failed ({e}) — affect will be FLAT/"
                  "neutral (no fitted dynamics, and no authored fallback).")
        dynamics = None

    metadata = ProfileMetadata(
        model_name=model_name,
        target_layer=target_layer,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        vector_names=list(vectors.keys()),
        created_at=datetime.now().isoformat(),
        model_fingerprint=model_fingerprint,
    )

    profile = ModelProfile(metadata, vectors, baseline, dynamics=dynamics)

    save_path = save_profile(profile, profiles_dir)
    if verbose:
        print(f"\n  Profile saved to: {save_path}")
        print(f"  {len(vectors)} vectors, {hidden_dim}d, layer {target_layer}/{n_layers}")

    return profile


def save_profile(profile: ModelProfile, profiles_dir: Path = PROFILES_DIR) -> Path:
    """Save a profile to disk."""
    profile_dir = _profile_dir(profile.metadata.model_name, profiles_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    torch.save(profile.vectors, profile_dir / "vectors.pt")

    with open(profile_dir / "metadata.json", "w") as f:
        json.dump(asdict(profile.metadata), f, indent=2)

    with open(profile_dir / "baseline.json", "w") as f:
        json.dump(asdict(profile.baseline), f, indent=2)

    if profile.dynamics:
        with open(profile_dir / "dynamics.json", "w") as f:
            json.dump(profile.dynamics, f, indent=2)
    else:
        # Saving a full replacement must not retain another extraction's fit.
        (profile_dir / "dynamics.json").unlink(missing_ok=True)

    return profile_dir


def load_profile(
    path_or_model_name: Union[str, Path],
    profiles_dir: Path = PROFILES_DIR,
) -> ModelProfile:
    """Load a saved profile by model name or directory path."""
    path = Path(path_or_model_name)

    if not (path / "metadata.json").is_file():
        path = _profile_dir(canonical_model_source(str(path_or_model_name)), profiles_dir)

    if not path.is_dir():
        raise FileNotFoundError(
            f"No profile found at {path}. "
            f"Run: python -m magnum_opus_v2.profile create {path_or_model_name}"
        )

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

    dynamics = None
    dynamics_path = path / "dynamics.json"
    if dynamics_path.exists():
        with open(dynamics_path) as f:
            dynamics = json.load(f)

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

    return ModelProfile(metadata, vectors, baseline, dynamics=dynamics)


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


def _cli():
    parser = argparse.ArgumentParser(
        prog="python -m magnum_opus_v2.profile",
        description="Manage model profiles for Magnum Opus Vitalis",
    )
    sub = parser.add_subparsers(dest="command")

    p_create = sub.add_parser("create", help="Extract vectors and create a profile")
    p_create.add_argument("model_name", help="HuggingFace model ID (e.g. gpt2, gpt2-medium)")
    p_create.add_argument("--device", default=None, help="cpu, cuda, or mps")
    p_create.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    p_dyn = sub.add_parser(
        "dynamics",
        help="Fit Mirror dynamics (M1) onto an EXISTING profile",
    )
    p_dyn.add_argument("model_name", help="Model with a saved profile")
    p_dyn.add_argument("--device", default=None)
    p_dyn.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    p_list = sub.add_parser("list", help="List all saved profiles")
    p_list.add_argument("--profiles-dir", default=str(PROFILES_DIR))

    p_info = sub.add_parser("info", help="Show profile details")
    p_info.add_argument("model_name", help="Model name or profile path")
    p_info.add_argument("--profiles-dir", default=str(PROFILES_DIR))

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

    elif args.command == "dynamics":
        from magnum_opus_v2.mirror import extract_dynamics
        profile = load_profile(args.model_name, profiles_dir)
        model, tokenizer, device = load_model(args.model_name, args.device)
        dynamics = extract_dynamics(
            model, tokenizer, profile.vectors, profile.baseline.projections,
            profile.target_layer, device, verbose=True,
        )
        if dynamics is None:
            print("  Mirror extraction produced nothing — profile unchanged.")
            return
        profile.dynamics = dynamics
        save_profile(profile, profiles_dir)
        print(f"\n  Mirror dynamics saved into profile for '{args.model_name}'.")

    elif args.command == "list":
        profiles = list_profiles(profiles_dir)
        if not profiles:
            print("  No profiles found.")
            print(f"  Create one: python -m magnum_opus_v2.profile create gpt2")
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
        if profile.dynamics:
            n_inter = len(profile.dynamics.get("interactions", []))
            print(f"  Mirror:  fitted dynamics present "
                  f"({n_inter} interaction couplings)")
        else:
            print("  Mirror:  none — run: python -m magnum_opus_v2.profile "
                  f"dynamics {m.model_name}")
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
