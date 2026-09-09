"""Paired intervention evaluation using externally specified continuations.

Run with --help. This measures behavior under static interventions; it does
not evaluate consciousness or establish the benefit of the full runtime.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
import transformers

from magnum_opus_v2.loader import load_model
from magnum_opus_v2.profile import load_profile
from magnum_opus_v2.steering_hook import SteeringHook


def continuation_score(model, tokenizer, hook, prompt, completion, device, vector=None):
    """Mean conditional log probability of fixed completion tokens.

    Prompt and completion are tokenized separately, identically in every arm.
    This defines the token boundary explicitly (including leading spaces).
    No generated response or emotion projection is used as its own judge.
    """
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(completion, add_special_tokens=False)
    if not prefix or not target:
        raise ValueError("Each case needs a nonempty prompt and completion")
    ids = torch.tensor([prefix + target], device=device)
    with hook.isolated(vector), torch.no_grad():
        logits = model(ids).logits[0, len(prefix)-1:-1].float()
    logps = logits.log_softmax(-1)
    return float(logps.gather(1, torch.tensor(target, device=device)[:, None]).mean())


def paired_summary(values, seed=0):
    """Descriptive paired effect with an example-level bootstrap interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])]}


def evaluate_directions(model, tokenizer, profile, cases, vector_name, strengths,
                        seeds, device="cpu"):
    profile.validate_activation_site()
    if not cases or not seeds or not strengths:
        raise ValueError("Cases, strengths, and random-control seeds must be nonempty")
    if any(not np.isfinite(s) or s <= 0 for s in strengths):
        raise ValueError("Strengths must be finite and positive")
    direction = profile.vectors[vector_name].detach().float().to(device)
    norm = direction.norm()
    if not torch.isfinite(direction).all() or norm <= 1e-8:
        raise ValueError("Direction must be finite and nonzero")
    direction = direction / norm
    controls = []
    for seed in seeds:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        rand = torch.randn(direction.numel(), generator=gen).to(device)
        controls.append(rand / rand.norm())
    hook = SteeringHook().attach(model, profile.target_layer, layer_span=0)
    training = model.training
    model.eval()
    rows = []
    try:
        for case in cases:
            def margin(v):
                args = (model, tokenizer, hook, case["prompt"])
                return (continuation_score(*args, case["target"], device, v) -
                        continuation_score(*args, case["contrast"], device, v))
            baseline = margin(None)
            for strength in strengths:
                plus = margin(direction * strength)
                minus = margin(-direction * strength)
                randoms = [margin(rand * strength) for rand in controls]
                rows.append({"case_id": case["id"], "strength": strength,
                             "baseline_margin": baseline, "positive_margin": plus,
                             "negative_margin": minus, "random_margins": randoms,
                             "positive_minus_baseline": plus - baseline,
                             "positive_minus_negative": plus - minus,
                             "positive_minus_random_mean": plus - float(np.mean(randoms))})
    finally:
        hook.detach()
        model.train(training)
    summaries = {}
    for strength in strengths:
        selected = [r for r in rows if r["strength"] == strength]
        summaries[str(strength)] = {
            key: paired_summary([r[key] for r in selected])
            for key in ("positive_minus_baseline", "positive_minus_negative",
                        "positive_minus_random_mean")}
    return {"metric": "target_minus_contrast_mean_conditional_logprob",
            "rows": rows, "summary": summaries,
            "interpretation": "Static behavioral intervention pilot; no life or sentience measure. "
                              "Intervals resample cases, not models. Random seeds define control "
                              "directions, not independent replications. Labels require review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--profile", help="Profile path or model name; defaults to --model")
    parser.add_argument("--cases", type=Path, required=True, help="JSONL: id, prompt, target, contrast")
    parser.add_argument("--vector", default="calm")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/intervention.json"))
    args = parser.parse_args()
    payload = args.cases.read_bytes()
    cases = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case ids must be unique")
    profile = load_profile(args.profile or args.model)
    if profile.model_name != args.model:
        raise ValueError("Profile belongs to a different model")
    profile.validate_activation_site()
    model, tokenizer, device = load_model(args.model, args.device)
    report = evaluate_directions(model, tokenizer, profile, cases, args.vector,
                                 args.strengths, args.seeds, device)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report["manifest"] = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "activation_site": profile.metadata.activation_site,
        "target_layer": profile.target_layer, "profile_created_at": profile.metadata.created_at,
        "vector": args.vector, "strengths": args.strengths, "control_seeds": args.seeds,
        "vector_sha256": hashlib.sha256(profile.vectors[args.vector].float().cpu().numpy().tobytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(payload).hexdigest(),
        "cases": cases, "device": device, "dtype": str(next(model.parameters()).dtype),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "git_commit": commit, "working_tree_dirty": dirty,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)).replace("\\", "/"):
                hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output}; results are a pilot, not a validated emotion classifier.")


if __name__ == "__main__":
    main()
