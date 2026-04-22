"""
Steering verification: bus -> hook -> model generation.

Loads gpt2 and its profile, attaches the v2 steering hook driven by a
LatentBus, then generates three completions of the same prompt under three
different bus states:

  1. Raw           — hook detached. Pure model output.
  2. Bus at zero   — hook attached, bus.state = 0. Should match Raw closely.
  3. Bus pushed    — bus.state set to a known emotion vector × magnitude.
                     Should differ from Raw in the direction of that emotion.

This proves the bus -> driver -> hook pipeline works end-to-end and that
generation actually shifts when the substrate moves. No regions are wired
up yet — this only tests the steering plumbing.

Usage:
    python demo_v2_steering.py
    python demo_v2_steering.py --model gpt2 --emotion joy --magnitude 8.0
"""

import argparse

import torch

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile

from magnum_opus_v2 import LatentBus, V2Config
from magnum_opus_v2.steering_hook import SteeringHook, BusSteeringDriver


PROMPT = "Today I went outside and"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30,
             seed: int = 0) -> str:
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--emotion", default="joy",
                    help="Which named emotion vector to push the bus toward")
    ap.add_argument("--magnitude", type=float, default=8.0,
                    help="How far to push bus.state in that direction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=30)
    args = ap.parse_args()

    print(f"\n  Loading model + profile: {args.model}")
    model, tokenizer, device = load_model(args.model)
    profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}, "
          f"vectors={list(profile.vectors.keys())}")

    cfg = V2Config(hidden_dim=profile.hidden_dim, device=device)
    bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)
    driver = BusSteeringDriver(bus, steering_strength=1.0, smoothing=0.0)

    hook = SteeringHook()

    # ---------- (1) Raw, hook detached ----------
    print(f"\n=== (1) RAW (hook detached) ===")
    print(f"  prompt: {PROMPT!r}")
    out_raw = generate(model, tokenizer, PROMPT, args.max_tokens, args.seed)
    print(f"  output: {out_raw}")

    # Now attach the hook for the rest of the runs.
    hook.attach(model, profile.target_layer)

    # ---------- (2) Hook attached, bus at zero ----------
    bus.state.zero_()
    bus.velocity.zero_()
    hook.set_steering(driver.read())
    print(f"\n=== (2) HOOK attached, bus.state = 0 (should match RAW closely) ===")
    out_zero = generate(model, tokenizer, PROMPT, args.max_tokens, args.seed)
    print(f"  bus.state.norm() = {bus.state.norm().item():.4f}")
    print(f"  output: {out_zero}")

    # ---------- (3) Bus pushed toward an emotion vector ----------
    if args.emotion not in profile.vectors:
        print(f"\n  emotion {args.emotion!r} not in profile. "
              f"available: {list(profile.vectors.keys())}")
        emo_vec = next(iter(profile.vectors.values()))
        emo_name = next(iter(profile.vectors.keys()))
    else:
        emo_vec = profile.vectors[args.emotion]
        emo_name = args.emotion

    target = emo_vec.to(device).float() * args.magnitude
    bus.state.copy_(target)
    bus.velocity.zero_()
    driver.reset_smoothing()
    hook.set_steering(driver.read())

    print(f"\n=== (3) BUS pushed toward {emo_name!r} × {args.magnitude} ===")
    print(f"  bus.state.norm() = {bus.state.norm().item():.4f}")
    print(f"  steering vec norm = {driver.read().norm().item():.4f}")
    out_pushed = generate(model, tokenizer, PROMPT, args.max_tokens, args.seed)
    print(f"  output: {out_pushed}")

    # ---------- Verdicts ----------
    print(f"\n=== Verdicts ===")
    same_zero = out_raw == out_zero
    diff_pushed = out_pushed != out_raw
    print(f"  zero-bus matches raw?     "
          f"{'OK' if same_zero else 'DIFFERENT (small numerical drift OK)'}")
    print(f"  pushed-bus differs raw?   "
          f"{'OK — steering is wired' if diff_pushed else 'FAIL — hook not active'}")


if __name__ == "__main__":
    main()
