"""
Limbic + Temporal verification.

Loads gpt2 + profile, builds a LatentBus, registers Limbic and Temporal as
flow/perception regions, runs the flow loop. Then:

  1. Idle 4s — bus should mostly stay near baseline (homeostatic decay).
  2. Stimulate "joy" 1.0 + "trust" 0.6 — Limbic should perturb the bus
     in the direction of those emotion vectors. Bus norm should rise.
  3. Generate text mid-flow with the live bus.state driving the steering
     hook. The output should reflect the current emotional tilt.
  4. Stop stimulation, idle 6s — emotions should decay, bus should drift
     back toward baseline. Subjective time should accumulate.

Proves the chain: external stimulus -> Limbic.stimulate -> bus perturbation
-> hook reads live bus -> model generation reflects state.
"""

import argparse
import time

import torch

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile

from magnum_opus_v2 import (
    LatentBus, FlowRunner, V2Config, Limbic, Temporal,
    SteeringHook, BusSteeringDriver,
)


PROMPT = "Today I went outside and"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30,
             seed: int = 42) -> str:
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


def log_state(label: str, bus: LatentBus, limbic: Limbic, temporal: Temporal):
    snap_b = bus.snapshot()
    snap_l = limbic.snapshot()
    snap_t = temporal.snapshot(bus)
    blended = snap_l["blended"]
    top_emo = sorted(blended.items(), key=lambda kv: -abs(kv[1]))[:3]
    top_str = "  ".join(f"{n}={v:+.3f}" for n, v in top_emo)
    print(
        f"  [{label:12s}]  bus_norm={snap_b['state_norm']:.3f}  "
        f"vel={snap_b['velocity_norm']:.3f}  "
        f"div={snap_b['divergence']:.3f}  "
        f"subj_t={snap_t['subjective_elapsed']:.2f}  "
        f"top: {top_str}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--idle1", type=float, default=4.0)
    ap.add_argument("--stim", type=float, default=4.0,
                    help="Seconds between stimulation and decay phase")
    ap.add_argument("--idle2", type=float, default=6.0)
    args = ap.parse_args()

    print(f"\n  Loading model + profile: {args.model}")
    model, tokenizer, device = load_model(args.model)
    profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}, "
          f"vectors={list(profile.vectors.keys())}")

    cfg = V2Config(hidden_dim=profile.hidden_dim, device=device)
    bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)

    limbic = Limbic(
        emotion_vectors=profile.vectors,
        device=device,
        steering_strength=1.0,
    )
    temporal = Temporal(
        temporal_vectors=profile.vectors,
        device=device,
        steering_strength=0.3,
    )

    # Set the bus's homeostatic baseline to the personality vector so the
    # substrate settles at the model's natural emotional tilt (calm + trust
    # + curious) rather than at zero. Without this, the attractor pulls
    # toward zero and Limbic forever pushes outward — bus norm climbs.
    bus.set_baseline(limbic.baseline_vector(), weight=1.0)

    runner = FlowRunner(
        bus=bus,
        regions=[limbic, temporal],
        clock_config=cfg.clock,
        verbose_errors=True,
    )

    # Hook + driver for live generation
    hook = SteeringHook()
    hook.attach(model, profile.target_layer)
    driver = BusSteeringDriver(bus, steering_strength=1.0, smoothing=0.0)

    runner.start()
    print(f"  flow runner started, flow_dt={cfg.clock.flow_dt_seconds*1000:.0f}ms")

    # ------------------- Phase 1: idle -------------------
    print(f"\n=== Phase 1: idle {args.idle1:.1f}s ===")
    end = time.monotonic() + args.idle1
    while time.monotonic() < end:
        time.sleep(0.5)
        log_state("idle", bus, limbic, temporal)

    # ------------------- Phase 2: stimulate -------------------
    print(f"\n=== Phase 2: stimulate joy=1.0 trust=0.6 then hold {args.stim:.1f}s ===")
    limbic.stimulate_many({"joy": 1.0, "trust": 0.6})
    end = time.monotonic() + args.stim
    while time.monotonic() < end:
        time.sleep(0.5)
        log_state("stim", bus, limbic, temporal)

    # ------------------- Phase 3: generate mid-flow -------------------
    print(f"\n=== Phase 3: generate with live bus.state ===")
    hook.set_steering(driver.read())
    print(f"  steering vec norm at gen time = {driver.read().norm().item():.3f}")
    out_steered = generate(model, tokenizer, PROMPT)
    print(f"  steered output: {out_steered}")

    # Compare to a raw generation (steering disabled)
    hook.set_steering(None)
    out_raw = generate(model, tokenizer, PROMPT)
    print(f"  raw     output: {out_raw}")

    # ------------------- Phase 4: idle decay -------------------
    print(f"\n=== Phase 4: stop stimulation, idle {args.idle2:.1f}s, watch decay ===")
    end = time.monotonic() + args.idle2
    while time.monotonic() < end:
        time.sleep(1.0)
        log_state("decay", bus, limbic, temporal)

    runner.stop()
    print("\n  flow runner stopped")

    # Verdicts
    final_blended = limbic.snapshot()["blended"]
    final_joy = final_blended.get("joy", 0.0)
    final_norm = bus.snapshot()["state_norm"]
    print("\n=== Verdicts ===")
    print(f"  final joy after decay: {final_joy:+.4f}  "
          f"({'OK' if final_joy < 0.5 else 'still high'})")
    print(f"  final bus norm:        {final_norm:.4f}")
    print(f"  steered vs raw differ: "
          f"{'OK' if out_steered != out_raw else 'FAIL — output unchanged'}")


if __name__ == "__main__":
    main()
