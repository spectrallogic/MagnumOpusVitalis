"""
SubconsciousStack verification — point 4 of the user's vision.

Loads gpt2 + profile, builds the bus + Limbic + Temporal + SubconsciousStack
(with NoiseSampler and TokenEmbeddingSampler — Memory comes later).

Runs three phases:

  1. Idle 6s — subconscious produces intrusive thoughts even with no
     stimulation. Logs each tick whose intrusive thought changed:
     source, surprise flag, magnitude, decoded token if from tokens.

  2. Stimulate joy=1.0 + curious=0.6 — emotion bias should tilt L0
     sampling and L2 rescoring toward joy/curious-aligned candidates.
     Intrusive thoughts should noticeably shift in flavor.

  3. Generate text mid-flow with the live bus driving steering. Compare
     against raw gpt2 generation.

Verdicts:
  - intrusive thoughts vary across ticks (not always the same)
  - at least one "surprise" promotion occurs over the run
  - both samplers contribute (sources include both noise and tokens)
  - emotion stimulation visibly shifts the distribution of sources/tokens
"""

import argparse
import time
from collections import Counter

import torch

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile

from magnum_opus_v2 import (
    LatentBus, FlowRunner, V2Config, Limbic, Temporal,
    SubconsciousStack, NoiseSampler, TokenEmbeddingSampler,
    SteeringHook, BusSteeringDriver,
)


PROMPT = "Today I went outside and"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30,
             seed: int = 42) -> str:
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=0.92, temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def sample_phase(
    name: str, seconds: float,
    limbic: Limbic, subc: SubconsciousStack,
    tokenizer,
) -> dict:
    """Sample subconscious snapshots every 100ms. Push limbic blend each tick."""
    last_intrusive_norm = -1.0
    intrusive_log = []  # list of (t, source, conf, surprise, token_str)
    end = time.monotonic() + seconds
    t0 = time.monotonic()
    while time.monotonic() < end:
        # Push the live emotion blend to the subconscious so L0/L2 use it.
        subc.set_emotion_blend(limbic.snapshot()["blended"])
        snap = subc.snapshot()
        # Only log when the intrusive vector changed (tracked by norm — coarse but free)
        if abs(snap["intrusive_norm"] - last_intrusive_norm) > 1e-6:
            last_intrusive_norm = snap["intrusive_norm"]
            tok = None
            meta = snap.get("intrusive_meta") or {}
            if "token_id" in meta:
                tok = tokenizer.decode([meta["token_id"]]).strip()
            intrusive_log.append({
                "t": time.monotonic() - t0,
                "source": snap.get("intrusive_source"),
                "conf": snap.get("intrusive_confidence"),
                "surprise": snap.get("surprise"),
                "token": tok,
            })
        time.sleep(0.1)

    # Print summary table
    print(f"\n  --- {name} subconscious log ({len(intrusive_log)} unique intrusives) ---")
    sources = Counter(e["source"] for e in intrusive_log)
    surprises = sum(1 for e in intrusive_log if e["surprise"])
    print(f"  source distribution: {dict(sources)}")
    print(f"  surprise promotions: {surprises}")

    # Show first 12
    for e in intrusive_log[:12]:
        flag = " (!)" if e["surprise"] else ""
        tok = f"  token={e['token']!r}" if e["token"] else ""
        src = e["source"] or "(none)"
        conf = e["conf"] if e["conf"] is not None else 0.0
        print(f"   t={e['t']:5.2f}s  source={src:18s}  "
              f"conf={conf:.2f}{flag}{tok}")
    if len(intrusive_log) > 12:
        print(f"   ... +{len(intrusive_log)-12} more")

    return {
        "n_unique": len(intrusive_log),
        "sources": dict(sources),
        "surprises": surprises,
        "tokens_seen": [e["token"] for e in intrusive_log if e["token"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--idle-seconds", type=float, default=6.0)
    ap.add_argument("--stim-seconds", type=float, default=6.0)
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}")

    cfg = V2Config(hidden_dim=profile.hidden_dim, device=device)
    bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)

    limbic = Limbic(profile.vectors, device=device, steering_strength=1.0)
    temporal = Temporal(profile.vectors, device=device, steering_strength=0.3)
    bus.set_baseline(limbic.baseline_vector(), weight=1.0)

    # Subconscious with two samplers: noise + token-embedding sampling.
    embedding_matrix = model.get_input_embeddings().weight.detach().to(device)
    forbidden = [
        tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id,
    ]
    forbidden = [f for f in forbidden if f is not None]
    samplers = [
        NoiseSampler(profile.hidden_dim, device=device, magnitude=1.0),
        TokenEmbeddingSampler(
            embedding_matrix, device=device,
            candidate_pool=64, forbidden_token_ids=forbidden,
        ),
    ]
    subc = SubconsciousStack(
        hidden_dim=profile.hidden_dim, device=device,
        samplers=samplers,
        l0_per_sampler_count=4,
        l0_emotion_bias_strength=0.5,
        l1_keep_top_k=8,
        l2_keep_top_k=3,
        l2_surprise_probability=0.15,
        l3_perturbation_strength=0.4,
        emotion_vectors=profile.vectors,
    )

    runner = FlowRunner(
        bus=bus, regions=[limbic, temporal, subc],
        clock_config=cfg.clock, verbose_errors=True,
    )

    hook = SteeringHook()
    hook.attach(model, profile.target_layer)
    driver = BusSteeringDriver(bus, steering_strength=1.0, smoothing=0.0)

    runner.start()
    print(f"  flow runner started, flow_dt={cfg.clock.flow_dt_seconds*1000:.0f}ms")

    # ------------------- Phase 1: idle -------------------
    print(f"\n=== Phase 1: idle, watch subconscious ({args.idle_seconds:.0f}s) ===")
    idle_stats = sample_phase(
        "idle", args.idle_seconds, limbic, subc, tokenizer,
    )

    # ------------------- Phase 2: stimulate -------------------
    print(f"\n=== Phase 2: stimulate joy=1.0 curious=0.6 ({args.stim_seconds:.0f}s) ===")
    limbic.stimulate_many({"joy": 1.0, "curious": 0.6})
    stim_stats = sample_phase(
        "stim", args.stim_seconds, limbic, subc, tokenizer,
    )

    # ------------------- Phase 3: generate -------------------
    print(f"\n=== Phase 3: generate with live bus ===")
    snap_b = bus.snapshot()
    snap_l = limbic.snapshot()
    print(f"  bus: norm={snap_b['state_norm']:.3f}  div={snap_b['divergence']:.3f}")
    top_emo = sorted(snap_l['blended'].items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"  top emotions: {top_emo}")

    hook.set_steering(driver.read())
    out_steered = generate(model, tokenizer, PROMPT)
    print(f"  steered output: {out_steered}")
    hook.set_steering(None)
    out_raw = generate(model, tokenizer, PROMPT)
    print(f"  raw     output: {out_raw}")

    runner.stop()

    # -------- Verdicts --------
    print(f"\n=== Verdicts ===")
    varies = idle_stats["n_unique"] >= 5
    surprises = (idle_stats["surprises"] + stim_stats["surprises"]) > 0
    multi_source = (
        len(idle_stats["sources"]) >= 2 or len(stim_stats["sources"]) >= 2
    )
    print(f"  intrusives vary across ticks?     "
          f"{'OK' if varies else 'FAIL'}  "
          f"(idle unique={idle_stats['n_unique']}, stim unique={stim_stats['n_unique']})")
    print(f"  surprise promotions occurred?     "
          f"{'OK' if surprises else 'FAIL'}  "
          f"(idle={idle_stats['surprises']}, stim={stim_stats['surprises']})")
    print(f"  multiple samplers contribute?     "
          f"{'OK' if multi_source else 'FAIL'}  "
          f"(idle={idle_stats['sources']}, stim={stim_stats['sources']})")
    print(f"  steered != raw?                   "
          f"{'OK' if out_steered != out_raw else 'FAIL'}")


if __name__ == "__main__":
    main()
