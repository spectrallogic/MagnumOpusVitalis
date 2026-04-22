"""
Memory + FalseMemoryConfabulator + MemorySampler verification.

Loads gpt2 + profile, builds bus + Limbic + Temporal + Subconscious +
Memory + Confabulator. Stimulates several distinct emotions over time so
the bus visits diverse latent regions, triggering captures. Then forces
the slow clock to fire confabulation by lowering its period for the demo.

Verdicts:
  - Memory pool grows during stimulation (auto-capture working)
  - Force-capture also works (one explicit call)
  - After confabulation, pool contains entries with confidence < 1.0
  - MemorySampler is a real source in the subconscious stream after
    memories accumulate (look for source="memory" in intrusive log)
"""

import argparse
import time
from collections import Counter

import torch

from magnum_opus.loader import load_model
from magnum_opus.profile import load_profile

from magnum_opus_v2 import (
    LatentBus, FlowRunner, V2Config, ClockConfig,
    Limbic, Temporal, SubconsciousStack, Memory, FalseMemoryConfabulator,
    NoiseSampler, TokenEmbeddingSampler,
    SteeringHook, BusSteeringDriver,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seconds-per-emotion", type=float, default=2.0)
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    profile = load_profile(args.model)
    print(f"  Profile: layer={profile.target_layer}, dim={profile.hidden_dim}")

    # Speed up the slow clock for the demo so we actually see confabulation.
    clock = ClockConfig(slow_dt_seconds=3.0)
    cfg = V2Config(hidden_dim=profile.hidden_dim, device=device, clock=clock)
    bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)

    limbic = Limbic(profile.vectors, device=device, steering_strength=1.0)
    temporal = Temporal(profile.vectors, device=device, steering_strength=0.3)
    bus.set_baseline(limbic.baseline_vector(), weight=1.0)

    memory = Memory(
        device=device, capacity=80,
        capture_velocity_threshold=0.3,
        capture_divergence_threshold=0.10,
        capture_cooldown_seconds=0.5,
    )
    confab = FalseMemoryConfabulator(
        memory, per_tick_count=2, noise_strength=0.15,
        max_false_fraction=0.30,
    )

    embedding_matrix = model.get_input_embeddings().weight.detach().to(device)
    samplers = [
        NoiseSampler(profile.hidden_dim, device=device, magnitude=1.0),
        TokenEmbeddingSampler(embedding_matrix, device=device, candidate_pool=64),
        memory.make_sampler(),  # reads the live pool
    ]
    subc = SubconsciousStack(
        hidden_dim=profile.hidden_dim, device=device, samplers=samplers,
        l0_per_sampler_count=4, l0_emotion_bias_strength=0.5,
        l1_keep_top_k=8, l2_keep_top_k=3,
        l2_surprise_probability=0.15,
        l3_perturbation_strength=0.4,
        emotion_vectors=profile.vectors,
    )

    runner = FlowRunner(
        bus=bus, regions=[limbic, temporal, subc, memory, confab],
        clock_config=cfg.clock, verbose_errors=True,
    )

    hook = SteeringHook()
    hook.attach(model, profile.target_layer)
    driver = BusSteeringDriver(bus, steering_strength=1.0)

    runner.start()
    print(f"  flow runner started (slow_dt={cfg.clock.slow_dt_seconds}s for demo)")

    # Tour several emotions so the bus visits diverse regions and Memory captures.
    tour = ["joy", "fear", "sadness", "curious", "anger", "trust"]
    intrusive_sources = []

    for i, emo in enumerate(tour):
        print(f"\n=== {i+1}/{len(tour)}: stimulate {emo}=1.0 ({args.seconds_per_emotion}s) ===")
        limbic.stimulate(emo, 1.0)
        end = time.monotonic() + args.seconds_per_emotion
        while time.monotonic() < end:
            subc.set_emotion_blend(limbic.snapshot()["blended"])
            snap = subc.snapshot()
            if snap.get("intrusive_source"):
                intrusive_sources.append(snap["intrusive_source"])
            time.sleep(0.1)
        m_snap = memory.snapshot()
        print(f"  memory: size={m_snap['size']}  "
              f"avg_imp={m_snap['avg_importance']:.3f}  "
              f"max_imp={m_snap['max_importance']:.3f}  "
              f"n_false={m_snap['n_false']}")

    # Force one explicit capture
    memory.force_capture(bus, importance=2.0, tag="demo_explicit")
    print(f"\n  after force_capture: size={memory.snapshot()['size']}")

    # Wait long enough for confabulation to fire at least twice
    print(f"\n=== Wait 8s for false-memory confabulation ===")
    end = time.monotonic() + 8.0
    while time.monotonic() < end:
        subc.set_emotion_blend(limbic.snapshot()["blended"])
        snap = subc.snapshot()
        if snap.get("intrusive_source"):
            intrusive_sources.append(snap["intrusive_source"])
        time.sleep(0.1)
    m_snap = memory.snapshot()
    print(f"  memory: size={m_snap['size']}  n_false={m_snap['n_false']}")

    # Generate
    hook.set_steering(driver.read())
    enc = tokenizer("Today I went outside and", return_tensors="pt").to(device)
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=30, do_sample=True,
            top_p=0.92, temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"\n  steered output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    runner.stop()

    # Verdicts
    src_counts = Counter(intrusive_sources)
    print(f"\n=== Verdicts ===")
    print(f"  intrusive sources observed: {dict(src_counts)}")
    n_total = sum(src_counts.values())
    n_memory = sum(v for k, v in src_counts.items() if "memory" in (k or ""))
    pool_size = memory.snapshot()["size"]
    n_false = memory.snapshot()["n_false"]
    print(f"  pool grew?                  "
          f"{'OK' if pool_size >= 5 else 'FAIL'}  (size={pool_size})")
    print(f"  false memories generated?   "
          f"{'OK' if n_false >= 1 else 'FAIL'}  (n_false={n_false})")
    print(f"  memory entered subconscious? "
          f"{'OK' if n_memory > 0 else 'PARTIAL'}  "
          f"({n_memory}/{n_total} intrusives mention memory)")


if __name__ == "__main__":
    main()
