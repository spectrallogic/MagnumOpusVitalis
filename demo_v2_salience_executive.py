"""
Salience + Executive verification.

Two checks:

  1. Salience gates the SubconsciousStack. Watch attention_gain over time:
     it should rise when intrusives are novel (different from recent bus
     history) and fall when redundant. Idle periods should boost it.

  2. Executive pressure rises when the bus is held off-baseline (we
     simulate a sustained tilt with a stimulation that decays slowly),
     and the on_should_speak callback fires at least once.

No model loaded — pure substrate test, runs fast.
"""

import argparse
import time

from magnum_opus.profile import load_profile

from magnum_opus_v2 import (
    LatentBus, FlowRunner, V2Config, ClockConfig,
    Limbic, SubconsciousStack, Salience, Executive,
    NoiseSampler,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="Used only to load profile vectors; no model run.")
    ap.add_argument("--idle", type=float, default=4.0)
    ap.add_argument("--press", type=float, default=15.0)
    args = ap.parse_args()

    profile = load_profile(args.model)
    print(f"  Profile: dim={profile.hidden_dim}")

    # Speed up perception clock so Executive ticks faster for the demo
    cfg = V2Config(
        hidden_dim=profile.hidden_dim, device="cpu",
        clock=ClockConfig(perception_dt_seconds=0.1),
    )
    bus = LatentBus(profile.hidden_dim, device="cpu", config=cfg.bus)

    limbic = Limbic(profile.vectors, device="cpu", steering_strength=1.0)
    bus.set_baseline(limbic.baseline_vector(), weight=1.0)

    subc = SubconsciousStack(
        hidden_dim=profile.hidden_dim, device="cpu",
        samplers=[NoiseSampler(profile.hidden_dim, device="cpu", magnitude=1.0)],
        l3_perturbation_strength=0.3,
        emotion_vectors=profile.vectors,
    )
    salience = Salience(subc, history_size=20)

    speech_calls = []
    def on_speak():
        speech_calls.append(time.monotonic())
        print(f"  *** Executive callback fired at t={time.monotonic()-t0:.1f}s "
              f"(pressure={execu.snapshot()['pressure']:.3f}) ***")

    execu = Executive(
        base_threshold=0.20,              # low for demo
        pressure_growth=1.5,              # fast accumulation for demo
        pressure_decay=0.985,             # slow decay so it can build
        freshness_tau_seconds=2.0,        # decay freshness fast for demo
        post_speech_silence_seconds=2.0,
        on_should_speak=on_speak,
    )

    runner = FlowRunner(
        bus=bus, regions=[limbic, subc, salience, execu],
        clock_config=cfg.clock, verbose_errors=True,
    )
    runner.start()
    t0 = time.monotonic()
    print(f"  flow runner started")

    # Mark a fake user interaction so Executive starts with high freshness
    execu.mark_interaction()

    # Phase 1: idle, watch salience gain
    print(f"\n=== Phase 1: idle {args.idle}s, watch Salience gating ===")
    gains = []
    novelties = []
    end = time.monotonic() + args.idle
    while time.monotonic() < end:
        subc.set_emotion_blend(limbic.snapshot()["blended"])
        s = salience.snapshot()
        gains.append(s["gain"])
        novelties.append(s["novelty"])
        time.sleep(0.2)
    print(f"  gain: min={min(gains):.3f} mean={sum(gains)/len(gains):.3f} "
          f"max={max(gains):.3f}")
    print(f"  novelty: min={min(novelties):.3f} mean={sum(novelties)/len(novelties):.3f} "
          f"max={max(novelties):.3f}")

    # Phase 2: hold the bus off-baseline by stimulating repeatedly
    print(f"\n=== Phase 2: sustained stimulation for {args.press}s — Executive should fire ===")
    end = time.monotonic() + args.press
    pressures = []
    while time.monotonic() < end:
        # Re-stimulate every 0.5s to keep the bus off-baseline
        limbic.stimulate_many({"desperate": 0.6, "fear": 0.4})
        for _ in range(5):
            subc.set_emotion_blend(limbic.snapshot()["blended"])
            pressures.append(execu.snapshot()["pressure"])
            time.sleep(0.1)
        snap_b = bus.snapshot()
        snap_e = execu.snapshot()
        snap_s = salience.snapshot()
        print(f"  t={time.monotonic()-t0:5.1f}s  "
              f"bus_div={snap_b['divergence']:.3f}  "
              f"vel={snap_b['velocity_norm']:.3f}  "
              f"pressure={snap_e['pressure']:.3f}/{snap_e['effective_threshold']:.3f}  "
              f"freshness={snap_e['freshness']:.3f}  "
              f"sal_gain={snap_s['gain']:.3f}")

    runner.stop()

    # Verdicts
    print(f"\n=== Verdicts ===")
    print(f"  salience gain varies?      "
          f"{'OK' if max(gains) - min(gains) > 0.1 else 'PARTIAL'}  "
          f"(range={max(gains)-min(gains):.3f})")
    print(f"  executive pressure built?  "
          f"{'OK' if max(pressures) > 0.3 else 'FAIL'}  (peak={max(pressures):.3f})")
    print(f"  callback fired?            "
          f"{'OK' if len(speech_calls) > 0 else 'FAIL'}  (count={len(speech_calls)})")


if __name__ == "__main__":
    main()
