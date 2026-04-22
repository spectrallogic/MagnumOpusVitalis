"""
Substrate-only verification for magnum_opus_v2.

Runs the LatentBus + FlowRunner with no model attached. Two scenarios:

  1. Idle  — only attractor pull + temperature noise. Bus norm should stay
             small and bounded (returns to baseline). Velocity should be
             nonzero from noise but should not grow.

  2. Perturbed — register a PerturbationRegion on the flow clock. Bus norm
             should rise, velocity should reflect motion toward the
             perturbation direction, then equilibrate at a balance point
             where attractor pull cancels the perturbation.

Logs a snapshot every 100ms for the configured duration and prints a small
table at the end. No model passes anywhere — proves the substrate flows
on its own.

Usage:
    python demo_v2_substrate.py                      # 5s idle, 10s perturbed
    python demo_v2_substrate.py --idle 10 --pert 20  # custom durations
"""

import argparse
import time

from magnum_opus_v2 import (
    LatentBus, FlowRunner, V2Config, NoOpRegion, PerturbationRegion,
)


def run_phase(name: str, runner: FlowRunner, seconds: float) -> list:
    """Sample bus snapshots every 100ms for `seconds`. Return list of dicts."""
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        samples.append({"phase": name, **runner.bus.snapshot()})
        time.sleep(0.1)
    return samples


def summarize(samples: list, label: str) -> None:
    if not samples:
        print(f"  [{label}] no samples")
        return
    norms = [s["state_norm"] for s in samples]
    vels = [s["velocity_norm"] for s in samples]
    divs = [s["divergence"] for s in samples]
    n = len(samples)
    print(
        f"  [{label}] n={n:3d}  "
        f"state_norm  min={min(norms):.4f} mean={sum(norms)/n:.4f} max={max(norms):.4f}  "
        f"vel_norm  min={min(vels):.4f} mean={sum(vels)/n:.4f} max={max(vels):.4f}  "
        f"div  mean={sum(divs)/n:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dim", type=int, default=768)
    ap.add_argument("--idle", type=float, default=5.0,
                    help="Idle phase duration (seconds)")
    ap.add_argument("--pert", type=float, default=10.0,
                    help="Perturbation phase duration (seconds)")
    ap.add_argument("--pert-mag", type=float, default=2.0,
                    help="Perturbation magnitude")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = V2Config(hidden_dim=args.hidden_dim, device=args.device)
    bus = LatentBus(args.hidden_dim, device=args.device, config=cfg.bus)

    # Start with no flow regions — only substrate dynamics (attractor + noise).
    runner = FlowRunner(
        bus=bus, regions=[], clock_config=cfg.clock,
        verbose_errors=True,
    )
    runner.start()
    print(f"  flow runner started: pid={id(runner)}, "
          f"flow_dt={cfg.clock.flow_dt_seconds*1000:.0f}ms")

    # Phase 1: idle. Should stay near baseline.
    print(f"\n=== Phase 1: idle for {args.idle:.1f}s ===")
    idle = run_phase("idle", runner, args.idle)
    summarize(idle, "idle")

    # Phase 2: register perturbation region, run again.
    pert_region = PerturbationRegion(
        hidden_dim=args.hidden_dim, device=args.device,
        magnitude=args.pert_mag, seed=42,
    )
    runner.add_region(pert_region)
    print(f"\n=== Phase 2: perturbed for {args.pert:.1f}s "
          f"(direction magnitude={args.pert_mag}) ===")
    pert = run_phase("pert", runner, args.pert)
    summarize(pert, "pert")

    # Drop the perturbation region, watch it return to baseline.
    runner.regions = [r for r in runner.regions if r is not pert_region]
    relax_seconds = 3.0
    print(f"\n=== Phase 3: relax for {relax_seconds:.1f}s after dropping perturb ===")
    relax = run_phase("relax", runner, relax_seconds)
    summarize(relax, "relax")

    # Per-clock metrics
    print("\n=== Clock metrics ===")
    for cname, m in runner.metrics.items():
        ticks = m["ticks"]
        last_dt = m["last_wall_dt"]
        target_dt = getattr(cfg.clock, f"{cname}_dt_seconds")
        print(f"  {cname:11s}  ticks={ticks:5d}  "
              f"last_wall_dt={last_dt*1000:7.1f}ms  "
              f"target={target_dt*1000:6.0f}ms  "
              f"jitter={(last_dt - target_dt)*1000:+7.1f}ms")

    runner.stop()
    print("\n  flow runner stopped")

    # Quick verdicts
    print("\n=== Verdicts ===")
    idle_max_norm = max(s["state_norm"] for s in idle) if idle else 0.0
    pert_max_norm = max(s["state_norm"] for s in pert) if pert else 0.0
    relax_end_norm = relax[-1]["state_norm"] if relax else 0.0
    print(f"  idle bounded?      "
          f"{'OK' if idle_max_norm < 1.5 else 'FAIL'}  (idle max norm = {idle_max_norm:.3f})")
    print(f"  perturb moves bus? "
          f"{'OK' if pert_max_norm > idle_max_norm * 1.5 else 'FAIL'}  "
          f"(pert max = {pert_max_norm:.3f}, idle max = {idle_max_norm:.3f})")
    print(f"  relax returns?     "
          f"{'OK' if relax_end_norm < pert_max_norm * 0.8 else 'PARTIAL'}  "
          f"(relax end = {relax_end_norm:.3f}, pert peak = {pert_max_norm:.3f})")


if __name__ == "__main__":
    main()
