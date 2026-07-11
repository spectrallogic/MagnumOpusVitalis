"""V3 Era 6 — the Wheel states its confidence before the chapter
arrives and is Brier-scored on what it said; and the action-conditioning
probe, whose honest result (so far) is a NEGATIVE finding on record."""

import numpy as np
import torch

from primordium.config import OrganismConfig
from primordium.loom.wheel import Wheel


def _spin_regime(wh, base, turns, k, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(turns * k):
        wh.spin(base + torch.randn(base.shape, generator=g) * noise)


def test_wheel_states_confidence_before_the_outcome():
    cfg = OrganismConfig()
    cfg.wheel_window_ticks = 4
    torch.manual_seed(9)
    wh = Wheel(cfg)
    d = cfg.d_model
    home = torch.randn(d)

    _spin_regime(wh, home, turns=40, k=4, seed=1)
    snap = wh.snapshot()
    assert "calib" in snap and snap["calib"]["n"] > 10
    stable_brier = snap["calib"]["brier"]
    # a coin-flip forecaster scores 0.25 on Brier; a wheel that has
    # learned its own predictability must beat it on a stable chapter
    assert stable_brier < 0.25, f"brier={stable_brier}"
    assert snap["calib"]["conf"] is not None

    # a regime change produces outcomes its stated confidence did not
    # expect — Brier honestly worsens
    away = torch.randn(d)
    _spin_regime(wh, away, turns=6, k=4, seed=2)
    worse = wh.snapshot()["calib"]["brier"]
    assert worse >= stable_brier

    # the whole calibration ring survives a nap
    wh2 = Wheel(cfg)
    wh2.load_state_dict(wh.state_dict())
    wh2.load_bank_state(wh.bank_state())
    s1, s2 = wh.snapshot()["calib"], wh2.snapshot()["calib"]
    assert s1 == s2


def test_action_conditioning_probe_and_the_finding():
    """The probe: re-encode the lived ring with paint efference ZEROED
    and compare canvas prediction error against the true-efference
    forward. MEASURED FINDING (Era 6, ~400-tick womb lives): ratio
    ~1.000. Era 7 found the structural truth behind the number: the
    stroke that paints canvas_t is only recorded in proprio_t, causally
    invisible to a prediction formed a tick earlier — and with the
    canvas visible in context, older efference is redundant BY
    CONSTRUCTION. So this OPEN-EYES ratio measures redundancy, not use;
    USE is measured blindfolded by the Grip (test_v3e7, live probe on
    the dashboard). This probe stays as the redundancy tracker, and its
    banner still fires if open-eyes prediction ever leans on the hand
    (e.g. through multi-step or degraded-vision regimes)."""
    from primordium.run import build
    from primordium.senses import SyntheticWorld
    torch.manual_seed(7)
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e6_act"
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=191)
    for i in range(cfg.window + 120):
        world.step(0.15)
        if i % 4 == 0:
            ctx.router.impulse()
        ctx.drives.levels["energy"] = 1.0
        w.step_once()
    assert ctx.easel.strokes_total > 0     # it painted; actions existed

    G = w.tokenizer.group_size
    sens = w.tokenizer.sensory_slice
    n_vis = w.tokenizer.n_vis

    def cnv_err(zero_paint):
        entries = []
        for e in w.ring:
            e2 = dict(e)
            if zero_paint:
                p = e["proprio"].clone()
                p[15:23] = 0.0             # paint efference dims
                e2["proprio"] = p
            entries.append(e2)
        tokens = torch.cat([w._encode_entry(e) for e in entries], 0)
        S = tokens.shape[0]
        with torch.no_grad():
            hidden = w.model(tokens, group_size=G)
            pred = w.model.predictor(
                hidden[S - 2 * G + sens.start: S - 2 * G + sens.stop])
            last = entries[-1]
            tgt = w.objective.targets(last["vis"], last["aud"],
                                      last["txt"], last["cnv"])
            online = hidden[S - G + sens.start: S - G + sens.stop]
            _l, parts = w.objective.loss(pred, tgt, online, n_vis)
        return parts["cnv"]

    true_e, zero_e = cnv_err(False), cnv_err(True)
    ratio = zero_e / max(true_e, 1e-9)
    print(f"action-conditioning probe: cnv err true={true_e:.6f} "
          f"zeroed={zero_e:.6f} ratio={ratio:.4f}")
    assert np.isfinite(true_e) and np.isfinite(zero_e)
    assert true_e > 0 and zero_e > 0
    # If this ever fires, celebrate AND update BLUEPRINT_MAP: the model
    # will have started genuinely predicting through its own actions.
    if ratio > 1.2:
        print("FINDING CHANGED: efference is now load-bearing "
              f"(ratio {ratio:.3f}) — update docs/BLUEPRINT_MAP.md")
    w.stop()
