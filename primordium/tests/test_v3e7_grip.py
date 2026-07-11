"""V3 Era 7 — the Grip: the hand made load-bearing.

Era 6 measured the negative: paint efference was an input the world
model ignored (counterfactual ratio ~1.0). Chasing it exposed the
structural truth: the stroke that paints canvas_t is only recorded in
proprio_t — causally INVISIBLE to the prediction formed a tick earlier
— and with the canvas visible in context, older efference is redundant
by construction. The open-eyes probe measures redundancy, not use.

The Grip therefore trains and measures UNDER A BLINDFOLD: recent
canvases masked, canvas loss delta-weighted toward the tokens the
strokes changed, plus an inverse-dynamics head. The era's claims,
each falsifiable here:

  1. MICRO SCALE, hot and isolated: the blindfold pressure CARVES the
     route — after training, erasing the efference must hurt
     blindfolded canvas prediction by a real margin. (The mechanism
     works.)
  2. ORGANISM SCALE: the same machinery runs in the womb — gradient
     demonstrably flows into the proprio pathway, the live probe
     measures, and the ratio is REPORTED, not asserted: at the
     organism's gentle live learning rate the carve is a lifetime
     process, and pretending a 90-second womb proves it would be the
     exact dishonesty this project exists to avoid.
"""

import numpy as np
import torch
import torch.nn.functional as F

from primordium.body.expression import PAD, Easel
from primordium.config import OrganismConfig
from primordium.loom.core import LoomCore
from primordium.loom.grip import Grip, PAINT_SLICE
from primordium.loom.objective import JepaObjective
from primordium.loom.tokenizer import SensoryTokenizer


def test_blindfold_masks_only_recent_canvases():
    cfg = OrganismConfig()
    torch.manual_seed(3)
    grip = Grip(cfg)
    tok = SensoryTokenizer(cfg, stage=0)
    G, n_cnv = tok.group_size, tok.n_cnv
    n_groups, m = 6, 3
    tokens = torch.randn(n_groups * G, cfg.d_model)
    masked = grip.mask_canvas(tokens, G, n_cnv, m)

    changed = (masked != tokens).any(dim=-1)
    expect = torch.zeros(n_groups * G, dtype=torch.bool)
    for g in range(n_groups - 1 - m, n_groups - 1):
        expect[(g + 1) * G - n_cnv: (g + 1) * G] = True
    assert torch.equal(changed, expect)
    # the final group's canvas is untouched (it cannot leak into the
    # prediction; masking it would only corrupt the online latents)
    assert not changed[(n_groups - 1) * G:].any()


def test_inverse_head_recovers_actions_from_state_pairs():
    cfg = OrganismConfig()
    torch.manual_seed(5)
    grip = Grip(cfg)
    d, A = cfg.d_model, SensoryTokenizer.PROPRIO_DIM
    W = torch.randn(2 * d, A) * 0.3          # the world's secret
    opt = torch.optim.Adam(grip.inv.parameters(), lr=1e-3)

    def batch():
        before = torch.randn(64, d)
        after = torch.randn(64, d)
        act = torch.tanh(torch.cat([before, after], -1) @ W)
        return before, after, act

    b, a, act = batch()
    with torch.no_grad():
        first = float(F.smooth_l1_loss(
            grip.inv(torch.cat([b, a], -1)), act))
    for _ in range(400):
        b, a, act = batch()
        loss = F.smooth_l1_loss(grip.inv(torch.cat([b, a], -1)), act)
        opt.zero_grad()
        loss.backward()
        opt.step()
    b, a, act = batch()
    with torch.no_grad():
        last = float(F.smooth_l1_loss(
            grip.inv(torch.cat([b, a], -1)), act))
    assert last < 0.5 * first, f"inverse head did not learn: {first}->{last}"


def test_checkpoint_carries_the_grip():
    cfg = OrganismConfig()
    torch.manual_seed(7)
    g1 = Grip(cfg)
    g1.note_probe(1.31)
    g1.note_probe(1.18)
    g1.note_inv(0.02)
    g1.note_grip(0.4)
    g1.note_hand(0.013)
    g2 = Grip(cfg)
    g2.load_state_dict(g1.state_dict())
    assert g1.snapshot() == g2.snapshot()
    assert g2.snapshot()["probes"] == 2


# ----------------------------------------------------------------------
# the era's claim, at the scale where it is honestly provable
# ----------------------------------------------------------------------
def _proprio_of(paint: np.ndarray) -> torch.Tensor:
    p = np.zeros(SensoryTokenizer.PROPRIO_DIM, dtype=np.float32)
    p[PAINT_SLICE] = paint
    p[23:26] = [0.0, 0.0, 1.0]               # gaze: wide, centered
    return torch.from_numpy(p)


def test_blindfold_pressure_carves_the_route():
    """A small REAL Loom (4 blocks), the REAL Easel physics, the REAL
    grip code — trained hot and isolated on the blindfolded,
    delta-weighted canvas term over fresh randomly-stroked windows.
    Fresh strokes every window: nothing can be memorized; the only way
    to predict the hidden canvas stretch is to READ THE HAND. After
    training, erasing the paint efference must hurt.

    The perceptual pathway is FROZEN here: with the grip term isolated
    from the main objective there is no VICReg to stop the canvas
    embedder collapsing every canvas to one latent (measured: it does,
    loss 6e-5 with and without the hand). In the organism the live
    objective guards that; this test asks whether the hand model +
    trunk can learn the efference route. This is the pressure the
    organism lives with at lower heat. Measured while building: without
    the supervised hand model the ratio stays pinned at 1.0000 (two
    failed designs, recorded in grip.py's docstring); with it, the
    carve shows within ~250 steps and reaches ~4x by 4000."""
    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OrganismConfig()
    W, m = 8, cfg.grip_mask_groups

    tok = SensoryTokenizer(cfg, stage=0).to(dev)
    for p in tok.parameters():
        p.requires_grad_(False)               # frozen senses, honest targets
    model = LoomCore(cfg, {"mlp_dims": [cfg.d_model * 2] * 4}).to(dev)
    obj = JepaObjective(cfg, tok).to(dev)
    grip = Grip(cfg).to(dev)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(grip.parameters()), lr=1e-3)

    G = tok.group_size
    sens = tok.sensory_slice
    c0 = cfg.audio_tokens + tok.n_vis + cfg.text_tokens
    spec = cfg.stages[0]
    vis = tok.patchify(np.zeros((spec.res, spec.res, spec.channels),
                                dtype=np.uint8)).to(dev)
    aud = tok.melify(torch.zeros(
        cfg.n_mels, cfg.audio_tokens * cfg.mel_frames_per_token)).to(dev)
    txt = torch.full((cfg.text_tokens,), PAD, dtype=torch.long, device=dev)
    intero = torch.zeros(SensoryTokenizer.INTERO_DIM, device=dev)
    state = torch.zeros(cfg.state_tokens, cfg.d_model, device=dev)

    def make_window():
        easel = Easel(cfg)
        entries = []
        a_prev = np.zeros(8, dtype=np.float32)
        for _ in range(W):
            cnv = tok.cnvify(easel.view()).to(dev)
            entries.append({"cnv": cnv,
                            "proprio": _proprio_of(a_prev).to(dev)})
            a = rng.uniform(-1, 1, 8).astype(np.float32)
            a[0] = 0.5 + abs(a[0]) * 0.5      # gate firmly open
            a[3] = abs(a[3])                  # a visible radius
            a[7] = 0.8                        # strong alpha
            easel.paint_impulse()
            assert easel.stroke(a)
            a_prev = a
        return entries

    def encode(entries, zero_paint: bool):
        groups = []
        for e in entries:
            p = e["proprio"]
            if zero_paint:
                p = p.clone()
                p[PAINT_SLICE] = 0.0
            groups.append(tok.encode_tick(vis, aud, intero, p, state,
                                          txt, e["cnv"]))
        return torch.cat(groups, dim=0)

    def blind_err(entries, zero_paint: bool, train: bool = False):
        paints = torch.stack([e["proprio"][PAINT_SLICE] for e in entries])
        if zero_paint:
            paints = torch.zeros_like(paints)
        with torch.no_grad():
            lats = torch.stack([obj.target_ln(obj.t_canvas(e["cnv"]))
                                for e in entries])
            base = lats[W - 2 - m]
            target = obj.targets(vis, aud, txt, entries[-1]["cnv"])
            w_tok = grip.delta_weights(base, target[c0:])
        tokens = grip.mask_canvas(encode(entries, zero_paint),
                                  G, tok.n_cnv, m, paints=paints,
                                  base=base)
        hidden = model(tokens, group_size=G)
        S = tokens.shape[0]
        pred = model.predictor(
            hidden[S - 2 * G + sens.start: S - 2 * G + sens.stop])
        l = grip.cnv_loss(pred[c0:], target[c0:], w_tok)
        if train:   # the hand learns its strokes, supervised
            l = l + grip.hand_loss(paints[1:], lats[1:] - lats[:-1])
        return l

    B = 4
    for step in range(700):
        loss = sum(blind_err(make_window(), False, train=True)
                   for _ in range(B)) / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        errs_t, errs_z = [], []
        for _ in range(10):
            entries = make_window()
            errs_t.append(float(blind_err(entries, False)))
            errs_z.append(float(blind_err(entries, True)))
    ratio = float(np.mean(errs_z)) / max(float(np.mean(errs_t)), 1e-9)
    print(f"grip micro: blindfold ratio={ratio:.4f} "
          f"(true={np.mean(errs_t):.5f} zeroed={np.mean(errs_z):.5f})")
    assert ratio > 1.2, (   # measured 1.36 at seed 13; null sits at 1.0000
        f"the blindfold pressure did not carve the route "
        f"(ratio {ratio:.4f}) — the grip failed its own reason to exist")


# ----------------------------------------------------------------------
# organism scale: machinery, gradient flow, and honest measurement
# ----------------------------------------------------------------------
def test_grip_runs_in_the_womb_and_gradient_reaches_the_hand():
    """A womb life with the grip hot: the blindfold trains only where
    the hand acted, the live probe measures under the mask, gradient
    demonstrably flows into the proprio pathway — and the measured
    ratios are PRINTED, not asserted. At the organism's gentle live
    learning rate the route carves over a lifetime, tracked by the
    probe on the dashboard; the Era-6 banner still fires the day
    open-eyes prediction leans on the hand."""
    from primordium.run import build
    from primordium.senses import SyntheticWorld
    torch.manual_seed(11)
    np.random.seed(11)
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e7_grip"
    cfg.grip_every = 2
    cfg.grip_probe_every = 50
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=311)
    for i in range(cfg.window + 300):
        world.step(0.15)
        if i % 4 == 0:
            ctx.router.impulse()
        ctx.drives.levels["energy"] = 1.0
        w.step_once()
    assert w.easel.strokes_total > 0          # actions existed
    snap = w.grip.snapshot()
    assert snap["steps"] > 30, f"blindfold barely trained: {snap}"
    assert snap["probes"] > 0                 # the live probe measured
    assert np.isfinite(snap["inv_loss"]) and snap["inv_loss"] > 0

    # provoke strokes into the maskable stretch, then prove gradient
    # flows from the blindfolded canvas loss into the hand's pathway
    extra = 0
    while not w._hand_acted() and extra < 200:
        world.step(0.15)
        ctx.router.impulse()
        ctx.drives.levels["energy"] = 1.0
        w.step_once()
        extra += 1
    assert w._hand_acted(), "could not provoke strokes to measure"

    tokens = torch.cat([w._encode_entry(e) for e in w.ring], dim=0)
    last = w.ring[-1]
    target = w.objective.targets(last["vis"], last["aud"],
                                 last["txt"], last["cnv"])
    w.opt.zero_grad(set_to_none=True)
    term = w._grip_term(tokens, target, {})
    term.backward()
    g = w.tokenizer.proprio_embed[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0, (
        "no gradient reached the proprio embedder — the blindfold "
        "is not training the hand's route")
    w.opt.zero_grad(set_to_none=True)

    # the honest numbers, on the record (probe = blindfolded ratio)
    ratio = w._grip_probe(target)
    print(f"grip womb: blindfold probe ratio="
          f"{ratio if ratio is None else round(ratio, 4)} · "
          f"snapshot={w.grip.snapshot()}")
    if ratio is not None and ratio > 1.2:
        print("ROUTE CARVED AT WOMB SCALE — update docs/BLUEPRINT_MAP.md")
    w.stop()
