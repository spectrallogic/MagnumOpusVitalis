"""V3 Era 4, M2 — motors that learn. The audit found the three motor
heads were frozen random matrices; these tests make that impossible to
regress. Building this surfaced a second honest failure worth its own
test: dense punishment collapsed the typing policy into all-silence
(byte 257 at p=1.0), where both the policy and entropy gradients are
exactly zero — the logit leash makes that state unreachable."""

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def _step(ctx, w, world, i, reward_fn=None):
    world.step(0.15)
    if i % 4 == 0:
        ctx.router.impulse()               # gates open: it acts
    ctx.drives.levels["energy"] = 1.0      # the womb keeps it awake
    w.step_once()
    tr = w._motor_trace or {}
    ids = tr["keys"][0] if tr.get("keys") else None
    if reward_fn is not None:
        ctx.drives.last_reward = reward_fn(ids or [])
    return ids, tr


def test_motor_heads_are_no_longer_frozen():
    """The audit's exact finding, inverted into a regression test."""
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e4_motor"
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=161)

    before = {
        "keys": w.keys_head[0].weight.detach().clone(),
        "voice": w.voice_head[0].weight.detach().clone(),
        "paint": w.paint_head[0].weight.detach().clone(),
    }
    for i in range(cfg.window + 20):
        _step(ctx, w, world, i,
              lambda ids: 0.02 if i % 3 else -0.01)

    pub = w.get_published()
    assert pub["motor"]["updates"] > 0, "no motor credit ever flowed"
    moved = {k: float((getattr(w, f"{k}_head")[0].weight.detach()
                       - v).abs().max()) for k, v in before.items()}
    assert all(m > 0 for m in moved.values()), \
        f"a motor head stayed frozen: {moved}"
    w.stop()


def test_shaped_reward_bends_the_typing():
    """Falsifiable shaping, designed after a wrong first attempt: you
    cannot shape a byte the organism never emits (rewarding unseen 'a'
    produced pure noise), so the target is chosen from a census of its
    OWN babble. Reward then follows that byte — its logit gain must be
    an extreme outlier against all 257 others."""
    torch.manual_seed(23)
    np.random.seed(23)
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e4_shape"
    cfg.w_motor = 0.6                      # loud lesson, short life
    cfg.motor_lr_mult = 100.0              # test-speed heat, same code
    cfg.chars_out_per_tick = 8             # denser emissions to shape
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=162)

    census = Counter()
    for i in range(cfg.window + 100):      # what does it already babble?
        ids, _ = _step(ctx, w, world, i)
        if ids:
            census.update(ids)
    target = census.most_common(1)[0][0]

    probe = torch.zeros(cfg.d_model, device=w.device)
    with torch.no_grad():
        l0 = w.keys_head(probe).detach().cpu().numpy()
    hits = 0
    for i in range(500):
        ids, _ = _step(ctx, w, world, i,
                       lambda ids: 0.5 if target in ids else -0.05)
        hits += target in (ids or [])
    with torch.no_grad():
        l1 = w.keys_head(probe).detach().cpu().numpy()

    d = (l1 - l0)[:258]
    da = float(d[target])
    rest = np.delete(d, target)
    z = (da - rest.mean()) / rest.std()
    rank = int((d >= da).sum())
    print(f"target={target} hits={hits}/500 delta={da:+.3f} "
          f"z={z:.2f} rank={rank}/258")
    assert hits > 10, "the rewarded byte stopped being emitted"
    assert rank <= 3, "rewarded byte's gain was not an outlier"
    assert z > 4.0, f"shaping not specific (z={z:.2f})"
    assert w.get_published()["motor"]["updates"] > 100
    w.stop()


def test_punishment_cannot_silence_the_keyboard():
    """The absorbing-state regression: 300 ticks of dense punishment
    must NOT collapse the policy into certainty (measured failure mode:
    p(silence)=1.0, all gradients zero, mutism forever)."""
    torch.manual_seed(31)
    np.random.seed(31)
    cfg = OrganismConfig()
    cfg.run_name = "_test_v3e4_mutism"
    cfg.w_motor = 0.6
    cfg.motor_lr_mult = 100.0
    cfg.chars_out_per_tick = 8
    ctx, board, flow = build(cfg)
    w = ctx.worker
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=163)

    last_tr = {}
    for i in range(cfg.window + 300):
        _, tr = _step(ctx, w, world, i, lambda ids: -0.05)  # all is pain
        last_tr = tr or last_tr
    with torch.no_grad():
        p = F.softmax(w.keys_head(last_tr["h"]) / last_tr["keys"][1], dim=-1)
    assert float(p.max()) < 0.99, \
        f"policy collapsed to certainty (pmax={float(p.max()):.4f})"
    assert w.wordstream.chars_typed_total > 0   # it still speaks
    w.stop()
