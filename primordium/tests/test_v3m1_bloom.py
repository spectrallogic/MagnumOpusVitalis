"""V3-M1 — growth surgery is EXACT: a bloomed brain gives the same
answers the moment after surgery as the moment before, then learns."""

import torch

from primordium.config import OrganismConfig
from primordium.loom.bloom import Bloom, insert_block, widen_mlp
from primordium.loom.core import LoomCore, default_anatomy
from primordium.loom.fringe import Fringe


def _toy_core(cfg, dims):
    torch.manual_seed(11)
    return LoomCore(cfg, {"mlp_dims": dims})


def test_default_anatomy_is_the_old_body():
    cfg = OrganismConfig()
    core = LoomCore(cfg)
    a = core.anatomy()
    assert a["mlp_dims"] == [cfg.d_model * cfg.mlp_ratio] * cfg.n_layers
    assert a["n_blocks"] == cfg.n_layers and a["d_model"] == cfg.d_model
    assert default_anatomy(cfg)["mlp_dims"] == a["mlp_dims"]


def test_insert_block_is_identity_at_birth():
    cfg = OrganismConfig()
    core = _toy_core(cfg, [512, 512, 512])
    tokens = torch.randn(2 * 37, cfg.d_model) * 0.1
    before = core(tokens, group_size=37)
    anatomy = insert_block(core, len(core.blocks) - 1)
    after = core(tokens, group_size=37)
    assert torch.allclose(before, after, atol=1e-5), \
        float((before - after).abs().max())
    assert anatomy["n_blocks"] == 4
    assert anatomy["mlp_dims"][2] == cfg.d_model * cfg.mlp_ratio


def test_widen_mlp_is_silent_at_birth():
    cfg = OrganismConfig()
    core = _toy_core(cfg, [512, 512, 512])
    old_up = core.blocks[1].mlp[0].weight.detach().clone()
    old_down = core.blocks[1].mlp[2].weight.detach().clone()
    tokens = torch.randn(2 * 37, cfg.d_model) * 0.1
    before = core(tokens, group_size=37)
    anatomy = widen_mlp(core, 1, 64)
    after = core(tokens, group_size=37)
    assert torch.allclose(before, after, atol=1e-5)
    assert anatomy["mlp_dims"] == [512, 576, 512]
    # the old voice is preserved verbatim, the new one is silent
    assert torch.equal(core.blocks[1].mlp[0].weight[:512], old_up)
    assert torch.equal(core.blocks[1].mlp[2].weight[:, :512], old_down)
    assert float(core.blocks[1].mlp[2].weight[:, 512:].abs().max()) == 0.0


def test_grown_capacity_actually_learns():
    """The new block must come alive: its zeroed gates get gradients,
    move off zero, and the whole grown core reduces a real objective."""
    cfg = OrganismConfig()
    core = _toy_core(cfg, [512, 512])
    insert_block(core, 1)
    widen_mlp(core, 0, 64)
    new_blk = core.blocks[1]
    opt = torch.optim.AdamW(core.parameters(), lr=3e-4)
    x = torch.randn(2 * 37, cfg.d_model) * 0.1
    y = torch.randn(2 * 37, cfg.d_model) * 0.1
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(core(x, group_size=37), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.9
    assert float(new_blk.mlp[2].weight.abs().max()) > 0     # it woke up
    assert float(new_blk.attn.out_proj.weight.abs().max()) > 0


def test_fringe_survives_surgery():
    """Detach, grow, re-attach: edges stay edges, silence stays exact."""
    cfg = OrganismConfig()
    core = _toy_core(cfg, [512, 512, 512])
    fringe = Fringe(cfg, core)
    with torch.no_grad():
        fringe.sprouts[0].B.weight.normal_(std=0.05)
    tokens = torch.randn(2 * 37, cfg.d_model) * 0.1
    with_sprout = core(tokens, group_size=37)

    fringe.detach()
    bare = core(tokens, group_size=37)
    assert not torch.allclose(with_sprout, bare, atol=1e-6)  # it let go

    last_before = core.blocks[-1]
    insert_block(core, len(core.blocks) - 1)
    assert core.blocks[-1] is last_before      # the edge block is the same
    fringe2 = Fringe(cfg, core)                # newborn hive, silent
    regrown = core(tokens, group_size=37)
    assert torch.allclose(bare, regrown, atol=1e-5)
    assert len(fringe2) == 2 * cfg.fringe_sprouts_per_site


def test_bloom_manager_plans_and_grows():
    cfg = OrganismConfig()
    cfg.bloom_widen_k = 32
    core = _toy_core(cfg, [256, 256, 256])
    bloom = Bloom(cfg, core)
    bloom.strain = [0.1, 0.9, 0.2]             # block 1 works hardest

    r1 = bloom.grow(tick_id=100)               # bloom #1: width, block 1
    assert r1["growth"] == "width" and r1["site"] == 1
    assert r1["params_after"] > r1["params_before"]
    assert core.anatomy()["mlp_dims"][1] == 288

    r2 = bloom.grow(tick_id=200)               # bloom #2: width again
    assert r2["growth"] == "width"

    r3 = bloom.grow(tick_id=300)               # bloom #3: depth
    assert r3["growth"] == "block" and r3["site"] == 2
    assert core.anatomy()["n_blocks"] == 4
    assert len(bloom.strain) == 4              # gauges grew with the body
    assert bloom.blooms_total == 3 and bloom.last_bloom_tick == 300

    st = bloom.state_dict()
    b2 = Bloom(cfg, core)
    b2.load_state_dict(st)
    assert b2.blooms_total == 3 and b2.strain == bloom.strain
