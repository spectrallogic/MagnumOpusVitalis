"""Mind's eye — the imagined stream renders when open, and the
subconscious drinks from it either way."""

from primordium.config import OrganismConfig
from primordium.run import build
from primordium.senses import SyntheticWorld


def test_minds_eye_stream_and_subconscious_coupling():
    cfg = OrganismConfig()
    cfg.run_name = "_test_meye"
    ctx, board, flow = build(cfg)
    world = SyntheticWorld(ctx.retina, ctx.cochlea, seed=11)
    w = ctx.worker

    w.set_minds_eye(True)
    for _ in range(20):
        world.step(0.15)
        w.step_once()

    pub = w.get_published()
    assert isinstance(pub.get("mind_png"), bytes) and len(pub["mind_png"]) > 100
    mel = pub.get("mind_mel")
    assert isinstance(mel, list) and len(mel) == cfg.n_mels

    # the ring always flows, and the sampler serves it to the subconscious
    assert len(w.minds_eye_ring) > 10
    cands = w.minds_eye_ring.sample(4)
    assert cands and all(c.source.startswith("minds_eye") for c in cands)

    # closing the eye stops the render (the internal stream continues)
    w.set_minds_eye(False)
    n_before = len(w.minds_eye_ring)
    world.step(0.15)
    w.step_once()
    assert len(w.minds_eye_ring) >= n_before   # still flowing inside
    w.stop()
