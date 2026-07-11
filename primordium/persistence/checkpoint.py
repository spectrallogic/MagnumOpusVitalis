"""
Checkpoint — the whole organism, so a restart is a nap, not a death.

Saves core weights and the slow self, the drives and their discovered
setpoints, installed affects and their signatures, the imprint anchors,
the bus state and attractors, the memory pool, the ladder's concepts, and
felt time. Rotates latest→prev. Called only from the Loom thread
(it owns the GPU) via `checkpoint_fn`.
"""

import time
from pathlib import Path
from typing import Optional

import torch

from primordium.persistence import calib


def save_checkpoint(ctx, tag: Optional[str] = None) -> Path:
    w = ctx.worker
    run_dir = ctx.cfg.run_dir()
    name = f"ckpt_{tag}.pt" if tag else "ckpt_latest.pt"
    path = run_dir / name

    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}  # noqa: E731
    data = {
        "version": 5,
        "saved_at": time.time(),
        "tick_id": w.tick_id,
        "stage": w.stage,
        "anatomy": w.model.anatomy(),
        "bloom": (w.bloom.state_dict() if getattr(w, "bloom", None) else {}),
        "model": cpu(w.model.state_dict()),
        "tokenizer": cpu(w.tokenizer.state_dict()),
        "voice_head": cpu(w.voice_head.state_dict()),
        "keys_head": cpu(w.keys_head.state_dict()),
        "paint_head": cpu(w.paint_head.state_dict()),
        "scaffold": cpu(w.scaffold.state_dict()),
        "fringe": {"params": cpu(w.fringe.state_dict()),
                   "stats": w.fringe.stats_state()},
        "reach": {"params": cpu(w.reach.state_dict()),
                  "bank": w.reach.bank_state()},
        "gaze": (w.gaze.state_dict() if getattr(w, "gaze", None) else {}),
        "watch": w.watch.state(),
        "wheel": {"params": cpu(w.wheel.state_dict()),
                  "bank": w.wheel.bank_state()},
        "grip": cpu(w.grip.state_dict()),
        "anneal": {"L_birth": w._L_birth,
                   "ema_fast": w.objective.ema_fast,
                   "ema_slow": w.objective.ema_slow},
        "decoder": cpu(w.decoder.state_dict()),
        "audio_decoder": cpu(w.audio_decoder.state_dict()),
        "w_slow": {k: v.detach().cpu() for k, v in w.w_slow.items()},
        "t_vision": cpu(w.objective.t_vision.state_dict()),
        "t_audio": cpu(w.objective.t_audio.state_dict()),
        "t_text": cpu(w.objective.t_text.state_dict()),
        "t_canvas": cpu(w.objective.t_canvas.state_dict()),
        "wordstream": ctx.wordstream.state_dict(),
        "easel": ctx.easel.state_dict(),
        "gates": (ctx.gatehouse.state_dict()
                  if getattr(ctx, "gatehouse", None) else {}),
        "drives": ctx.drives.state_dict(),
        "sleep": ctx.sleep.state_dict(),
        "compass": ctx.compass.state_dict(),
        "imprint": ctx.imprint.state_dict(),
        "tide": ctx.neuromod.state(),
        "bus": {
            "state": ctx.bus.state.detach().cpu(),
            "velocity": ctx.bus.velocity.detach().cpu(),
            "attractors": [(v.detach().cpu(), float(wt))
                           for v, wt in ctx.bus.attractors],
            "tick_count": ctx.bus.tick_count,
        },
        "self_model": {
            "identity": (ctx.self_model.identity.detach().cpu()
                         if ctx.self_model.identity is not None else None),
            "felt_time": ctx.self_model.felt_time,
        },
        "situation_vec": (ctx.situation.vec.detach().cpu()
                          if ctx.situation.vec is not None else None),
        "memory_pool": [
            {"vec": c.vec.detach().cpu(), "confidence": c.confidence,
             "meta": dict(c.meta or {})}
            for c in list(ctx.memory.pool)[-200:]
        ],
        "abstraction": {
            "observations": ctx.abstraction.observations,
            "levels": [
                {"centroids": [c.detach().cpu() for c in lvl.centroids],
                 "counts": list(lvl.counts), "updates": lvl.updates}
                for lvl in ctx.abstraction.levels
            ],
        },
    }
    rec = calib.record()
    if rec is not None:
        data["calib"] = rec

    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    if not tag:
        prev = run_dir / "ckpt_prev.pt"
        if path.exists():
            if prev.exists():
                prev.unlink()
            path.rename(prev)
    if path.exists():
        path.unlink()
    tmp.rename(path)
    return path


def load_checkpoint(ctx, path: Path) -> None:
    from primordium.loom.tokenizer import SensoryTokenizer

    data = torch.load(path, map_location="cpu", weights_only=False)
    ver = int(data.get("version", 1))
    if ver < 5:
        era = {1: "v1", 2: "v2", 3: "pre-Gaze v3",
               4: "pre-Tide v3"}.get(ver, "an earlier")
        raise RuntimeError(
            f"This is a {era} life. This body's organs have changed "
            "(the Tide replaced the hormone layer; interoception is "
            "smaller) and it requires a new birth — see "
            "primordium/README.md.")
    w = ctx.worker
    dev = w.device

    w.stage = int(data["stage"])
    w.tick_id = int(data["tick_id"])
    w.tokenizer = SensoryTokenizer(ctx.cfg, stage=w.stage).to(dev)
    w.tokenizer.load_state_dict(data["tokenizer"])
    # the body must be regrown to its saved anatomy before the weights fit
    from primordium.loom.core import LoomCore
    from primordium.loom.fringe import Fringe
    anatomy = data.get("anatomy")
    if anatomy and anatomy["mlp_dims"] != w.model.anatomy()["mlp_dims"]:
        w.fringe.detach()
        w.model = LoomCore(ctx.cfg, anatomy).to(dev)
        w.fringe = Fringe(ctx.cfg, w.model).to(dev)
    w.model.load_state_dict(data["model"])
    if getattr(w, "bloom", None) and data.get("bloom"):
        w.bloom.load_state_dict(data["bloom"])
        w.bloom.rebind(w.model)
    w.voice_head.load_state_dict(data["voice_head"])
    w.keys_head.load_state_dict(data["keys_head"])
    w.paint_head.load_state_dict(data["paint_head"])
    w.scaffold.load_state_dict(data["scaffold"])
    fr = data.get("fringe")
    if fr and len(w.fringe):
        w.fringe.load_state_dict(fr["params"])
        w.fringe.load_stats_state(fr.get("stats", []))
    rc = data.get("reach")
    if rc:
        w.reach.load_state_dict(rc["params"])
        w.reach.load_bank_state(rc.get("bank", {}))
    if getattr(w, "gaze", None) and data.get("gaze"):
        w.gaze.load_state_dict(data["gaze"])
    if data.get("watch"):
        w.watch.load_state(data["watch"])
    wh = data.get("wheel")
    if wh:
        w.wheel.load_state_dict(wh["params"])
        w.wheel.load_bank_state(wh.get("bank", {}))
    gp = data.get("grip")      # guarded: absent in pre-Grip v5 lives
    if gp:
        w.grip.load_state_dict(gp)
    ann = data.get("anneal", {})
    w._L_birth = ann.get("L_birth")
    w.objective.ema_fast = ann.get("ema_fast")
    w.objective.ema_slow = ann.get("ema_slow")
    w.decoder.build(w.stage)
    w.decoder.load_state_dict(data["decoder"])
    w.decoder.to(dev)
    w.audio_decoder.load_state_dict(data["audio_decoder"])
    w.audio_decoder.to(dev)
    w.objective.on_stage_grown(w.tokenizer)
    for key, mod in (("t_vision", w.objective.t_vision),
                     ("t_audio", w.objective.t_audio),
                     ("t_text", w.objective.t_text),
                     ("t_canvas", w.objective.t_canvas)):
        mod.load_state_dict(data[key])
        mod.to(dev)
    w.w_slow = {k: v.to(dev) for k, v in data["w_slow"].items()}
    w._build_optimizers()
    w.ring.clear()

    ctx.wordstream.load_state_dict(data.get("wordstream", {}))
    ctx.easel.load_state_dict(data.get("easel", {}))
    if getattr(ctx, "gatehouse", None) and data.get("gates"):
        ctx.gatehouse.load_state_dict(data["gates"])

    ctx.drives.load_state_dict(data.get("drives", {}))
    ctx.sleep.load_state_dict(data.get("sleep", {}))
    ctx.compass.load_state_dict(data.get("compass", {}))
    ctx.imprint.load_state_dict(data.get("imprint", {}))

    ctx.neuromod.load_state(data.get("tide", {}))

    b = data.get("bus", {})
    if "state" in b and b["state"].numel() == ctx.bus.state.numel():
        with ctx.bus._lock:  # noqa: SLF001 — restore is a privileged act
            ctx.bus.state = b["state"].to(ctx.bus.device)
            ctx.bus.velocity = b["velocity"].to(ctx.bus.device)
            ctx.bus.attractors = [
                (v.to(ctx.bus.device), wt) for v, wt in b.get("attractors", [])
            ] or ctx.bus.attractors
            ctx.bus.tick_count = int(b.get("tick_count", 0))

    sm = data.get("self_model", {})
    if sm.get("identity") is not None:
        ctx.self_model.identity = sm["identity"]
    ctx.self_model.felt_time = float(sm.get("felt_time", 0.0))

    if data.get("situation_vec") is not None:
        ctx.situation.vec = data["situation_vec"]

    from magnum_opus_v2.regions.subconscious import Candidate
    ctx.memory.pool.clear()
    for m in data.get("memory_pool", []):
        ctx.memory.pool.append(Candidate(
            vec=m["vec"], source="memory",
            confidence=float(m.get("confidence", 1.0)),
            meta=dict(m.get("meta", {}))))

    ab = data.get("abstraction")
    if ab:
        ctx.abstraction.observations = int(ab.get("observations", 0))
        for lvl, st in zip(ctx.abstraction.levels, ab.get("levels", [])):
            lvl.centroids = [c for c in st.get("centroids", [])]
            lvl.counts = list(st.get("counts", []))
            lvl.updates = int(st.get("updates", 0))

    # re-install discovered affects into the plastic regions
    vecs = ctx.compass.vectors()
    if vecs:
        from magnum_opus_v2._dynamics import EmotionConfig
        configs = {n: EmotionConfig(onset_rate=0.4, decay_rate=0.05,
                                    baseline=0.05 if n.startswith("drive_") else 0.0)
                   for n in vecs}
        ctx.mood.set_vectors(vecs, configs=configs)
        try:
            ctx.subc.set_affect_vectors(vecs)
        except Exception:  # noqa: BLE001
            pass
