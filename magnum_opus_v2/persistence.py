"""
Engine persistence — a restart is a nap, not a death.

The engine used to be reborn empty each launch: bus, emotions, memory,
abstraction, self-model, situation, and the forecast ledger's earned
calibration all vanished on exit. This module saves and restores that
runtime state so a woken engine continues the same continuous mind.

Two design rules make the nap honest:

1. SHARED DATA CONTRACT. The per-block codecs write shapes byte-compatible
   with the organism's checkpoint (primordium/persistence/checkpoint.py):
   the bus block, the memory pool (a list of `Candidate {vec, confidence,
   meta}`), the abstraction ladder, the self-model, and the situation all
   share the same on-disk shape. The two minds stay separate processes
   with separate live state, but speak one persistence format.

2. MONOTONIC-CLOCK REBASING. Every `*_wall` field is a `time.monotonic()`
   reading, meaningless across a process restart. A naive restore would
   put forecast deadlines and speech cooldowns in the past and make the
   engine blurt on wake. We store the AGE of each such field at save time
   and rebase it on load: wall-start fields reset to now (felt-time itself
   persists), cooldown/freshness fields become `now - min(age, cap)`, and
   forecast deadlines are rebuilt from remaining seconds by the ledger.

A checkpoint refuses to load into a different body (hidden_dim / model
name mismatch) — you cannot pour a gpt2 nap into a Qwen.
"""

import time
from pathlib import Path
from typing import Optional

import torch

from magnum_opus_v2.regions.subconscious import Candidate
from magnum_opus_v2.model_sources import model_storage_key

ENGINE_VERSION = 2
STATE_DIR = Path(__file__).parent.parent / "state"

# cap for rebased cooldown/freshness ages: an ancient nap should wake
# "rested" (fully cooled down), never with a negative or runaway clock.
_AGE_CAP_S = 3600.0


def _sanitize(model_name: str) -> str:
    return model_storage_key(model_name)


def default_path(model_name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"engine_{_sanitize(model_name)}.pt"


def _cpu(t: Optional[torch.Tensor]):
    return t.detach().float().cpu() if t is not None else None


# ----------------------------------------------------------------------
# per-block codecs (shapes shared with the organism checkpoint)
# ----------------------------------------------------------------------
def encode_bus(bus) -> dict:
    with bus._lock:  # noqa: SLF001 — snapshot the substrate atomically
        return {
            "state": _cpu(bus.state),
            "velocity": _cpu(bus.velocity),
            "attractors": [(_cpu(v), float(w)) for v, w in bus.attractors],
            "tick_count": int(bus.tick_count),
            "temperature": float(bus.temperature),
        }


def apply_bus(bus, blk: dict) -> None:
    if not blk:
        return
    with bus._lock:  # noqa: SLF001
        bus.state = blk["state"].to(bus.device)
        bus.velocity = blk["velocity"].to(bus.device)
        atts = [(v.to(bus.device), w) for v, w in blk.get("attractors", [])]
        if atts:
            bus.attractors = atts
        bus.tick_count = int(blk.get("tick_count", 0))
        bus.temperature = float(blk.get("temperature", bus.temperature))
        # wall clocks are rebased by the caller (see load_engine)


def encode_neuromod(nm) -> dict:
    return {
        "stress": nm.stress, "reward": nm.reward,
        "calm": nm.calm, "arousal": nm.arousal,
        "stress_baseline": nm.stress_baseline,
        "reward_baseline": nm.reward_baseline,
        "calm_baseline": nm.calm_baseline,
        "arousal_baseline": nm.arousal_baseline,
        "drift_speed": nm.drift_speed,
    }


def apply_neuromod(nm, blk: dict) -> None:
    if not blk:
        return
    for k, v in blk.items():
        if hasattr(nm, k):
            setattr(nm, k, float(v))
    # NeuromodulatorRegion._calm_streak is transient (re-accrues from
    # measured stillness within ~1 min) and deliberately not persisted.


def encode_limbic(limbic) -> Optional[dict]:
    st = getattr(limbic, "_state", None)
    if st is None:
        return None
    return {"names": list(st.names),
            "fast": dict(st.fast), "medium": dict(st.medium),
            "slow": dict(st.slow)}


def apply_limbic(limbic, blk: dict) -> None:
    st = getattr(limbic, "_state", None)
    if not blk or st is None:
        return
    # fitted dynamics come from the profile; only channel LEVELS persist,
    # and only for emotions this body actually has.
    for speed in ("fast", "medium", "slow"):
        vals = blk.get(speed, {})
        target = getattr(st, speed)
        for n in target:
            if n in vals:
                target[n] = float(vals[n])


def encode_memory_pool(mem, limit: int = 200) -> list:
    with mem._lock:  # noqa: SLF001
        return [{"vec": _cpu(c.vec), "confidence": float(c.confidence),
                 "source": c.source, "meta": dict(c.meta or {})}
                for c in list(mem.pool)[-limit:]]


def apply_memory_pool(mem, items: list) -> None:
    if not items:
        return
    with mem._lock:  # noqa: SLF001
        mem.pool.clear()
        for m in items:
            if not isinstance(m.get("vec"), torch.Tensor):
                continue
            mem.pool.append(Candidate(
                vec=m["vec"].to(mem.device),
                source=m.get("source", "memory"),
                confidence=float(m.get("confidence", 1.0)),
                meta=dict(m.get("meta", {}))))


def encode_abstraction(ab) -> Optional[dict]:
    if ab is None:
        return None
    return {
        "observations": int(ab.observations),
        "novelty": float(getattr(ab, "novelty", 0.0)),
        "levels": [{"centroids": [_cpu(c) for c in lvl.centroids],
                    "counts": list(lvl.counts),
                    "updates": int(lvl.updates)} for lvl in ab.levels],
    }


def apply_abstraction(ab, blk: dict) -> None:
    if ab is None or not blk:
        return
    ab.observations = int(blk.get("observations", 0))
    ab.novelty = float(blk.get("novelty", 0.0))
    for lvl, st in zip(ab.levels, blk.get("levels", [])):
        lvl.centroids = [c.to(ab.device) for c in st.get("centroids", [])]
        lvl.counts = list(st.get("counts", []))
        lvl.updates = int(st.get("updates", 0))


def encode_self_model(sm) -> Optional[dict]:
    if sm is None:
        return None
    return {"identity": _cpu(sm.identity),
            "felt_time": float(sm.felt_time),
            "continuity": float(sm.continuity)}


def apply_self_model(sm, blk: dict) -> None:
    if sm is None or not blk:
        return
    ident = blk.get("identity")
    sm.identity = ident.to(sm.device) if isinstance(ident, torch.Tensor) else None
    sm.felt_time = float(blk.get("felt_time", 0.0))
    sm.continuity = float(blk.get("continuity", 1.0))


def encode_situation(sit) -> Optional[dict]:
    if sit is None:
        return None
    return {"vec": _cpu(sit.vec), "narrative": sit.narrative,
            "last_sim": float(sit.last_sim),
            "shift_count": int(sit.shift_count)}


def apply_situation(sit, blk: dict) -> None:
    if sit is None or not blk:
        return
    vec = blk.get("vec")
    sit.vec = vec.to(sit.device) if isinstance(vec, torch.Tensor) else None
    sit.narrative = blk.get("narrative")
    sit.last_sim = float(blk.get("last_sim", 1.0))
    sit.shift_count = int(blk.get("shift_count", 0))


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------
def save_engine(engine, path: Optional[Path] = None) -> Optional[Path]:
    """Snapshot runtime state under each region's own lock (copying to
    CPU), then torch.save OUTSIDE all locks so a save never stalls the
    50ms flow clock. Atomic tmp+rename, rotate latest->prev."""
    model_name = engine.profile.model_name if engine.profile else "unknown"
    path = Path(path) if path else default_path(model_name)
    now = time.monotonic()

    # --- gather under locks (each codec takes the region's own lock) ---
    data = {
        "engine_version": ENGINE_VERSION,
        "saved_at": time.time(),
        "model_name": model_name,
        "profile_signature": engine.profile.signature() if engine.profile else None,
        "hidden_dim": int(engine.bus.hidden_dim),
        "bus": encode_bus(engine.bus),
        "neuromod": encode_neuromod(engine.neuromod),
        "limbic": encode_limbic(engine.limbic),
        "memory_pool": encode_memory_pool(engine.memory),
        "abstraction": encode_abstraction(engine.abstraction),
        "self_model": encode_self_model(engine.self_model),
        "situation": encode_situation(engine.situation),
        "executive": {"pressure": float(engine.executive.pressure)},
        "alignment_gate": engine.alignment_gate.state_dict(),
        "rumination_steps": int(engine.rumination_steps),
    }
    with engine._history_lock:  # noqa: SLF001
        data["chat_history"] = [dict(m) for m in engine.chat_history]
    if engine.speculative is not None:
        data["forecast"] = engine.speculative.ledger.state_dict()

    # AGE of each monotonic wall field (rebased on load)
    ages = {}
    sm = engine.self_model
    if sm is not None:
        ages["self_model_started"] = now - sm._wall_started  # noqa: SLF001
    ages["memory_last_capture"] = (
        (now - engine.memory._last_capture_wall)  # noqa: SLF001
        if engine.memory._last_capture_wall > 0 else None)  # noqa: SLF001
    ex = engine.executive
    ages["exec_interaction"] = now - ex._wall_at_interaction  # noqa: SLF001
    ages["exec_speech"] = (
        (now - ex._wall_at_speech) if ex._wall_at_speech > 0 else None)  # noqa: SLF001
    if engine.situation is not None:
        ages["situation_updated"] = (
            (now - engine.situation._updated_wall)  # noqa: SLF001
            if engine.situation._updated_wall > 0 else None)
    ages["temporal_interaction"] = (
        now - engine.temporal._wall_at_interaction)  # noqa: SLF001
    data["wall_ages"] = ages

    # --- write OUTSIDE all locks ---
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        if path.exists():
            prev = path.with_suffix(".prev")
            if prev.exists():
                prev.unlink()
            path.replace(prev)
        tmp.replace(path)
        return path
    except Exception:  # noqa: BLE001 — a failed save must never crash stop()
        return None


def _rebase_monotonic(age: Optional[float], now: float,
                      reset: bool = False) -> float:
    """Turn a saved AGE back into a monotonic timestamp. reset=True pins
    to now (wall-start clocks); otherwise `now - min(age, cap)` so a long
    nap wakes fully cooled-down, never with a future or runaway clock."""
    if age is None:
        return 0.0
    if reset:
        return now
    return now - min(max(age, 0.0), _AGE_CAP_S)


def load_engine(engine, path: Optional[Path] = None) -> bool:
    """Load a saved nap into a BUILT engine (call before start()). Returns
    True on success. Refuses a body mismatch."""
    model_name = engine.profile.model_name if engine.profile else "unknown"
    path = Path(path) if path else default_path(model_name)
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("engine_version") != ENGINE_VERSION:
        raise RuntimeError(
            "Checkpoint predates the block-output activation convention or uses "
            "an unsupported version. Keep it as an archive and start fresh state.")

    if engine.profile and data.get("profile_signature") != engine.profile.signature():
        raise RuntimeError(
            "Checkpoint calibration differs from the selected profile, or predates "
            "calibration identity checks. Keep it as an archive and start fresh state.")

    if int(data.get("hidden_dim", -1)) != int(engine.bus.hidden_dim):
        raise RuntimeError(
            f"engine checkpoint is a different body (hidden_dim "
            f"{data.get('hidden_dim')} vs {engine.bus.hidden_dim}); "
            "it needs a fresh start.")
    saved_model = data.get("model_name", "unknown")
    if saved_model != model_name:
        raise RuntimeError(
            f"engine checkpoint was saved for '{saved_model}', not "
            f"'{model_name}'. Steering vectors are per-model; start fresh.")

    apply_bus(engine.bus, data.get("bus", {}))
    apply_neuromod(engine.neuromod, data.get("neuromod", {}))
    apply_limbic(engine.limbic, data.get("limbic", {}))
    apply_memory_pool(engine.memory, data.get("memory_pool", []))
    apply_abstraction(engine.abstraction, data.get("abstraction", {}))
    apply_self_model(engine.self_model, data.get("self_model", {}))
    apply_situation(engine.situation, data.get("situation", {}))
    engine.executive.pressure = float(
        data.get("executive", {}).get("pressure", 0.0))
    engine.alignment_gate.load_state_dict(data.get("alignment_gate"))
    with engine._history_lock:  # noqa: SLF001
        engine.chat_history = [dict(m) for m in data.get("chat_history", [])]
    if engine.speculative is not None and data.get("forecast"):
        engine.speculative.ledger.load_state_dict(data["forecast"])

    # --- rebase every monotonic wall clock ---
    now = time.monotonic()
    ages = data.get("wall_ages", {})
    engine.bus.wall_time_started = now       # dilation restarts cleanly
    engine.bus.last_flow_time = now
    if engine.self_model is not None:
        engine.self_model._wall_started = now  # noqa: SLF001 (felt_time persists)
    engine.memory._last_capture_wall = _rebase_monotonic(  # noqa: SLF001
        ages.get("memory_last_capture"), now)
    engine.executive._wall_at_interaction = _rebase_monotonic(  # noqa: SLF001
        ages.get("exec_interaction"), now)
    engine.executive._wall_at_speech = _rebase_monotonic(  # noqa: SLF001
        ages.get("exec_speech"), now)
    if engine.situation is not None:
        engine.situation._updated_wall = _rebase_monotonic(  # noqa: SLF001
            ages.get("situation_updated"), now)
    engine.temporal._wall_at_interaction = _rebase_monotonic(  # noqa: SLF001
        ages.get("temporal_interaction"), now)
    return True
