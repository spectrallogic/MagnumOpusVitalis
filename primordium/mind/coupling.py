"""
Coupling — the bridges between the loom, body, and substrate.

Every class here is a magnum_opus_v2 Region running on the engine's
clocks. None of them touch the GPU: they exchange small tensors and
numbers with the Loom through thread-safe slots, and they speak
to the bus in its own language — perturbations.
"""

import threading
import time
from typing import Optional

import numpy as np
import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.region import Region


class Blackboard:
    """Tiny shared scratchpad between regions (thread-safe)."""

    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def put(self, **kv) -> None:
        with self._lock:
            self._d.update(kv)

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)

    def all(self) -> dict:
        with self._lock:
            return dict(self._d)


# ---------------------------------------------------------------------------
class LoomBridge(Region):
    """Percepts out of the loom, steering into it, thought-feedback into
    the bus, kinship against the imprint anchors."""

    name = "loom_bridge"
    clock = "perception"

    def __init__(self, cfg, worker, situation, abstraction, memory,
                 imprint, board):
        self.cfg = cfg
        self.worker = worker
        self.situation = situation
        self.abstraction = abstraction
        self.memory = memory
        self.imprint = imprint
        self.board = board
        self._prev_z: Optional[torch.Tensor] = None
        self.kinship = 0.0

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        # the bus whispers into the loom
        self.worker.set_steer(bus.state.detach().clone())

        pub = self.worker.get_published()
        z = pub.get("z")
        if z is None:
            return None
        z = torch.as_tensor(z, dtype=torch.float32)

        # world-content fanout — same organs the engine feeds
        try:
            if self.situation is not None:
                self.situation.observe_percept(z, neuromod)
            if self.abstraction is not None:
                self.abstraction.observe_external(z, neuromod)
            surprise = float(pub.get("surprise", 1.0))
            if surprise > self.cfg.percept_surprise_capture:
                self.memory.capture_experience(
                    z, importance=min(2.0, surprise),
                    tag=f"percept:t{pub.get('tick', 0)}")
            if surprise > 2.0 and hasattr(neuromod, "bump"):
                neuromod.raise_("arousal", 0.05 * (surprise - 2.0),
                                cause=f"surprise {surprise:.1f}")
        except Exception:  # noqa: BLE001 — never crash the substrate
            pass

        # dz for the compass; kinship against the caregiver anchors
        if self._prev_z is not None:
            self.board.put(dz=(z - self._prev_z).numpy())
        self._prev_z = z
        self.kinship = self.imprint.kinship(z)
        self.board.put(kinship=self.kinship, z_age=time.monotonic())

        # the imprint ritual: while held, this moment becomes an anchor
        if self.board.get("imprint_hold", False):
            self.imprint.imprint(z, weight=1.0)
            self.board.put(imprint_count=self.imprint.count())

        # thought moves feeling — the loom tugs the bus
        delta = z.to(bus.device) - bus.state
        n = float(delta.norm())
        if n > 4.0:
            delta = delta * (4.0 / n)
        return delta * self.cfg.thought_feedback_gain


# ---------------------------------------------------------------------------
class DriveRegion(Region):
    """The body updating its needs, and the needs updating the chemistry."""

    name = "drives"
    clock = "perception"

    def __init__(self, cfg, drives, worker, compass, executive, sleep, board):
        self.cfg = cfg
        self.drives = drives
        self.worker = worker
        self.compass = compass
        self.executive = executive
        self.sleep = sleep
        self.board = board
        self._prev_errors: Optional[np.ndarray] = None
        self._prev_speech = 0.0

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        pub = self.worker.get_published()
        presence = self.board.get("presence", {}) or {}
        kinship = float(self.board.get("kinship", 0.0))

        presence_score = min(1.0, (0.6 if presence.get("face") else 0.0)
                             + presence.get("speechiness", 0.0)
                             + 0.3 * kinship
                             + (0.3 if time.monotonic() < float(
                                 self.board.get("foster_contact_until", 0.0))
                                else 0.0))
        novelty = float(pub.get("novelty", 0.0))
        vitality = float(pub.get("vitality", 0.0))
        lp = float(pub.get("lp", 0.0))

        self.drives.update(lp=lp, novelty=novelty,
                           presence_score=presence_score,
                           vitality=vitality,
                           asleep=self.sleep.asleep, neuromod=neuromod)

        # directed speech = interaction (the engine's social heartbeat)
        sp = presence.get("speechiness", 0.0)
        if sp > 0.4 and self._prev_speech <= 0.4:
            self.executive.mark_interaction()
        self._prev_speech = sp

        # the compass watches what latent motion does to the needs
        errs = np.array(list(self.drives.errors().values()), dtype=np.float32)
        dz = self.board.get("dz")
        if dz is not None and self._prev_errors is not None:
            self.compass.observe(
                dz, errs - self._prev_errors,
                float(pub.get("surprise", 1.0)),
                float(self.drives.last_reward))
        self._prev_errors = errs
        return None


# ---------------------------------------------------------------------------
class AffectProjector(Region):
    """Discovered directions, felt: latent motion along an installed affect
    stimulates the MoodField, which then lives its fitted
    dynamics on the bus."""

    name = "affect_projector"
    clock = "perception"

    def __init__(self, compass, mood_field, board):
        self.compass = compass
        self.mood = mood_field
        self.board = board
        self._stats = {}     # name -> (mean, var, n) running

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        dz = self.board.get("dz")
        if dz is None:
            return None
        vecs = self.compass.vectors()
        if not vecs:
            return None
        x = torch.from_numpy(dz).float()
        acts = {}
        for name, v in vecs.items():
            p = float(torch.dot(x, v))
            self.compass.observe_activation(name, p)   # lived dynamics data
            m, var, n = self._stats.get(name, (0.0, 1.0, 1))
            n = min(n + 1, 5000)
            m += (p - m) / n
            var += ((p - m) ** 2 - var) / n
            self._stats[name] = (m, var, n)
            zscore = (p - m) / ((var ** 0.5) + 1e-6)
            acts[name] = zscore
            if abs(zscore) > 2.0 and n > 100:
                try:
                    self.mood.stimulate(name, min(1.0, abs(zscore) / 4.0),
                                          neuromod=neuromod)
                except Exception:  # noqa: BLE001
                    pass
        self.board.put(affect_acts={k: round(v, 3) for k, v in acts.items()})
        return None


# ---------------------------------------------------------------------------
class PresenceRegion(Region):
    name = "presence"
    clock = "expensive"

    def __init__(self, sensor, retina, cochlea, voicebox, board):
        self.sensor = sensor
        self.retina = retina
        self.cochlea = cochlea
        self.voice = voicebox
        self.board = board

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        frame, _ = self.retina.latest()
        pcm = self.cochlea.take_latest(16000)
        self.voice.hear(self.cochlea.rms())
        p = self.sensor.measure(frame, pcm, self_heard=self.voice.self_heard)
        self.board.put(presence=p)
        return None


# ---------------------------------------------------------------------------
class ReplayScheduler(Region):
    name = "replay_scheduler"
    clock = "expensive"

    def __init__(self, cfg, worker, sleep):
        self.cfg = cfg
        self.worker = worker
        self.sleep = sleep
        self._last = 0.0
        self._decode_last = 0.0

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        now = time.monotonic()
        every = self.cfg.replay_every_s / max(1, self.sleep.replay_multiplier())
        if now - self._last >= every:
            self._last = now
            self.worker.enqueue({"kind": "replay"})
        if now - self._decode_last >= 1.0:
            self._decode_last = now
            self.worker.enqueue({"kind": "decode"})
        return None


# ---------------------------------------------------------------------------
class ReverieRegion(Region):
    name = "reverie"
    clock = "expensive"

    def __init__(self, cfg, worker, reverie, sleep, board):
        self.cfg = cfg
        self.worker = worker
        self.imag = reverie
        self.sleep = sleep
        self.board = board

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        pub = self.worker.get_published()
        presence = (self.board.get("presence", {}) or {})
        pres = (0.6 if presence.get("face") else 0.0) + presence.get("speechiness", 0.0)
        if self.imag.due(self.sleep.asleep, float(pub.get("novelty", 1.0)), pres):
            self.worker.enqueue({"kind": "dream", "asleep": self.sleep.asleep})
        d = self.imag.take_delta()
        if d is not None:
            v = d.to(bus.device)
            n = float(v.norm())
            if n > 1e-6:
                bus.add_perturbation((v / n) * 0.05,
                                     source="reverie")  # the mind wanders
        return None


# ---------------------------------------------------------------------------
class CompassRegion(Region):
    """Single owner of affect installation: lived directions from the
    compass's own estimation, innate needles from the Instincts (at
    reduced weight and gentler dynamics), and the retirement that ends
    infancy — priors removed at stage 2 or once enough lived affects
    exist to navigate by."""

    name = "compass"
    clock = "slow"

    def __init__(self, compass, mood_field, plastic_subc, bus, board,
                 cfg=None, worker=None, pulse=None):
        self.compass = compass
        self.mood = mood_field
        self.subc = plastic_subc
        self.bus = bus
        self.board = board
        self.cfg = cfg
        self.worker = worker
        self.pulse = pulse

    def _maybe_retire(self) -> bool:
        if self.cfg is None or self.worker is None:
            return False
        prov = self.compass.snapshot().get("installed", [])
        if not any(a.get("provenance") == "innate" for a in prov):
            return False
        stage = int(getattr(self.worker, "stage", 0))
        lived = self.compass.lived_count()
        if stage < 2 and lived < self.cfg.instinct_retire_lived:
            return False
        gone = self.compass.retire_innate()
        if gone:
            self.board.put(instincts_retired=True)
            if self.pulse is not None:
                self.pulse.emit("instincts_retired", "SCAFFOLD", "SELF",
                                retired=gone, stage=stage, lived=lived)
        return bool(gone)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        from magnum_opus_v2._dynamics import EmotionConfig
        dirty = self._maybe_retire()
        dirty = bool(self.board.get("affects_dirty", False)) or dirty
        cands = self.compass.estimate()
        newly = self.compass.stabilize_and_install(cands) if cands else []
        vecs = self.compass.vectors()
        if vecs and (newly or dirty):
            innate_w = self.cfg.instinct_weight if self.cfg else 0.3
            prov = {a["id"]: a.get("provenance", "lived")
                    for a in self.compass.snapshot().get("installed", [])}
            configs, scaled, lived_only = {}, {}, {}
            for n, v in vecs.items():
                # dynamics are FITTED from the affect's own activation
                # history when enough of it exists; until then, the
                # disclosed defaults below, flagged as defaults
                fd = self.compass.fitted_dynamics(n, dt=0.2)
                if fd is not None:
                    onset, decay, meta = fd
                    self.compass.signatures.setdefault(n, {})["dynamics"] = {
                        "mode": "fitted", "onset": round(onset, 4),
                        "decay": round(decay, 4), **meta}
                else:
                    onset = 0.15 if prov.get(n) == "innate" else 0.4
                    decay = 0.05
                    self.compass.signatures.setdefault(n, {})["dynamics"] = {
                        "mode": "default", "onset": onset, "decay": decay}
                if prov.get(n) == "innate":
                    configs[n] = EmotionConfig(onset_rate=onset,
                                               decay_rate=decay, baseline=0.0)
                    scaled[n] = v * innate_w
                else:
                    configs[n] = EmotionConfig(
                        onset_rate=onset, decay_rate=decay,
                        baseline=0.05 if n.startswith("drive_") else 0.0)
                    scaled[n] = v
                    lived_only[n] = v
            self.mood.set_vectors(scaled, configs=configs)
            try:
                # the Undertow imagines with lived directions only —
                # inherited leanings color mood, not imagination scoring
                self.subc.set_affect_vectors(lived_only)
            except Exception:  # noqa: BLE001
                pass
            bl = self.mood.baseline_vector()
            if bl.numel() == bus.state.numel():
                bus.set_baseline(bl)
            self.board.put(affects_installed=list(vecs.keys()),
                           affect_event=newly, affects_dirty=False)
        return None


# ---------------------------------------------------------------------------
class LodestarRegion(Region):
    """The frozen teacher's only doorway into the organism. On the
    expensive clock it (a) hands the Loom one feature of the frame the
    organism just saw, for the annealed alignment term, and (b) refreshes
    the Instinct needles as the scaffold's P projection trains. At stage
    2 it unloads the teacher entirely — measurably gone."""

    name = "lodestar"
    clock = "expensive"

    def __init__(self, cfg, worker, teacher, retina, sleep, compass,
                 board, pulse=None):
        self.cfg = cfg
        self.worker = worker
        self.teacher = teacher
        self.retina = retina
        self.sleep = sleep
        self.compass = compass
        self.board = board
        self.pulse = pulse
        self._last_feed = 0.0
        self._last_needles = 0.0
        self._probe_feats = None      # {name: (n,1024)} computed once
        self._released = False

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self.teacher.unload()
            if self.pulse is not None:
                self.pulse.emit("lodestar_released", "SCAFFOLD", "KEEP",
                                reason="stage>=2: annealed grounding over")

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        if self._released or not self.teacher.ok:
            return None
        if int(getattr(self.worker, "stage", 0)) >= 2:
            self._release()
            return None
        now = time.monotonic()

        # (a) the distill feature — same frame, one vector, nothing else
        if not self.sleep.asleep and now - self._last_feed >= self.cfg.distill_every_s:
            frame, _ts = self.retina.latest()
            if frame is not None:
                feat = self.teacher.embed_frame(frame)
                if feat is not None:
                    self._last_feed = now
                    self.worker.set_teacher_feat(feat)

        # (b) instinct needles, recomputed as P trains
        if self.board.get("instincts_retired", False):
            return None
        if now - self._last_needles >= self.cfg.needle_every_s:
            self._last_needles = now
            if self._probe_feats is None:
                from primordium.teachers.instincts import all_probe_frames
                frames = all_probe_frames(self.cfg.source_res,
                                          self.cfg.probe_jitters)
                feats = {}
                for name, batch in frames.items():
                    f = self.teacher.embed_batch(batch)
                    if f is None:
                        return None
                    feats[name] = f.mean(0)          # (1024,) per instinct
                self._probe_feats = feats
            names = list(self._probe_feats.keys())
            stacked = torch.stack([self._probe_feats[n] for n in names])
            grand = stacked.mean(0)
            self.worker.enqueue({
                "kind": "needles", "feats": stacked, "mean": grand,
                "cb": lambda needles, names=names: self._on_needles(
                    names, needles)})
        return None

    def _on_needles(self, names, needles: torch.Tensor) -> None:
        if self.board.get("instincts_retired", False):
            return
        for i, name in enumerate(names):
            self.compass.install_prior(
                name, needles[i],
                sig={"origin": "procedural probe", "probe": name})
        self.board.put(affects_dirty=True)
        if self.pulse is not None:
            self.pulse.emit("needles", "SCAFFOLD", "SELF",
                            installed=[f"innate:{n}" for n in names])


# ---------------------------------------------------------------------------
class GateSenseRegion(Region):
    """Measures the milestones honestly, every slow tick, and reads the
    creator's words from gates.json. Eligibility is automatic and can
    regress; UNLOCKED only ever arrives from outside the organism."""

    name = "gate_sense"
    clock = "slow"

    def __init__(self, cfg, worker, compass, gatehouse, board, pulse=None):
        self.cfg = cfg
        self.worker = worker
        self.compass = compass
        self.gatehouse = gatehouse
        self.board = board
        self.pulse = pulse

    def _emit(self, **kw) -> None:
        if self.pulse is not None:
            self.pulse.emit("gate_state", "KEEP", "SELF", **kw)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        cfg = self.cfg
        stage = int(getattr(self.worker, "stage", 0))
        age = int(getattr(self.worker, "tick_id", 0))
        pub = self.worker.get_published()
        ema = pub.get("ema_slow")
        birth = getattr(self.worker, "_L_birth", None)
        lived = self.compass.lived_count()

        loss_frac = (0.0 if not birth or ema is None
                     else min(1.0, max(0.0, 1.0 - ema / birth)
                              / max(1e-6, 1.0 - cfg.gate_caregiver_loss_frac)))
        loss_met = bool(birth and ema is not None
                        and ema < cfg.gate_caregiver_loss_frac * birth)
        ch = self.gatehouse.set_progress("caregiver", {
            "stage": (min(1.0, stage / max(1, cfg.gate_caregiver_stage)),
                      stage >= cfg.gate_caregiver_stage),
            "learning": (loss_frac, loss_met),
            "affects": (min(1.0, lived / cfg.gate_caregiver_affects),
                        lived >= cfg.gate_caregiver_affects),
            "age": (min(1.0, age / cfg.gate_caregiver_age_ticks),
                    age >= cfg.gate_caregiver_age_ticks),
        })
        if ch:
            self._emit(gate="caregiver", to=ch, measured=True)

        care = self.gatehouse.gates["caregiver"]
        care_days = 0.0
        if care.get("unlocked_at"):
            care_days = (time.time() - care["unlocked_at"]) / 86400.0
        ch = self.gatehouse.set_progress("webtext", {
            "stage": (min(1.0, stage / max(1, cfg.gate_webtext_stage)),
                      stage >= cfg.gate_webtext_stage),
            "caregiver_era": (
                min(1.0, care_days / cfg.gate_webtext_caregiver_days),
                care_days >= cfg.gate_webtext_caregiver_days),
        })
        if ch:
            self._emit(gate="webtext", to=ch, measured=True)

        for change in self.gatehouse.poll_creator_word():
            self._emit(**change, creator_word=True)
        return None


# ---------------------------------------------------------------------------
class FosterRegion(Region):
    """The caregiver, embodied as environment: spawned only while its
    gate stands open AND the run was started with --caregiver; killed
    when either key is withdrawn. Rate limits live here, on the
    organism's side of the process boundary. Death is surfaced as
    honest absence, never papered over."""

    name = "foster"
    clock = "expensive"

    def __init__(self, cfg, wordstream, gatehouse, sleep, executive,
                 board, worker=None, word_teacher=None, pulse=None,
                 foster_factory=None):
        self.cfg = cfg
        self.wordstream = wordstream
        self.gatehouse = gatehouse
        self.sleep = sleep
        self.executive = executive
        self.board = board
        self.worker = worker
        self.word_teacher = word_teacher
        self.pulse = pulse
        self._factory = foster_factory
        self.foster = None
        self._last_sent = 0.0
        self._last_reply = 0.0
        self._last_absent_emit = 0.0

    def _emit(self, kind, **kw) -> None:
        if self.pulse is not None:
            self.pulse.emit(kind, "WORLD", "SENSE", **kw)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        now = time.monotonic()
        gate_open = self.cfg.caregiver and self.gatehouse.is_open("caregiver")

        if gate_open and self.foster is None and self._factory is not None:
            self.foster = self._factory()
            self.foster.start()
            self._emit("caregiver_present",
                       model=self.cfg.caregiver_model,
                       stub=bool(self.cfg.caregiver_stub))
        if not gate_open:
            if self.foster is not None:
                self.foster.stop()
                self.foster = None
                self._emit("caregiver_absent", reason="gate closed")
            return None
        if self.foster is None:
            return None

        if not self.foster.alive:
            if self.foster.maybe_restart():
                self._emit("caregiver_absent", reason="crashed; restarted",
                           restarts=self.foster.restarts)
            elif now - self._last_absent_emit > 60.0:
                self._last_absent_emit = now
                self._emit("caregiver_absent", reason="process down")
            return None

        # its typed babble goes out — at most one utterance per window
        babble = self.wordstream.drain_outbox()
        state = {"awake": not self.sleep.asleep,
                 "stage": int(getattr(self.worker, "stage", 0))}
        if not self.sleep.asleep and self.foster.ready:
            if babble and now - self._last_sent >= self.cfg.caregiver_reply_min_s:
                self._last_sent = now
                self.foster.observe(babble, state)
            elif (now - self._last_reply >= self.cfg.caregiver_checkin_s
                    and now - self._last_sent >= self.cfg.caregiver_checkin_s):
                self._last_sent = now
                self.foster.observe("", state)     # quiet check-in

        reply = self.foster.poll_reply()
        if reply is not None:
            text = str(reply.get("text", ""))[:self.cfg.caregiver_max_chars]
            if text:
                self._last_reply = now
                target = None
                if self.word_teacher is not None and self.word_teacher.ok:
                    target = self.word_teacher.embed_text(text)
                self.wordstream.push(text, source="foster",
                                     target_vec=target)
                self.executive.mark_interaction()
                self.board.put(foster_contact_until=now + 10.0)
                self._emit("caregiver_msg", chars=len(text),
                           stub=bool(reply.get("stub")))
        return None


# ---------------------------------------------------------------------------
class InstinctRegion(Region):
    """Feels along the inherited needles: motion of the organism's own
    q(s) embedding projected onto each innate direction; a spike
    stimulates the corresponding mood. Runs in EMBED space — lived
    affects run in hidden space; the asymmetry is documented, not
    hidden. Goes quiet forever once the compass retires the priors."""

    name = "instincts"
    clock = "perception"

    def __init__(self, cfg, worker, compass, mood_field, board):
        self.cfg = cfg
        self.worker = worker
        self.compass = compass
        self.mood = mood_field
        self.board = board
        self._prev_qs: Optional[np.ndarray] = None
        self._stats = {}              # name -> (mean, var, n)

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        if self.board.get("instincts_retired", False):
            return None
        pub = self.worker.get_published()
        qs = pub.get("qs")
        if qs is None:
            return None
        qs = np.asarray(qs, dtype=np.float32)
        prev, self._prev_qs = self._prev_qs, qs
        if prev is None:
            return None
        innate = {n: v for n, v in self.compass.vectors().items()
                  if n.startswith("innate:")}
        if not innate:
            return None
        dq = torch.from_numpy(qs - prev).float()
        acts = {}
        for name, v in innate.items():
            p = float(torch.dot(dq, v))
            m, var, n = self._stats.get(name, (0.0, 1.0, 1))
            n = min(n + 1, 5000)
            m += (p - m) / n
            var += ((p - m) ** 2 - var) / n
            self._stats[name] = (m, var, n)
            z = (p - m) / ((var ** 0.5) + 1e-6)
            acts[name] = round(z, 3)
            if abs(z) > 2.0 and n > 100:
                try:
                    self.mood.stimulate(
                        name, min(1.0, abs(z) / 4.0), neuromod=neuromod)
                except Exception:  # noqa: BLE001
                    pass
        merged = dict(self.board.get("affect_acts", {}) or {})
        merged.update(acts)
        self.board.put(affect_acts=merged)
        return None


# ---------------------------------------------------------------------------
class SleepRegion(Region):
    name = "sleep"
    clock = "slow"

    def __init__(self, sleep, drives, worker, board):
        self.sleep = sleep
        self.drives = drives
        self.worker = worker
        self.board = board

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        pub = self.worker.get_published()
        presence = (self.board.get("presence", {}) or {})
        pres = (0.6 if presence.get("face") else 0.0) + presence.get("speechiness", 0.0)
        plateau = 1.0 if abs(float(pub.get("lp", 1.0))) < 0.002 else 0.0
        self.sleep.update(
            energy=self.drives.levels["energy"], plateau=plateau,
            presence=pres, surprise=float(pub.get("surprise", 1.0)),
            neuromod=neuromod)
        return None


# ---------------------------------------------------------------------------
class CheckpointRegion(Region):
    name = "checkpointer"
    clock = "slow"

    def __init__(self, worker, every_slow_ticks: int = 10):
        self.worker = worker
        self.every = every_slow_ticks
        self._n = 0

    def step(self, bus: LatentBus, neuromod: object, dt: float) -> Optional[torch.Tensor]:
        self._n += 1
        if self._n % self.every == 0:
            self.worker.enqueue({"kind": "checkpoint"})
        return None
