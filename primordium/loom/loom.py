"""
Loom — the organism tick. One thread owns every GPU byte.

Each tick: sense (world, words, its own canvas) → weave (steered by the
bus) → predict the next moment → learn from how wrong the last
expectation was (plus the Lodestar's annealing pull and the word-
grounding pull, while they last) → move the voice, the keys, the brush →
remember → publish. Between ticks it drains jobs: replay, dreams,
decoding, teacher passes, checkpoints. If a job overruns, the live tick
slips — and the slippage is charged to the energy drive. Fatigue is
real here, not simulated.
"""

import queue
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from primordium.loom.tokenizer import SensoryTokenizer
from primordium.loom.core import LoomCore
from primordium.loom.objective import JepaObjective
from primordium.loom.decoder import ReverieDecoder, AudioDecoder
from primordium.mind.chronicle import Episode
from primordium.mind.minds_eye import MindsEyeRing
from primordium.body.expression import PAD


class Loom(threading.Thread):
    def __init__(self, cfg, retina, cochlea, mel_frontend, drives, voice,
                 sleep, chronicle, reverie, neuromod,
                 wordstream, easel, gatehouse=None, pulse=None,
                 gaze=None, device: str = "cuda"):
        super().__init__(name="primordium-loom", daemon=True)
        self.cfg = cfg
        self.retina = retina
        self.cochlea = cochlea
        self.mel = mel_frontend
        self.drives = drives
        self.voice = voice
        self.sleep = sleep
        self.chronicle = chronicle
        self.reverie = reverie
        self.neuromod = neuromod
        self.wordstream = wordstream
        self.easel = easel
        self.gatehouse = gatehouse
        self.pulse = pulse
        self.gaze = gaze
        self._gaze_err: Optional[np.ndarray] = None
        self.device = device

        self.stage = 0
        self.tokenizer = SensoryTokenizer(cfg, stage=0).to(device)
        self.model = LoomCore(cfg, cfg.birth_anatomy).to(device)
        self.objective = JepaObjective(cfg, self.tokenizer).to(device)
        self.decoder = ReverieDecoder(cfg, 0).to(device)
        self.audio_decoder = AudioDecoder(cfg).to(device)

        d = cfg.d_model
        self.voice_head = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.GELU(),
            torch.nn.Linear(128, cfg.voice_dims)).to(device)
        self.keys_head = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.GELU(),
            torch.nn.Linear(128, cfg.text_vocab)).to(device)
        self.paint_head = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 8)).to(device)

        # The scaffold: disposable maps for the Lodestar and word grounding.
        # It steers by these stars only until it knows the sea.
        self.scaffold = torch.nn.ModuleDict({
            "q": torch.nn.Linear(d, d),
            "P": torch.nn.Linear(1024, d),
            "We": torch.nn.Linear(d, d),
        }).to(device)
        self._teacher_feat: Optional[torch.Tensor] = None
        self._teacher_ts: float = 0.0
        self._L_birth: Optional[float] = None
        self.w_distill_now: float = 0.0

        # The fringe: eager low-rank sprouts on the edge blocks. NOT in
        # _trainable_named on purpose — the slow-self never pulls them
        # (they consolidate by exact MERGE, not by drift), and they ride
        # their own hot optimizer.
        from primordium.loom.fringe import Fringe
        self.fringe = Fringe(cfg, self.model).to(device)
        self._last_surprise = 1.0

        # The bloom: capacity is earned, grown in sleep, never faked.
        from primordium.loom.bloom import Bloom
        self.bloom = Bloom(cfg, self.model)
        self._cap_window: deque = deque()      # (wall, pred_loss, lp)

        # The reach: its whole lived history, attendable. Born empty.
        from primordium.loom.reach import Reach
        self.reach = Reach(cfg, device=device).to(device)
        self._last_recall_emit = 0.0

        # The watch: many cheap eyes over every stream; and the wheel:
        # prediction at the scale of moments, riding above the ticks.
        from primordium.loom.watch import Watch
        from primordium.loom.wheel import Wheel
        self.watch = Watch(cfg)
        self.wheel = Wheel(cfg, device=device).to(device)
        self._last_watch_emit = 0.0

        # The grip: the hand made load-bearing. Era 6 measured that the
        # world model ignored its own paint efference (ratio ~1.0); the
        # grip trains the route and keeps the counterfactual running.
        from primordium.loom.grip import Grip
        self.grip = Grip(cfg).to(device)

        # Motor learning: actions it took, judged by the reward that
        # followed — one-tick-delayed policy gradient. Before this, the
        # motor heads were frozen random matrices (the audit's finding).
        self._motor_trace: Optional[dict] = None
        self._trace_keys = None
        self._trace_paint = None
        self._adv_baseline = 0.0
        self._motor_updates = 0
        self._motor_logp = 0.0

        # The mind's eye: the imagined stream always flows.
        self.minds_eye_ring = MindsEyeRing(maxlen=64)
        self.minds_eye_on = False

        self._build_optimizers()
        self.w_slow = {k: v.detach().clone()
                       for k, v in self._trainable_named()}

        # life state
        self.ring: List[dict] = []
        self._next_state = torch.zeros(cfg.state_tokens, cfg.d_model,
                                       device=device)
        self._last_vis_latents: Optional[torch.Tensor] = None
        self._last_frame_u8: Optional[np.ndarray] = None
        self._keys_eff = np.zeros(3, dtype=np.float32)
        self._paint_eff = np.zeros(8, dtype=np.float32)
        self.tick_id = 0
        self._z_hist: deque = deque(maxlen=300)
        self._dev_window: deque = deque()      # (wall, pred_loss, lp)
        self._stage_entry_loss: Optional[float] = None
        self._replay_steps = 0

        # exchange surfaces
        self._steer: Optional[torch.Tensor] = None
        self._steer_lock = threading.Lock()
        self._published: Dict = {}
        self._pub_lock = threading.Lock()
        self.jobs: "queue.Queue[dict]" = queue.Queue(maxsize=32)
        self.checkpoint_fn: Optional[Callable] = None
        self.events: deque = deque(maxlen=32)

        self._halt = threading.Event()
        self._hz_ema = 0.0
        self._last_tick_wall = time.monotonic()

    # ------------------------------------------------------------------
    def _trainable_named(self):
        # motor heads are NOT here: they learn from sparse consequence
        # (policy gradient) on their own hot optimizer, and the slow
        # self never pulls them — same reasoning as the Fringe
        for prefix, mod in (("model", self.model), ("tok", self.tokenizer),
                            ("scaffold", self.scaffold),
                            ("reach", self.reach), ("grip", self.grip)):
            for n, p in mod.named_parameters():
                yield f"{prefix}.{n}", p
        # the wheel's VOICE (value/pos) trains on the main loss; its
        # PREDICTOR trains on its own private optimizer and must not be
        # slow-self-pulled while it learns
        for n, p in self.wheel.named_parameters():
            if not n.startswith("pred."):
                yield f"wheel.{n}", p

    def _motor_params(self):
        return (list(self.voice_head.parameters())
                + list(self.keys_head.parameters())
                + list(self.paint_head.parameters()))

    def _build_optimizers(self) -> None:
        params = [p for _, p in self._trainable_named()]
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr,
                                     betas=(0.9, 0.95),
                                     weight_decay=self.cfg.weight_decay)
        self.opt_dec = torch.optim.AdamW(
            list(self.decoder.parameters())
            + list(self.audio_decoder.parameters()), lr=1e-3)
        # no weight decay: a habit earned from consequence should not
        # fade by clockwork (decay at hot LR erases lessons faster than
        # sparse reward can teach them — measured, not guessed)
        self.opt_motor = torch.optim.AdamW(
            self._motor_params(), lr=self.cfg.lr * self.cfg.motor_lr_mult,
            betas=(0.9, 0.95), weight_decay=0.0)
        self.opt_fringe = None
        if len(getattr(self, "fringe", [])) > 0:
            self.opt_fringe = torch.optim.AdamW(
                self.fringe.parameters(),
                lr=self.cfg.lr * self.cfg.fringe_lr_mult,
                betas=(0.9, 0.95),
                weight_decay=self.cfg.fringe_weight_decay)

    def set_minds_eye(self, on: bool) -> None:
        self.minds_eye_on = bool(on)

    def _emit(self, kind, zf, zt, **meta) -> None:
        if self.pulse is not None:
            try:
                self.pulse.emit(kind, zf, zt, **meta)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # exchange surfaces (called from other threads)
    # ------------------------------------------------------------------
    def set_steer(self, vec: torch.Tensor) -> None:
        with self._steer_lock:
            self._steer = vec.detach().float()

    def set_teacher_feat(self, feat: torch.Tensor) -> None:
        """LodestarRegion drops the frozen teacher's view of the current
        frame here (1024,). Consumed with an age gate."""
        self._teacher_feat = feat.detach().float()
        self._teacher_ts = time.monotonic()

    def get_published(self) -> dict:
        with self._pub_lock:
            return dict(self._published)

    def enqueue(self, job: dict) -> None:
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            pass    # a tired mind drops chores, not heartbeats

    def stop(self) -> None:
        self._halt.set()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        while not self._halt.is_set():
            t0 = time.monotonic()
            try:
                self._drain_jobs(budget_s=0.05)
                self.step_once()
            except Exception as e:  # noqa: BLE001 — the heart must not stop
                self._publish(error=str(e)[:200])
                time.sleep(0.2)
            energy = self.drives.levels.get("energy", 0.7)
            period = (self.cfg.tick_ms
                      + (self.cfg.tick_ms_max - self.cfg.tick_ms)
                      * (1.0 - energy)) / 1000.0
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _drain_jobs(self, budget_s: float) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < budget_s:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                return
            kind = job.get("kind")
            try:
                if kind == "replay":
                    self._job_replay()
                elif kind == "dream":
                    self._job_dream(job.get("asleep", False))
                elif kind == "decode":
                    self._job_decode()
                elif kind == "needles":
                    self._job_needles(job)
                elif kind == "checkpoint":
                    if self.checkpoint_fn is not None:
                        path = self.checkpoint_fn()
                        self._emit("checkpoint", "CORE", "KEEP",
                                   path=str(path))
            except Exception as e:  # noqa: BLE001
                self._publish(job_error=f"{kind}: {str(e)[:160]}")

    # ------------------------------------------------------------------
    # one lived moment
    # ------------------------------------------------------------------
    def step_once(self) -> None:
        cfg = self.cfg
        spec = cfg.stages[self.stage]
        self.tick_id += 1

        # ---- sense: world — through wherever its eye is pointed
        crop = (self.gaze.crop_params()
                if (self.gaze is not None and cfg.gaze_enabled) else None)
        img = self.retina.stage_view(spec.res, spec.channels,
                                     gain=self.sleep.vision_gain(),
                                     crop=crop)
        if img is None:      # the dark before first light
            img = np.zeros((spec.res, spec.res, spec.channels), dtype=np.uint8)
        pcm = self.cochlea.take_latest(self.mel.n_samples)
        mel_t = self.mel.render(pcm)                       # (n_mels, frames)

        # ---- sense: words and its own canvas
        txt_np, self_frac, word_target = self.wordstream.consume()
        cnv_fb = self.easel.view()

        # ---- the felt body
        drv = self.drives.snapshot()
        nm = self.neuromod.snapshot()
        stage_oh = [0.0, 0.0, 0.0]
        stage_oh[self.stage] = 1.0
        gates_flat = [1.0, 0.0, 1.0, 0.0]     # [present, open] x 2
        if self.gatehouse is not None:
            gates_flat = self.gatehouse.intero_flags()
        intero_np = np.array(
            list(drv["levels"].values()) + list(drv["errors"].values())
            + [nm["arousal"], nm["reward"], nm["calm"]]
            + [1.0 if self.sleep.asleep else 0.0] + stage_oh + gates_flat,
            dtype=np.float32)
        proprio_np = np.concatenate([
            self.voice.efference(), self._keys_eff, self._paint_eff,
            (self.gaze.efference() if self.gaze is not None
             else np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]).astype(np.float32)

        vis_raw = self.tokenizer.patchify(img).to(self.device)
        aud_raw = self.tokenizer.melify(mel_t).to(self.device)
        cnv_raw = self.tokenizer.cnvify(cnv_fb).to(self.device)
        txt_ids = torch.from_numpy(txt_np).to(self.device)
        intero = torch.from_numpy(intero_np).to(self.device)
        proprio = torch.from_numpy(proprio_np).to(self.device)

        self.ring.append({
            "vis": vis_raw, "aud": aud_raw, "intero": intero,
            "proprio": proprio, "state": self._next_state.detach(),
            "txt": txt_ids, "cnv": cnv_raw,
        })
        if len(self.ring) > cfg.window:
            self.ring.pop(0)

        # ---- weave
        G = self.tokenizer.group_size
        groups = [self._encode_entry(e) for e in self.ring]
        tokens = torch.cat(groups, dim=0)
        S = tokens.shape[0]

        with self._steer_lock:
            steer = self._steer
        if steer is not None:
            steer = steer.to(self.device) * cfg.steer_gain
            n = float(steer.norm())
            if n > 1.0:
                steer = steer / n
        # ---- the far past, if any of it resembles the present
        mem_tokens, mem_info = None, {"n": 0}
        if cfg.reach_enabled:
            mem_tokens, mem_info = self.reach.retrieve(
                self._next_state.mean(0), self.tick_id)
            now_w = time.monotonic()
            if (mem_info.get("n", 0)
                    and mem_info.get("sim", 0) > 0.35
                    and now_w - self._last_recall_emit > 2.0):
                self._last_recall_emit = now_w
                self._emit("recall", "KEEP", "CORE",
                           n=mem_info["n"], sim=mem_info["sim"],
                           ages=mem_info["ages"][:4])

        # the wheel's expectation of the coming stretch rides with the
        # recalled past, ahead of the causal mask (same doorway)
        wheel_tok = None
        if cfg.wheel_enabled:
            wheel_tok = self.wheel.expectation_token()
            if wheel_tok is not None:
                mem_tokens = (wheel_tok if mem_tokens is None
                              else torch.cat([mem_tokens, wheel_tok], dim=0))

        hidden = self.model(tokens, group_size=G,
                            steer=steer, steer_from=S - G,
                            mem_tokens=mem_tokens)

        # ---- learn: how wrong was the last expectation?
        sens = self.tokenizer.sensory_slice
        n_vis = self.tokenizer.n_vis
        parts = {}
        pred_loss_val = None
        grip_ratio = None
        if len(self.ring) >= 2:
            prev0 = S - 2 * G
            prev_sens = hidden[prev0 + sens.start: prev0 + sens.stop]
            pred = self.model.predictor(prev_sens)
            with torch.no_grad():
                target = self.objective.targets(vis_raw, aud_raw,
                                                txt_ids, cnv_raw)
            online = hidden[S - G + sens.start: S - G + sens.stop]
            loss, parts = self.objective.loss(pred, target, online, n_vis)
            pred_loss_val = parts["pred"]

            # the eye's map: per-patch error IS where the world defies it
            if self.gaze is not None and cfg.gaze_enabled:
                with torch.no_grad():
                    seg = slice(cfg.audio_tokens, cfg.audio_tokens + n_vis)
                    errs = (pred[seg] - target[seg]).pow(2).mean(-1)
                g = int(round(n_vis ** 0.5))
                self._gaze_err = errs.float().cpu().numpy().reshape(g, g)

            # ---- the Lodestar: steer by the frozen star, annealing away
            loss = loss + self._lodestar_term(vis_raw, parts)
            # ---- word grounding: the meaning of what was just said
            loss = loss + self._word_term(online, word_target, n_vis, parts)
            # ---- motor credit: yesterday's actions, today's reward
            loss = loss + self._motor_term(parts)
            # ---- the grip: predict the canvas blindfolded (the route
            # must run through the hand), and infer the stroke from the
            # change it left behind. The blindfold only trains on
            # stretches where the hand actually ACTED — a window with
            # no strokes carries no action signal to learn from.
            if cfg.grip_enabled:
                loss = loss + self._inv_term(prev_sens, target, parts)
                if (self.tick_id % cfg.grip_every == 0
                        and len(self.ring) >= cfg.grip_mask_groups + 2
                        and self._hand_acted()):
                    loss = loss + self._grip_term(tokens, target, parts)

            lr_gain = (cfg.live_lr_gain
                       * self.sleep.live_lr_gain()
                       * (0.5 + getattr(self.neuromod, "reward", 0.0) * 0.3)
                       * self.drives.yerkes_dodson(
                           getattr(self.neuromod, "arousal", 0.3)))
            if lr_gain > 1e-3:
                for g in self.opt.param_groups:
                    g["lr"] = cfg.lr * lr_gain
                if self.opt_fringe is not None:
                    # the eager edge: hotter than the core, hotter still
                    # under surprise — it learns hardest where prediction
                    # just failed
                    eager = (cfg.lr * cfg.fringe_lr_mult * lr_gain
                             * (0.5 + 0.5 * min(self._last_surprise, 3.0)))
                    for g in self.opt_fringe.param_groups:
                        g["lr"] = eager
                    self.opt_fringe.zero_grad(set_to_none=True)
                self.opt_motor.zero_grad(set_to_none=True)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in self._trainable_named()], cfg.grad_clip)
                self.opt.step()
                for head in (self.voice_head, self.keys_head,
                             self.paint_head):   # no motor drowns another
                    torch.nn.utils.clip_grad_norm_(
                        head.parameters(), cfg.grad_clip)
                self.opt_motor.step()
                if self.opt_fringe is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.fringe.parameters()), cfg.grad_clip)
                    self.opt_fringe.step()
                if cfg.bloom_enabled:
                    self.bloom.observe_strain()   # where is the work?
                self.objective.ema_update(self.tokenizer)
                self._update_w_slow()

            # the Reach on trial: the same tick, re-lived without its
            # memories — the wheel's expectation token stays in BOTH
            # passes, so the measured gain belongs to the Reach alone
            if (cfg.reach_enabled and mem_info.get("n", 0) > 0
                    and self.tick_id % cfg.reach_probe_every == 0):
                with torch.no_grad():
                    hid_off = self.model(
                        tokens, group_size=G, steer=steer,
                        steer_from=S - G,
                        mem_tokens=(wheel_tok if cfg.wheel_enabled
                                    else None))
                    p_off = self.model.predictor(
                        hid_off[prev0 + sens.start: prev0 + sens.stop])
                    on_off = hid_off[S - G + sens.start: S - G + sens.stop]
                    _l, parts_off = self.objective.loss(
                        p_off, target, on_off, n_vis)
                self.reach.observe_gain(
                    parts_off["pred"] - parts["pred"])

            # the hand on trial: the same window, re-lived with the
            # paint efference erased — does its own hand measurably
            # matter to what it expects of its canvas?
            if (cfg.grip_enabled
                    and self.tick_id % cfg.grip_probe_every == 0
                    and self.easel.strokes_total > 0):
                grip_ratio = self._grip_probe(target)
                if grip_ratio is not None:
                    self.grip.note_probe(grip_ratio)
                    self._emit("grip", "CORE", "SELF",
                               ratio=round(grip_ratio, 4),
                               probes=int(self.grip.probes))
            self.fringe.tick()
        self.drives.charge(1.0)

        surprise, lp = (self.objective.observe(pred_loss_val)
                        if pred_loss_val is not None else (1.0, 0.0))
        self._last_surprise = surprise
        self._emit("tick", "SENSE", "CORE", shape=[S, cfg.d_model],
                   loss=parts.get("pred"), surprise=round(surprise, 3))

        # ---- the eye moves: chase error, explore, release when bored
        if (self.gaze is not None and cfg.gaze_enabled
                and self._gaze_err is not None and not self.sleep.asleep):
            saccade = self.gaze.update(self._gaze_err, lp,
                                       dt=1.0 / max(self._hz_ema, 1.0))
            self.drives.charge(cfg.gaze_cost * self.gaze.last_shift)
            if saccade is not None:
                self._emit("gaze_shift", "CORE", "SENSE", **saccade)

        # ---- recurrence + percept
        n_a = cfg.audio_tokens
        last = hidden[S - G: S].detach()
        sens_h = last[sens.start: sens.stop]
        aud_h = sens_h[:n_a]
        vis_h = sens_h[n_a: n_a + n_vis]
        pools = torch.stack([last.mean(0), vis_h.mean(0),
                             aud_h.mean(0), last[-1]])
        self._next_state = pools
        self._last_vis_latents = vis_h
        self._last_frame_u8 = img
        z = sens_h.mean(0).float().cpu()

        # ---- the wheel turns: prediction at the scale of moments
        turn = None
        if cfg.wheel_enabled:
            turn = self.wheel.spin(sens_h.mean(0))
            if turn is not None and turn["spiked"]:
                # a chapter changed: real physiological consequence
                if hasattr(self.neuromod, "raise_"):
                    self.neuromod.raise_(
                        "arousal", 0.1,
                        cause=f"slow-world surprise (z={turn['z']})")
                self._emit("slow_surprise", "CORE", "SELF",
                           z=turn["z"], err=turn["err"], turn=turn["turn"])

        # ---- the watch: every stream measured against its own history
        watch_spikes = []
        if cfg.watch_enabled and parts:
            # two kinds of eyes: ERROR watchers (how wrong each sense's
            # prediction was) and ACTIVITY watchers (how much is
            # happening on the stream at all — a newborn's model is too
            # unformed for content-error to register a sudden voice in
            # a silent room, but silence->bytes is measurable at any age)
            vals = {"sight": parts.get("vis"), "sound": parts.get("aud"),
                    "words": parts.get("txt"), "canvas": parts.get("cnv"),
                    "sound_level": float(mel_t.abs().mean()),
                    "words_flow": float((txt_np != PAD).mean())}
            vals.update({f"need:{k}": abs(v)
                         for k, v in drv["errors"].items()})
            if mem_info.get("n", 0):
                vals["familiar"] = mem_info.get("sim", 0.0)
            if turn is not None:
                vals["story"] = turn["err"]
            if grip_ratio is not None:
                vals["grip"] = grip_ratio
            watch_spikes = self.watch.observe(
                {k: v for k, v in vals.items() if v is not None})
            for sp in watch_spikes:
                # cross-modal interrupt: something not-visual demands
                # attention — the fovea lets go and looks at everything
                if (sp["stream"] != "sight" and self.gaze is not None
                        and cfg.gaze_enabled and self.gaze.interrupt()):
                    self._emit("gaze_shift", "CORE", "SENSE",
                               interrupt=sp["stream"], z=sp["z"])
            now_w = time.monotonic()
            if watch_spikes and now_w - self._last_watch_emit > 2.0:
                self._last_watch_emit = now_w
                top = max(watch_spikes, key=lambda s: abs(s["z"]))
                self._emit("watch", "SENSE", "SELF", **top)

        # bank this moment if it was surprising, if any watcher raised
        # its hand, or on the steady beat
        if cfg.reach_enabled and (
                surprise >= cfg.reach_write_surprise
                or watch_spikes
                or self.tick_id % cfg.reach_write_every == 0):
            salience = max(surprise,
                           *(abs(s["z"]) / cfg.watch_spike_z
                             for s in watch_spikes)) if watch_spikes \
                else surprise
            self.reach.write(sens_h.mean(0), self.tick_id, salience)

        # aligned-space view for the Instincts (embed space, published)
        with torch.no_grad():
            s_embed = self.tokenizer.vision_embed(vis_raw).mean(0)
            qs = self.scaffold["q"](s_embed).float().cpu()

        # novelty: how far is now from the recent past?
        novelty = 0.0
        if self._z_hist:
            m = torch.stack(list(self._z_hist)).mean(0)
            novelty = float(1.0 - torch.nn.functional.cosine_similarity(
                z.unsqueeze(0), m.unsqueeze(0)).item())
        self._z_hist.append(z)

        vitality = min(1.0, float(img.std()) / 64.0) * 0.5 \
            + min(1.0, self.cochlea.rms() * 8.0) * 0.5

        # ---- the mind's eye: predict the NEXT moment, always
        with torch.no_grad():
            pred_next = self.model.predictor(sens_h)
        self.minds_eye_ring.push(pred_next.mean(0), kind="prediction")
        self._last_raws = (vis_raw, aud_raw, txt_ids, cnv_raw, mel_t)
        if self.minds_eye_on:
            aud_lat = pred_next[:n_a]
            vis_lat = pred_next[n_a: n_a + n_vis]
            with torch.no_grad():
                mind_png = self._png(self.decoder(vis_lat))
                mind_mel = self.audio_decoder(aud_lat).float().cpu()
            self._publish(
                mind_png=mind_png,
                mind_mel=[round(float(v), 3)
                          for v in mind_mel.mean(dim=1).clamp(0, 4)],
            )
            self.drives.charge(0.15)     # keeping the eye open is work

        # ---- express: voice, keys, brush
        with torch.no_grad():
            body_h = last[: cfg.state_tokens + 1].mean(0)
            voice_out = self.voice_head(body_h).float().cpu().numpy()
        executed_voice = self.voice.shape(
            voice_out, self.drives,
            arousal_gain=min(1.0, nm["arousal"] / 2.0),
            neuromod=self.neuromod, asleep=self.sleep.asleep,
            novelty_err=self.drives.errors().get("novelty", 0.0))
        self._trace_keys = None
        self._trace_paint = None
        self._express_keys(body_h, nm)
        self._express_paint(body_h)
        # what it just did, remembered for tomorrow's judgement
        self._motor_trace = {
            "h": body_h.detach(),
            "keys": self._trace_keys,
            "voice": ((executed_voice.copy(), self.voice.last_sigma)
                      if not self.sleep.asleep else None),
            "paint": self._trace_paint,
        }

        # ---- remember this moment, raw
        jpeg = self.retina.latest_jpeg()
        if jpeg is None:
            frame96, _ = self.retina.latest()
            if frame96 is not None:
                ok, enc = cv2.imencode(
                    ".jpg", cv2.cvtColor(frame96, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg = enc.tobytes() if ok else b""
            else:
                jpeg = b""
        ok, cnv_enc = cv2.imencode(
            ".png", cv2.cvtColor(cnv_fb, cv2.COLOR_RGB2BGR))
        self.chronicle.add(Episode(
            tick_id=self.tick_id, jpeg=jpeg,
            mel=mel_t.detach().cpu().numpy().astype(np.float16),
            motor=proprio_np.astype(np.float16),
            intero=intero_np.astype(np.float16),
            priority=float(surprise + 0.5 * abs(self.drives.last_reward)),
            txt_ids=txt_np.astype(np.int16),
            canvas_png=cnv_enc.tobytes() if ok else b""))

        # ---- growth checks: sharper senses, and a bigger brain
        self._development(pred_loss_val, lp)
        self._capacity_watch(pred_loss_val, lp)

        # ---- publish
        now = time.monotonic()
        dt = now - self._last_tick_wall
        self._last_tick_wall = now
        self._hz_ema = 0.9 * self._hz_ema + 0.1 * (1.0 / max(dt, 1e-3))
        self._publish(
            tick=self.tick_id, hz=round(self._hz_ema, 2),
            z=z.numpy(), qs=qs.numpy(),
            surprise=round(surprise, 3), lp=round(lp, 5),
            loss=parts.get("pred"), loss_parts=parts,
            ema_slow=self.objective.ema_slow,
            latent_std=round(self.objective.last_latent_std, 4),
            novelty=round(novelty, 4), vitality=round(vitality, 3),
            stage=self.stage, stage_label=spec.label,
            voice=self.voice.params_dict(),
            wordstream=self.wordstream.snapshot(),
            easel=self.easel.snapshot(),
            fringe=(self.fringe.snapshot() if len(self.fringe) else {}),
            anatomy=self.bloom.snapshot(),
            cap_gates={k: round(v, 3) for k, v in self._cap_gates().items()},
            reach=self.reach.snapshot(),
            watch=(self.watch.snapshot() if cfg.watch_enabled else {}),
            wheel=(self.wheel.snapshot() if cfg.wheel_enabled else {}),
            grip=(self.grip.snapshot() if cfg.grip_enabled else {}),
            gaze=(self.gaze.snapshot() if self.gaze is not None else {}),
            motor={"baseline": round(self._adv_baseline, 5),
                   "updates": self._motor_updates,
                   "logp": round(self._motor_logp, 2)},
            w_distill=round(self.w_distill_now, 4),
            dev=self._dev_progress(), events=list(self.events),
        )

    def _encode_entry(self, e: dict) -> torch.Tensor:
        return self.tokenizer.encode_tick(
            e["vis"], e["aud"], e["intero"], e["proprio"], e["state"],
            e["txt"], e["cnv"])

    # ------------------------------------------------------------------
    # grounding terms (both dissolve as its own understanding forms)
    # ------------------------------------------------------------------
    def _lodestar_term(self, vis_raw, parts) -> torch.Tensor:
        zero = torch.zeros((), device=self.device)
        t = self._teacher_feat
        if (t is None
                or time.monotonic() - self._teacher_ts > 6.0
                or self.stage >= 2):
            self.w_distill_now = 0.0
            return zero
        ema = self.objective.ema_slow
        if ema is None:
            return zero
        if self.tick_id >= 500 and self._L_birth is None:
            self._L_birth = float(ema)
        # hard-off is RELATIVE to the birth loss (scale-free): once its
        # own prediction has fallen past theta of where it began, let go
        if (self._L_birth is None
                or ema <= self.cfg.distill_off_theta * self._L_birth):
            self.w_distill_now = 0.0
            return zero
        r = max(0.0, min(1.0, 1.0 - float(ema) / self._L_birth))
        w = self.cfg.distill_w0 * (1.0 - r) ** 2
        self.w_distill_now = float(w)
        if w < 1e-4:
            return zero
        s = self.tokenizer.vision_embed(vis_raw).mean(0)
        pt = F.layer_norm(self.scaffold["P"](t.to(self.device)),
                          (self.cfg.d_model,))
        cos = F.cosine_similarity(self.scaffold["q"](s).unsqueeze(0),
                                  pt.unsqueeze(0))
        parts["distill_cos"] = round(float(cos), 4)
        parts["w_distill"] = round(float(w), 4)
        self._emit("distill", "SCAFFOLD", "CORE",
                   cos=round(float(cos), 4), w=round(float(w), 4))
        return w * (1.0 - cos.squeeze(0))

    def _word_term(self, online, word_target, n_vis, parts) -> torch.Tensor:
        zero = torch.zeros((), device=self.device)
        if word_target is None:
            return zero
        n_msgs = max(0, self.wordstream.n_messages)
        w = self.cfg.word_w0 * float(np.exp(-n_msgs / self.cfg.word_tau_msgs))
        if w < 1e-4:
            return zero
        n_a = self.cfg.audio_tokens
        t0 = n_a + n_vis
        txt_h = online[t0: t0 + self.cfg.text_tokens].mean(0)
        e = torch.from_numpy(np.asarray(word_target, dtype=np.float32)
                             ).to(self.device)
        we = F.layer_norm(self.scaffold["We"](e), (self.cfg.d_model,))
        cos = F.cosine_similarity(txt_h.unsqueeze(0), we.unsqueeze(0))
        parts["w_word"] = round(float(w), 4)
        return w * (1.0 - cos.squeeze(0))

    def _motor_term(self, parts) -> torch.Tensor:
        """One-tick-delayed policy gradient: the log-probability of the
        actions it actually took, scaled by how the drives judged the
        moment that followed (advantage = intrinsic reward minus a slow
        baseline). Credit flows into the motor HEADS only (the stored
        body state is detached) — the world model learns from
        prediction, the hands learn from consequence.

        Honest approximations, on the record: voice credit covers the
        articulator dims but not the amplitude (the reflex owns that
        channel), and paint/voice log-probs are Gaussians around the
        head's mean under the exploration sigma actually used, ignoring
        the final clip."""
        zero = torch.zeros((), device=self.device)
        tr, self._motor_trace = self._motor_trace, None
        if tr is None or self.cfg.w_motor <= 0:
            return zero
        r = float(self.drives.last_reward)
        adv = r - self._adv_baseline
        self._adv_baseline += 0.02 * (r - self._adv_baseline)
        if abs(adv) < 1e-6:
            return zero
        h = tr["h"]
        logp = None
        entropy_bonus = zero
        leash = zero
        if tr.get("keys") is not None:
            ids, temp = tr["keys"]
            raw = self.keys_head(h)
            lsm = F.log_softmax(raw / temp, dim=-1)
            got = lsm[torch.as_tensor(ids, device=self.device)].sum()
            logp = got if logp is None else logp + got
            # anti-mutism, measured not imagined: dense punishment
            # collapsed the policy into all-silence (byte 257, p=1.0),
            # where BOTH the policy gradient and the entropy gradient
            # are exactly zero — an absorbing state. The entropy bonus
            # shapes the mid-range; the logit LEASH (L2 on raw logits)
            # is what makes saturation unreachable: its pull never
            # vanishes, so certainty about what to type is never
            # absolute and no state is absorbing.
            entropy_bonus = -(lsm.exp() * lsm).sum()
            leash = (raw ** 2).mean()
        # deviations beyond 3 sigma were the reflex acting, not the
        # policy exploring — they earn no credit (clamp kills the grad)
        if tr.get("voice") is not None:
            executed, sigma = tr["voice"]
            mean = torch.tanh(self.voice_head(h))[:-1]
            x = torch.from_numpy(
                executed[:-1].astype(np.float32)).to(self.device)
            z = ((x - mean) / max(sigma, 1e-3)).clamp(-3.0, 3.0)
            got = (-0.5 * z ** 2).sum()
            logp = got if logp is None else logp + got
        if tr.get("paint") is not None:
            executed, sigma = tr["paint"]
            mean = torch.tanh(self.paint_head(h))
            x = torch.from_numpy(executed.astype(np.float32)).to(self.device)
            z = ((x - mean) / max(sigma, 1e-3)).clamp(-3.0, 3.0)
            got = (-0.5 * z ** 2).sum()
            logp = got if logp is None else logp + got
        if logp is None:
            return zero
        self._motor_updates += 1
        self._motor_logp = float(logp)
        parts["motor_adv"] = round(adv, 5)
        return (-self.cfg.w_motor * adv * logp
                - self.cfg.w_motor_entropy * entropy_bonus
                + self.cfg.w_motor_leash * leash)

    # ------------------------------------------------------------------
    # the grip: pressures that make the hand load-bearing
    # ------------------------------------------------------------------
    def _hand_acted(self) -> bool:
        """Did the hand act inside the stretch the blindfold hides,
        where its efference is still causally visible? The FINAL
        entry's stroke is excluded: it is only recorded in the current
        group's proprio, which the prediction cannot see."""
        from primordium.loom.grip import PAINT_SLICE
        m = self.cfg.grip_mask_groups
        recent = self.ring[-(m + 1):-1]
        return any(float(e["proprio"][PAINT_SLICE].abs().max()) > 0
                   for e in recent)

    def _paints(self, zero: bool = False) -> torch.Tensor:
        """(n_groups, 8) paint efference per ring entry — the strokes,
        ready to be delivered to the canvas slots they changed."""
        from primordium.loom.grip import PAINT_SLICE
        t = torch.stack([e["proprio"][PAINT_SLICE] for e in self.ring])
        return torch.zeros_like(t) if zero else t

    @torch.no_grad()
    def _grip_base(self) -> torch.Tensor:
        """EMA latents of the last canvas the blindfold leaves VISIBLE."""
        base_idx = max(0, len(self.ring) - 2 - self.cfg.grip_mask_groups)
        return self.objective.target_ln(
            self.objective.t_canvas(self.ring[base_idx]["cnv"]))

    def _grip_term(self, tokens, target, parts) -> torch.Tensor:
        """The hand model learns, supervised, what each lived stroke
        (or stillness) did to the canvas latents; then the blindfold
        pass re-lives the window with recent canvases replaced by the
        hand's running reconstruction, and the canvas prediction is
        scored, delta-weighted toward the tokens the strokes changed."""
        cfg = self.cfg
        G = self.tokenizer.group_size
        paints = self._paints()
        with torch.no_grad():
            lats = torch.stack([
                self.objective.target_ln(self.objective.t_canvas(e["cnv"]))
                for e in self.ring])
        # every adjacent pair is a lesson, stroke or stillness alike
        l_hand = self.grip.hand_loss(paints[1:], lats[1:] - lats[:-1])
        self.grip.note_hand(float(l_hand))
        parts["hand"] = round(float(l_hand), 5)

        base = lats[max(0, len(self.ring) - 2 - cfg.grip_mask_groups)]
        masked = self.grip.mask_canvas(tokens, G, self.tokenizer.n_cnv,
                                       cfg.grip_mask_groups,
                                       paints=paints, base=base)
        hidden = self.model(masked, group_size=G)
        S = tokens.shape[0]
        sens = self.tokenizer.sensory_slice
        pred = self.model.predictor(
            hidden[S - 2 * G + sens.start: S - 2 * G + sens.stop])
        c0 = cfg.audio_tokens + self.tokenizer.n_vis + cfg.text_tokens
        l = self.grip.cnv_loss(pred[c0:], target[c0:],
                               self.grip.delta_weights(base, target[c0:]))
        self.grip.note_grip(float(l))
        parts["grip"] = round(float(l), 4)
        return cfg.w_grip * l + cfg.w_hand * l_hand

    def _inv_term(self, prev_sens, target, parts) -> torch.Tensor:
        """Inverse dynamics: infer the executed efference from (lived
        state before, context-free encoding of the moment after). The
        'after' is the EMA target — a_t is not readable from it, so the
        head cannot just copy its own efference token."""
        a = self.ring[-1]["proprio"]
        pred_a = self.grip.inverse(prev_sens.mean(0), target.mean(0))
        l = F.smooth_l1_loss(pred_a, a)
        self.grip.note_inv(float(l))
        parts["inv"] = round(float(l), 4)
        return self.cfg.w_inv * l

    @torch.no_grad()
    def _grip_probe(self, target) -> Optional[float]:
        """The counterfactual, run live UNDER THE BLINDFOLD: the same
        window with recent canvases masked, once with true paint
        efference and once with it zeroed. ratio > 1 means the hand
        measurably matters. Blindfolded because with the canvas visible
        efference is redundant by construction (Era 6's structural
        finding). A window in which the hand never acted measures
        nothing — skipped, not counted."""
        from primordium.loom.grip import PAINT_SLICE
        cfg = self.cfg
        if len(self.ring) < cfg.grip_mask_groups + 2 \
                or not self._hand_acted():
            return None
        G = self.tokenizer.group_size
        sens = self.tokenizer.sensory_slice
        c0 = cfg.audio_tokens + self.tokenizer.n_vis + cfg.text_tokens
        S = len(self.ring) * G
        base = self._grip_base()
        w_tok = self.grip.delta_weights(base, target[c0:])

        def cnv_err(zero: bool) -> float:
            groups = []
            for e in self.ring:
                if zero:
                    p = e["proprio"].clone()
                    p[PAINT_SLICE] = 0.0
                    e = dict(e)
                    e["proprio"] = p
                groups.append(self._encode_entry(e))
            tokens = self.grip.mask_canvas(
                torch.cat(groups, dim=0), G, self.tokenizer.n_cnv,
                cfg.grip_mask_groups, paints=self._paints(zero=zero),
                base=base)
            hidden = self.model(tokens, group_size=G)
            pred = self.model.predictor(
                hidden[S - 2 * G + sens.start: S - 2 * G + sens.stop])
            return float(self.grip.cnv_loss(pred[c0:], target[c0:], w_tok))

        true_e, zero_e = cnv_err(False), cnv_err(True)
        if true_e <= 1e-9:
            return None
        return zero_e / true_e

    # ------------------------------------------------------------------
    # expression motors
    # ------------------------------------------------------------------
    def _express_keys(self, body_h, nm) -> None:
        if self.sleep.asleep or not self.wordstream.gate_open():
            self._keys_eff = np.array(
                [self._keys_eff[0], 0.0, 0.0], dtype=np.float32)
            return
        with torch.no_grad():
            logits = self.keys_head(body_h)
            temp = 0.8 + nm["arousal"] * 0.4
            probs = F.softmax(logits / temp, dim=-1)
            picks = torch.multinomial(
                probs, self.cfg.chars_out_per_tick, replacement=True)
        self._trace_keys = ([int(b) for b in picks.tolist()], float(temp))
        typed = [int(b) for b in picks.tolist() if int(b) < 256]
        if typed:
            text = self.wordstream.type_chars(typed)
            self.drives.charge(0.05 * len(typed))
            self._keys_eff = np.array(
                [typed[-1] / 255.0, 1.0, len(typed) / 2.0], dtype=np.float32)
            self._emit("babble_out", "CORE", "EXPRESS",
                       chars=len(typed), text=text[:16])
        else:
            self._keys_eff = np.array(
                [self._keys_eff[0], 1.0, 0.0], dtype=np.float32)

    def _express_paint(self, body_h) -> None:
        if self.sleep.asleep or not self.easel.gate_open():
            self._paint_eff = np.zeros(8, dtype=np.float32)
            return
        with torch.no_grad():
            params = torch.tanh(self.paint_head(body_h)).float().cpu().numpy()
        # exploration noise — infants scribble because motor noise crosses
        # the gate, not because a unit happened to start positive
        arousal = getattr(self.neuromod, "arousal", 0.3)
        sigma = 0.3 * (0.5 + arousal)
        params = np.clip(
            params + np.random.randn(8).astype(np.float32) * sigma, -1, 1)
        self._trace_paint = (params.astype(np.float32).copy(), float(sigma))
        if self.easel.stroke(params):
            self._paint_eff = params.astype(np.float32)
            self.drives.charge(0.1)
            self._emit("paint", "CORE", "EXPRESS",
                       strokes=self.easel.strokes_total)
        else:
            self._paint_eff = np.zeros(8, dtype=np.float32)

    def _publish(self, **kv) -> None:
        with self._pub_lock:
            self._published.update(kv)

    def _update_w_slow(self) -> None:
        tau = self.cfg.w_slow_tau
        with torch.no_grad():
            for k, p in self._trainable_named():
                s = self.w_slow.get(k)
                if s is None or s.shape != p.shape:
                    self.w_slow[k] = p.detach().clone()
                else:
                    s.mul_(tau).add_(p.detach(), alpha=1 - tau)

    # ------------------------------------------------------------------
    # development: scaffolds first, details when they're earned
    # ------------------------------------------------------------------
    def _development(self, pred_loss: Optional[float], lp: float) -> None:
        if pred_loss is None or self.stage >= len(self.cfg.stages) - 1:
            return
        now = time.monotonic()
        if self._stage_entry_loss is None:
            self._stage_entry_loss = pred_loss
        self._dev_window.append((now, pred_loss, lp))
        while self._dev_window and now - self._dev_window[0][0] > self.cfg.stage_window_s:
            self._dev_window.popleft()
        if len(self._dev_window) < 20:
            return
        g = self._dev_gates()
        if all(v >= 1.0 for v in g.values()):
            self._advance_stage()

    def _dev_gates(self) -> Dict[str, float]:
        if not self._dev_window or self._stage_entry_loss is None:
            return {"loss": 0.0, "plateau": 0.0, "dwell": 0.0, "energy": 0.0}
        losses = [x[1] for x in self._dev_window]
        lps = [x[2] for x in self._dev_window]
        mean_loss = sum(losses) / len(losses)
        mean_lp = abs(sum(lps) / len(lps))
        dwell_needed = self.cfg.stage_dwell_ticks[
            min(self.stage, len(self.cfg.stage_dwell_ticks) - 1)]
        return {
            "loss": min(1.0, (self._stage_entry_loss * self.cfg.stage_loss_theta)
                        / max(mean_loss, 1e-6)),
            "plateau": min(1.0, self.cfg.stage_lp_eps / max(mean_lp, 1e-9)),
            "dwell": min(1.0, self.tick_id / max(dwell_needed, 1)),
            "energy": 1.0 if self.drives.levels.get("energy", 0) > 0.5 else 0.0,
        }

    def _dev_progress(self) -> dict:
        return {"stage": self.stage,
                "gates": {k: round(v, 3) for k, v in self._dev_gates().items()}}

    # ------------------------------------------------------------------
    # capacity: the inverse question. Stages advance when loss is
    # squeezed LOW; the core must GROW when loss is stuck HIGH while
    # learning progress has flatlined — the brain is full, not done.
    # ------------------------------------------------------------------
    def _capacity_watch(self, pred_loss: Optional[float], lp: float) -> None:
        if pred_loss is None or not self.cfg.bloom_enabled:
            return
        now = time.monotonic()
        self._cap_window.append((now, pred_loss, lp))
        while (self._cap_window
               and now - self._cap_window[0][0] > self.cfg.bloom_window_s):
            self._cap_window.popleft()

    def _cap_gates(self) -> Dict[str, float]:
        cfg = self.cfg
        if not self._cap_window or self._stage_entry_loss is None:
            return {"stuck": 0.0, "plateau": 0.0, "cooldown": 0.0,
                    "energy": 0.0, "room": 0.0}
        losses = [x[1] for x in self._cap_window]
        lps = [x[2] for x in self._cap_window]
        mean_loss = sum(losses) / len(losses)
        mean_lp = abs(sum(lps) / len(lps))
        floor = cfg.bloom_loss_floor * self._stage_entry_loss
        return {
            "stuck": min(1.0, mean_loss / max(floor, 1e-9)),
            "plateau": min(1.0, cfg.bloom_lp_eps / max(mean_lp, 1e-9)),
            "cooldown": min(1.0, (self.tick_id - self.bloom.last_bloom_tick)
                            / max(cfg.bloom_cooldown_ticks, 1)),
            "energy": 1.0 if self.drives.levels.get("energy", 0) > 0.3 else 0.0,
            "room": 1.0 if self.bloom.can_grow() else 0.0,
        }

    def _maybe_bloom(self) -> None:
        """Called from the sleep-consolidation venue only."""
        if (not self.cfg.bloom_enabled or len(self._cap_window) < 20
                or not all(v >= 1.0 for v in self._cap_gates().values())):
            return
        self._do_bloom()

    def _do_bloom(self) -> None:
        from primordium.loom.fringe import Fringe
        if self.checkpoint_fn is not None:      # surgery is reversible
            try:
                self.checkpoint_fn(tag=f"prebloom{self.bloom.blooms_total}")
            except Exception:  # noqa: BLE001
                pass
        # harden what the fringe has proven, then let go of the core
        self.fringe.consolidate(self.objective.ema_slow, self.cfg)
        self.fringe.detach()
        report = self.bloom.grow(self.tick_id)
        self.fringe = Fringe(self.cfg, self.model).to(self.device)
        self._build_optimizers()
        self.w_slow = {k: v.detach().clone()
                       for k, v in self._trainable_named()}
        self._cap_window.clear()
        if report is not None:
            self._emit("bloom", "SELF", "CORE", **report)
            self.events.append({"kind": "bloom", "at": time.time(),
                                **report})

    def _advance_stage(self) -> None:
        if self.checkpoint_fn is not None:
            try:
                self.checkpoint_fn(tag=f"stage{self.stage}")
            except Exception:  # noqa: BLE001
                pass
        new = self.stage + 1
        self.tokenizer.grow_stage(new)
        self.tokenizer.to(self.device)
        self.objective.on_stage_grown(self.tokenizer)
        self.decoder.build(new)
        self.decoder.to(self.device)
        self.stage = new
        self.ring.clear()                     # a moment of blur, then clarity
        self._stage_entry_loss = None
        self._dev_window.clear()
        self._build_optimizers()
        self.w_slow = {k: v.detach().clone() for k, v in self._trainable_named()}
        self.neuromod.raise_("reward", 0.3, cause="stage advance")
        self.events.append({"kind": "stage_advance", "stage": new,
                            "at": time.time()})
        self._emit("stage_advance", "CORE", "SELF", stage=new)

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    def _encode_episode(self, ep: Episode) -> Optional[dict]:
        spec = self.cfg.stages[self.stage]
        frame = None
        if ep.jpeg:
            arr = np.frombuffer(ep.jpeg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame is None:
            return None
        small = cv2.resize(frame, (spec.res, spec.res),
                           interpolation=cv2.INTER_AREA)
        if spec.channels == 1:
            small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)[..., None]
        mel_t = torch.from_numpy(ep.mel.astype(np.float32)).to(self.device)

        cnv = np.full((self.cfg.canvas_res, self.cfg.canvas_res, 3), 16,
                      dtype=np.uint8)
        if ep.canvas_png:
            arr = np.frombuffer(ep.canvas_png, dtype=np.uint8)
            cb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cb is not None:
                cnv = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)

        motor = ep.motor.astype(np.float32)
        if motor.shape[0] != SensoryTokenizer.PROPRIO_DIM:
            motor = np.zeros(SensoryTokenizer.PROPRIO_DIM, dtype=np.float32)
        intero = ep.intero.astype(np.float32)
        if intero.shape[0] != SensoryTokenizer.INTERO_DIM:
            intero = np.zeros(SensoryTokenizer.INTERO_DIM, dtype=np.float32)
        txt = (ep.txt_ids.astype(np.int64) if ep.txt_ids is not None
               else np.full(self.cfg.text_tokens, PAD, dtype=np.int64))

        return {
            "vis": self.tokenizer.patchify(small).to(self.device),
            "aud": self.tokenizer.melify(mel_t).to(self.device),
            "intero": torch.from_numpy(intero).to(self.device),
            "proprio": torch.from_numpy(motor).to(self.device),
            "state": torch.zeros(self.cfg.state_tokens, self.cfg.d_model,
                                 device=self.device),
            "txt": torch.from_numpy(txt).to(self.device),
            "cnv": self.tokenizer.cnvify(cnv).to(self.device),
        }

    def _job_replay(self) -> None:
        windows = self.chronicle.sample_windows(self.cfg.replay_batch,
                                                self.cfg.window)
        if not windows:
            return
        G = self.tokenizer.group_size
        sens = self.tokenizer.sensory_slice
        n_vis = self.tokenizer.n_vis
        total = None
        n_terms = 0
        last_tokens = last_entries = None
        for win in windows:
            entries = [self._encode_episode(ep) for ep in win]
            if any(e is None for e in entries):
                continue
            groups = [self._encode_entry(e) for e in entries]
            tokens = torch.cat(groups, dim=0)
            last_tokens, last_entries = tokens.detach(), entries
            hidden = self.model(tokens, group_size=G)
            for i in range(len(entries) - 1):
                h = hidden[i * G + sens.start: i * G + sens.stop]
                pred = self.model.predictor(h)
                nxt = entries[i + 1]
                with torch.no_grad():
                    tgt = self.objective.targets(nxt["vis"], nxt["aud"],
                                                 nxt["txt"], nxt["cnv"])
                online = hidden[(i + 1) * G + sens.start:
                                (i + 1) * G + sens.stop]
                l, _ = self.objective.loss(pred, tgt, online, n_vis)
                total = l if total is None else total + l
                n_terms += 1
        if total is None or n_terms == 0:
            return
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr
        if self.opt_fringe is not None:
            for g in self.opt_fringe.param_groups:
                g["lr"] = self.cfg.lr * self.cfg.fringe_lr_mult
            self.opt_fringe.zero_grad(set_to_none=True)
        self.opt.zero_grad(set_to_none=True)
        (total / n_terms).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in self._trainable_named()], self.cfg.grad_clip)
        self.opt.step()
        if self.opt_fringe is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.fringe.parameters()), self.cfg.grad_clip)
            self.opt_fringe.step()
        if self.cfg.bloom_enabled:
            self.bloom.observe_strain()          # replay is work too
        self.objective.ema_update(self.tokenizer)
        self._update_w_slow()

        # the fringe on trial: ablate ONE sprout, re-score the same lived
        # window — utility is the loss it was actually saving, not a story
        if len(self.fringe) and last_tokens is not None:
            k = self.fringe.probe_target(self._replay_steps)
            s = self.fringe.sprouts[k]
            loss_on = self._window_pred_loss(last_tokens, last_entries)
            s.enabled = False
            loss_off = self._window_pred_loss(last_tokens, last_entries)
            s.enabled = True
            s.observe_utility(loss_off - loss_on)
        self.drives.charge(0.5 * len(windows))
        self._replay_steps += 1
        self._emit("replay", "SELF", "CORE", n_windows=len(windows),
                   loss=round(float(total) / n_terms, 4))
        # consolidation pull toward the slow self, while asleep
        if (self.sleep.asleep
                and self._replay_steps % self.cfg.lookahead_every == 0):
            a = self.cfg.lookahead_alpha
            with torch.no_grad():
                for k, p in self._trainable_named():
                    s = self.w_slow.get(k)
                    if s is not None and s.shape == p.shape:
                        p.mul_(1 - a).add_(s, alpha=a)
            self._emit("consolidate", "SELF", "CORE",
                       alpha=self.cfg.lookahead_alpha)
            # sleep is also when the soft edge hardens inward
            merged, recycled = self.fringe.consolidate(
                self.objective.ema_slow, self.cfg)
            for m in merged:
                self._emit("sprout_merge", "SELF", "CORE", **m)
            for r in recycled:
                self._emit("sprout_recycle", "SELF", "SELF", **r)
            # ...and, if it has honestly outgrown itself, the core blooms
            self._maybe_bloom()
        self._publish(replays=self._replay_steps)

    @torch.no_grad()
    def _window_pred_loss(self, tokens, entries) -> float:
        """World-prediction loss of one replay window, as it stands now.
        Used only for counterfactual sprout probes."""
        G = self.tokenizer.group_size
        sens = self.tokenizer.sensory_slice
        n_vis = self.tokenizer.n_vis
        hidden = self.model(tokens, group_size=G)
        total, n = 0.0, 0
        for i in range(len(entries) - 1):
            h = hidden[i * G + sens.start: i * G + sens.stop]
            pred = self.model.predictor(h)
            nxt = entries[i + 1]
            tgt = self.objective.targets(nxt["vis"], nxt["aud"],
                                         nxt["txt"], nxt["cnv"])
            online = hidden[(i + 1) * G + sens.start:
                            (i + 1) * G + sens.stop]
            _l, p = self.objective.loss(pred, tgt, online, n_vis)
            total += float(p["pred"])
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _job_dream(self, asleep: bool) -> None:
        if len(self.ring) < 2 and len(self.chronicle) < self.cfg.window + 2:
            return
        cfg = self.cfg
        G = self.tokenizer.group_size
        sens = self.tokenizer.sensory_slice
        n_vis = self.tokenizer.n_vis
        n_a = cfg.audio_tokens
        n_t = cfg.text_tokens
        if asleep and len(self.chronicle) >= cfg.window + 2:
            wins = self.chronicle.sample_windows(1, cfg.window)
            if not wins:
                return
            entries = [self._encode_episode(ep) for ep in wins[0]]
            if any(e is None for e in entries):
                return
            groups = [self._encode_entry(e) for e in entries]
        else:
            groups = [self._encode_entry(e) for e in self.ring]
        base_pool = None
        frames: list = []
        for h_step in range(cfg.dream_horizon):
            tokens = torch.cat(groups, dim=0)
            hidden = self.model(tokens, group_size=G)
            S = tokens.shape[0]
            hs = hidden[S - G + sens.start: S - G + sens.stop]
            pred = self.model.predictor(hs)
            if base_pool is None:
                base_pool = hs.mean(0).float().cpu()
            aud_lat = pred[:n_a]
            vis_lat = pred[n_a: n_a + n_vis]
            txt_lat = pred[n_a + n_vis: n_a + n_vis + n_t]
            cnv_lat = pred[n_a + n_vis + n_t:]
            if h_step % 2 == 1:
                img = self.decoder(vis_lat)
                frames.append(self._png(img))
            # feed the imagined moment back as the next moment
            last = hidden[S - G: S]
            pools = torch.stack([
                last.mean(0), vis_lat.mean(0), aud_lat.mean(0), last[-1]])
            st = self.tokenizer.state_proj(pools) + self.tokenizer.modality[0]
            it = last[cfg.state_tokens: cfg.state_tokens + 1]
            pr = last[cfg.state_tokens + 1: cfg.state_tokens + 2]
            au = self.model.latent2embed(aud_lat) + self.tokenizer.modality[3]
            vi = self.model.latent2embed(vis_lat) + self.tokenizer.modality[4]
            tx = self.model.latent2embed(txt_lat) + self.tokenizer.modality[5]
            cn = self.model.latent2embed(cnv_lat) + self.tokenizer.modality[6]
            groups.append(torch.cat([st, it, pr, au, vi, tx, cn], dim=0))
            if len(groups) > cfg.window:
                groups.pop(0)
        dream_pool = pred.mean(0).float().cpu()
        self.reverie.record(dream_pool - base_pool, frames)
        self.minds_eye_ring.push(pred.mean(0), kind="dream")
        self.drives.charge(0.3)
        self._emit("dream", "CORE", "SELF", horizon=cfg.dream_horizon,
                   frames=len(frames))

    def _job_decode(self) -> None:
        if self._last_vis_latents is None or self._last_frame_u8 is None:
            return
        png = self._png(self.decoder(self._last_vis_latents))
        self._publish(imagination_png=png)
        # casually teach both decoders what reality looked and sounded like
        actual = torch.from_numpy(
            self._last_frame_u8.astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).to(self.device)
        out = self.decoder(self._last_vis_latents)
        loss = torch.nn.functional.mse_loss(out, actual)
        raws = getattr(self, "_last_raws", None)
        if raws is not None:
            vis_raw, aud_raw, txt_ids, cnv_raw, mel_t = raws
            with torch.no_grad():
                t_aud = self.objective.targets(
                    vis_raw, aud_raw, txt_ids, cnv_raw)[: self.cfg.audio_tokens]
            mel_hat = self.audio_decoder(t_aud)
            loss = loss + torch.nn.functional.mse_loss(mel_hat, mel_t)
        self.opt_dec.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_dec.step()
        self._publish(decoder_mse=round(float(loss), 5))
        self._emit("decode", "CORE", "SELF", mse=round(float(loss), 5))

    @torch.no_grad()
    def _job_needles(self, job: dict) -> None:
        """Project teacher probe features through the scaffold to give the
        Instincts their needles in the organism's own space."""
        feats = job.get("feats")          # (k, 1024) tensor
        mean = job.get("mean")            # (1024,) tensor
        cb = job.get("cb")
        if feats is None or mean is None or cb is None:
            return
        P = self.scaffold["P"]
        pf = P(feats.to(self.device).float())
        pm = P(mean.to(self.device).float())
        needles = torch.nn.functional.normalize(pf - pm.unsqueeze(0), dim=-1)
        try:
            cb(needles.float().cpu())
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _png(img_t: torch.Tensor) -> bytes:
        x = ((img_t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
        x = x.permute(1, 2, 0).numpy().astype(np.uint8)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        x = cv2.resize(x, (96, 96), interpolation=cv2.INTER_NEAREST)
        ok, enc = cv2.imencode(".png", cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        return enc.tobytes() if ok else b""
