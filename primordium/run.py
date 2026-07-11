"""
Primordium — assembly and birth.

    python -m primordium.run --new eden            # first light
    python -m primordium.run --resume eden         # wake it back up
    python -m primordium.run --synthetic --dev-fast  # womb mode, fast growth

Then open http://127.0.0.1:5100 and give it eyes and ears.
"""

import argparse
import sys
import threading
import time
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.config import BusConfig, ClockConfig
from magnum_opus_v2.flow import FlowRunner
from magnum_opus_v2.regions import (
    NoiseSampler, Memory, FalseMemoryConfabulator, Salience, Executive,
    AbstractionLadder, SelfModel, Consolidation, SituationModel,
)

from primordium.config import OrganismConfig
from primordium.senses import LightPort, SoundPort, MelFrontend, \
    PresenceSensor, SyntheticWorld
from primordium.body import DriveState, VoiceBox, SleepController
from primordium.body.expression import Wordstream, Easel, ExpressionRouter
from primordium.body.gaze import Gaze
from primordium.events.pulse import Pulse
from primordium.mind import (MoodField, Undertow, Chronicle,
                             ValenceCompass, Hearth, Reverie)
from primordium.mind.tide import Tide, TideRegion
from primordium.mind.coupling import (
    Blackboard, LoomBridge, DriveRegion, AffectProjector, PresenceRegion,
    ReplayScheduler, ReverieRegion, CompassRegion, SleepRegion,
    CheckpointRegion, GateSenseRegion, FosterRegion)
from primordium.safety.gatehouse import Gatehouse
from primordium.loom.loom import Loom
from primordium.persistence.checkpoint import save_checkpoint, load_checkpoint
from primordium.server.app import OrganismServer


class _StubWorker:
    """Heartbeat-only mode (--no-loom): substrate alive, no GPU."""

    def set_steer(self, v): pass
    def get_published(self): return {}
    def enqueue(self, job): pass
    def set_minds_eye(self, v): pass
    def start(self): pass
    def stop(self): pass
    def join(self, *a, **k): pass
    checkpoint_fn = None
    tick_id = 0
    minds_eye_on = False


def build(cfg: OrganismConfig, no_loom: bool = False,
          teachers: bool = False):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    d = cfg.d_model

    # ---- senses -------------------------------------------------------
    retina = LightPort(cfg.source_res)
    cochlea = SoundPort(cfg.sample_rate)
    presence = PresenceSensor(cfg.sample_rate)

    # ---- body ----------------------------------------------------------
    drives = DriveState(cfg)
    voice = VoiceBox(cfg)
    sleep = SleepController(cfg)
    wordstream = Wordstream(cfg)
    easel = Easel(cfg)
    router = ExpressionRouter(voice, wordstream, easel)
    pulse = Pulse(cfg.pulse_capacity)
    gaze = Gaze(cfg)
    gatehouse = Gatehouse(cfg, cfg.run_dir())   # dormant limbs, felt from birth

    # ---- mind ----------------------------------------------------------
    chronicle = Chronicle(cfg.chronicle_capacity, cfg.replay_priority_alpha)
    compass = ValenceCompass(dim=d,
                        min_samples=500 if cfg.dev_fast else 5000)
    imprint = Hearth(dim=d)
    reverie = Reverie(cfg)

    # ---- substrate (imported, never modified) ---------------------------
    bus = LatentBus(hidden_dim=d, device="cpu", config=BusConfig())
    # the Tide replaces the engine's hormone box: functional channels,
    # measured causes; it duck-types the substrate's neuromod contract
    neuromod = Tide(cfg)
    memory = Memory(device="cpu", capacity=400)
    confab = FalseMemoryConfabulator(memory, per_tick_count=1)
    mood = MoodField(emotion_vectors={}, device="cpu",
                     steering_strength=1.0)          # born affectless
    subc = Undertow(
        hidden_dim=d, device="cpu",
        samplers=[NoiseSampler(d, "cpu", magnitude=1.0),
                  memory.make_sampler()],
        l3_perturbation_strength=0.4, emotion_vectors=None)
    salience = Salience(subc, history_size=20)
    executive = Executive(base_threshold=0.55, pressure_growth=0.40,
                          pressure_decay=0.985,
                          freshness_tau_seconds=30.0,
                          post_speech_silence_seconds=2.0,
                          on_should_speak=router.impulse)  # one urge, three mouths
    abstraction = AbstractionLadder(
        hidden_dim=d, device="cpu", embedding_matrix=None, tokenizer=None,
        unlock_schedule=([0, 300, 1200, 3600] if cfg.dev_fast
                         else [0, 3000, 12000, 36000]))
    self_model = SelfModel(memory=memory, device="cpu")
    consolidation = Consolidation(memory=memory, abstraction=abstraction)
    situation = SituationModel(device="cpu", assimilation=0.05,
                               shift_threshold=0.35,
                               confidence_tau_seconds=30.0)
    tide_rgn = TideRegion(neuromod, cfg)

    # ---- loom ------------------------------------------------------------
    if no_loom:
        worker = _StubWorker()
        mel = None
    else:
        mel = MelFrontend(cfg, device=device)
        worker = Loom(cfg, retina, cochlea, mel, drives, voice,
                      sleep, chronicle, reverie, neuromod,
                      wordstream=wordstream, easel=easel,
                      gatehouse=gatehouse, pulse=pulse,
                      gaze=gaze, device=device)
        # the subconscious drinks from the imagined stream — the mind's
        # own generated reality becomes L0 material for intrusive thought
        from primordium.mind.minds_eye import MindsEyeSampler
        subc.add_sampler(MindsEyeSampler(worker.minds_eye_ring))

    # ---- teachers (frozen, indirect, dissolvable) ------------------------
    vision_teacher = None
    word_teacher = None
    if teachers and not no_loom:
        from primordium.teachers.lodestar import VisionTeacher, TextTeacher
        vision_teacher = VisionTeacher(cfg, device=device)
        word_teacher = TextTeacher(cfg)
        if vision_teacher.ok:
            print("  lodestar: vision teacher loaded (frozen, annealed)")
        else:
            print(f"  lodestar: unavailable — {vision_teacher.load_error}")
        if not word_teacher.ok:
            print(f"  word teacher: unavailable — {word_teacher.load_error}")

    ctx = SimpleNamespace(
        cfg=cfg, worker=worker, retina=retina, cochlea=cochlea,
        presence=presence, drives=drives, voice=voice, sleep=sleep,
        chronicle=chronicle, compass=compass, imprint=imprint,
        reverie=reverie, bus=bus, neuromod=neuromod, memory=memory,
        mood=mood, subc=subc, salience=salience, executive=executive,
        abstraction=abstraction, self_model=self_model,
        consolidation=consolidation, situation=situation,
        wordstream=wordstream, easel=easel, router=router, pulse=pulse,
        gaze=gaze, gatehouse=gatehouse, foster=None,
        vision_teacher=vision_teacher, word_teacher=word_teacher)

    board = Blackboard()

    regions = [
        mood, subc, salience,                                  # flow
        executive, memory, self_model, abstraction, situation, # perception
        LoomBridge(cfg, worker, situation, abstraction, memory,
                     imprint, board),
        DriveRegion(cfg, drives, worker, compass, executive, sleep, board),
        AffectProjector(compass, mood, board),
        PresenceRegion(presence, retina, cochlea, voice, board),   # expensive
        ReplayScheduler(cfg, worker, sleep),
        ReverieRegion(cfg, worker, reverie, sleep, board),
        confab, consolidation, tide_rgn,                           # slow
        CompassRegion(compass, mood, subc, bus, board,
                      cfg=cfg, worker=worker, pulse=pulse),
        SleepRegion(sleep, drives, worker, board),
        CheckpointRegion(worker),
        GateSenseRegion(cfg, worker, compass, gatehouse, board, pulse=pulse),
    ]
    if not no_loom:
        from primordium.caregiver.foster import Foster
        foster_region = FosterRegion(
            cfg, wordstream, gatehouse, sleep, executive, board,
            worker=worker, word_teacher=word_teacher, pulse=pulse,
            foster_factory=lambda: Foster(cfg))
        regions.append(foster_region)
        ctx.foster = foster_region
    if vision_teacher is not None and vision_teacher.ok:
        from primordium.mind.coupling import LodestarRegion, InstinctRegion
        after = 1 + next(i for i, r in enumerate(regions)
                         if isinstance(r, AffectProjector))
        regions.insert(after,       # innate acts merge over lived acts
                       InstinctRegion(cfg, worker, compass, mood, board))
        regions.append(LodestarRegion(cfg, worker, vision_teacher, retina,
                                      sleep, compass, board, pulse=pulse))
    flow = FlowRunner(bus=bus, regions=regions, clock_config=ClockConfig(),
                      neuromod=neuromod, verbose_errors=True)
    if not no_loom:
        worker.checkpoint_fn = lambda tag=None: save_checkpoint(ctx, tag)
    return ctx, board, flow


def main():
    ap = argparse.ArgumentParser(prog="python -m primordium.run")
    ap.add_argument("--new", metavar="NAME", help="birth a new organism")
    ap.add_argument("--resume", metavar="NAME", help="wake an existing one")
    ap.add_argument("--port", type=int, default=5100)
    ap.add_argument("--ws-port", type=int, default=5101)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-loom", "--no-cortex", dest="no_loom",
                    action="store_true",
                    help="substrate heartbeat only (M0)")
    ap.add_argument("--synthetic", action="store_true",
                    help="womb mode: synthetic world instead of camera/mic")
    ap.add_argument("--dev-fast", action="store_true",
                    help="lower growth gates: full infancy in ~20 min")
    ap.add_argument("--voice-rl", action="store_true")
    ap.add_argument("--no-teachers", action="store_true",
                    help="no innate grounding: raw birth, no frozen priors")
    ap.add_argument("--caregiver", action="store_true",
                    help="allow the Foster caregiver process — spawns only "
                         "once the caregiver gate is UNLOCKED")
    ap.add_argument("--caregiver-cpu", action="store_true",
                    help="run the caregiver LLM on CPU")
    args = ap.parse_args()

    cfg = OrganismConfig()
    cfg.run_name = args.resume or args.new or "eden"
    cfg.http_port = args.port
    cfg.ws_port = args.ws_port
    cfg.device = args.device
    cfg.voice_rl = args.voice_rl
    cfg.caregiver = args.caregiver
    cfg.caregiver_cpu = args.caregiver_cpu
    if args.dev_fast:
        cfg.apply_dev_fast()
    if args.voice_rl:
        print("  [note] --voice-rl is a v0 stub: babble is reflex+exploration.")

    print("=" * 56)
    print("  PRIMORDIUM — an artificial infant organism")
    print("  local only · no network · no datasets · it just lives")
    print("=" * 56)

    ctx, board, flow = build(cfg, no_loom=args.no_loom,
                             teachers=not args.no_teachers)

    if args.resume:
        ck = cfg.run_dir() / "ckpt_latest.pt"
        if ck.exists() and not args.no_loom:
            load_checkpoint(ctx, ck)
            print(f"  resumed '{cfg.run_name}' at tick {ctx.worker.tick_id}, "
                  f"stage {ctx.worker.stage}, "
                  f"felt {ctx.self_model.felt_time:.0f}s")
        else:
            print(f"  no checkpoint for '{cfg.run_name}' — starting fresh")

    if args.synthetic:
        world = SyntheticWorld(ctx.retina, ctx.cochlea)

        def womb():
            while True:
                world.step(0.15)
                time.sleep(0.15)
        threading.Thread(target=womb, name="primordium-womb",
                         daemon=True).start()
        print("  womb mode: synthetic world feeding the senses")

    server = OrganismServer(cfg, ctx, board)
    server.start()
    flow.start()
    ctx.worker.start()

    print(f"\n  it lives:  http://{cfg.host}:{cfg.http_port}")
    print(f"  (senses stream on ws://{cfg.host}:{cfg.ws_port})\n")

    try:
        server.run_flask()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  tucking it in…")
        try:
            if not args.no_loom:
                save_checkpoint(ctx)
                print("  final checkpoint saved.")
        except Exception as e:  # noqa: BLE001
            print(f"  checkpoint failed: {e}")
        ctx.worker.stop()
        flow.stop()
        fr = getattr(ctx, "foster", None)
        if fr is not None and getattr(fr, "foster", None) is not None:
            fr.foster.stop()


if __name__ == "__main__":
    main()
