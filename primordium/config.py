"""All Primordium knobs in one place."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class StageSpec:
    """One developmental acuity stage. Patch grid must divide res."""
    res: int          # retina resolution (square)
    channels: int     # 1 = gray, 3 = RGB
    patch: int        # patch size in px

    @property
    def n_tokens(self) -> int:
        return (self.res // self.patch) ** 2

    @property
    def patch_dim(self) -> int:
        return self.patch * self.patch * self.channels

    @property
    def label(self) -> str:
        return f"{self.res}x{self.res} {'RGB' if self.channels == 3 else 'gray'}"


@dataclass
class OrganismConfig:
    # ---- loom core ----
    d_model: int = 384            # == LatentBus hidden_dim
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    steer_layer: int = 4          # bus whispers into this block's output
    max_seq: int = 1152           # stage-2 window is 1020 tokens; headroom

    # ---- wordstream (text sense + keyboard motor) ----
    text_tokens: int = 8          # bytes consumed per tick
    text_vocab: int = 258         # 256 bytes + PAD(256) + MSG_END(257)
    chars_out_per_tick: int = 2
    txt_loss_weight: float = 0.5

    # ---- easel (its canvas) ----
    canvas_res: int = 32
    canvas_patch: int = 16        # 4 patches of 16x16x3
    cnv_loss_weight: float = 0.15 # EXCLUDED from surprise/competence (anti-hack)

    # ---- time ----
    tick_ms: int = 150            # target organism tick (~6.7 Hz)
    tick_ms_max: int = 400        # exhausted pace
    window: int = 12              # ticks of context
    state_tokens: int = 4         # detached recurrence tokens

    # ---- audio ----
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 160                # 10 ms
    n_mels: int = 32
    audio_tokens: int = 3         # tokens per tick
    mel_frames_per_token: int = 5 # 32 mels x 5 frames = 160 dims per token

    # ---- vision stages (sky and ground before clouds) ----
    stages: List[StageSpec] = field(default_factory=lambda: [
        StageSpec(res=8, channels=1, patch=2),    # 16 tokens, dim 4
        StageSpec(res=16, channels=1, patch=4),   # 16 tokens, dim 16
        StageSpec(res=32, channels=3, patch=4),   # 64 tokens, dim 48
    ])
    source_res: int = 96          # browser uplink resolution (always max)

    # ---- learning ----
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_target_momentum: float = 0.996
    w_slow_tau: float = 0.999     # consolidated-self parameter EMA
    live_lr_gain: float = 0.3     # wake steps are gentle
    audio_loss_weight: float = 0.7
    var_loss_weight: float = 0.5
    cov_loss_weight: float = 0.05
    surprise_fast_tau_s: float = 120.0
    surprise_slow_tau_s: float = 1200.0

    # ---- replay / chronicle ----
    chronicle_capacity: int = 20000
    replay_batch: int = 4
    replay_every_s: float = 3.0
    replay_priority_alpha: float = 0.6

    # ---- drives ----
    drive_weights: dict = field(default_factory=lambda: {
        "energy": 0.15, "competence": 0.30, "novelty": 0.20,
        "social": 0.25, "vitality": 0.10,
    })
    setpoint_calibration_s: float = 1800.0   # first 30 min of life
    setpoint_tau_s: float = 172800.0         # 48 h adaptation after

    # ---- sleep ----
    sleep_enter_pressure: float = 0.75
    sleep_exit_pressure: float = 0.25
    sleep_replay_multiplier: int = 4
    lookahead_every: int = 20
    lookahead_alpha: float = 0.3

    # ---- development gates ----
    stage_loss_theta: float = 0.45
    stage_lp_eps: float = 0.004
    stage_dwell_ticks: List[int] = field(default_factory=lambda: [60000, 120000])
    stage_window_s: float = 600.0
    dev_fast: bool = False        # --dev-fast: infancy in ~20 min

    # ---- voice ----
    voice_dims: int = 11          # f0_log + 8 band gains + noise_mix + amplitude
    f0_range: tuple = (80.0, 400.0)
    phonation_gate_s: float = 3.0
    exploration_sigma: float = 0.15
    voice_rl: bool = False

    # ---- reverie ----
    dream_horizon: int = 8
    dream_every_s: float = 10.0

    # ---- coupling ----
    steer_gain: float = 0.08
    thought_feedback_gain: float = 0.02
    percept_surprise_capture: float = 1.5

    # ---- serving ----
    http_port: int = 5100
    ws_port: int = 5101
    host: str = "127.0.0.1"

    # ---- lodestar (annealed frozen-teacher grounding) ----
    lodestar_model: str = "microsoft/Florence-2-large"
    text_teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    lodestar_feat_dim: int = 1024
    distill_w0: float = 0.2
    distill_every_s: float = 3.0
    distill_off_theta: float = 0.25   # hard-off below this FRACTION of L_birth
    word_w0: float = 0.1
    word_tau_msgs: float = 2000.0

    # ---- instincts (innate valence priors — annotated, retirable) ----
    probe_jitters: int = 16
    needle_every_s: float = 60.0
    instinct_weight: float = 0.3      # vs 1.0 for lived affects
    instinct_retire_lived: int = 5    # lived affects that end infancy priors

    # ---- foster (caregiver LLM process) ----
    caregiver: bool = False
    caregiver_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    caregiver_cpu: bool = False
    caregiver_stub: bool = False      # tests: protocol without the 3.6GB model
    caregiver_reply_min_s: float = 20.0
    caregiver_checkin_s: float = 60.0
    caregiver_max_chars: int = 120

    # ---- gatehouse (measured milestones; unlock is the creator's word) ----
    gate_caregiver_stage: int = 1
    gate_caregiver_loss_frac: float = 0.35   # ema_slow below this x L_birth
    gate_caregiver_affects: int = 3
    gate_caregiver_age_ticks: int = 50_000
    gate_webtext_stage: int = 2
    gate_webtext_caregiver_days: float = 7.0

    # ---- fringe (soft fast-plastic rim around the hard core) ----
    fringe_sprouts_per_site: int = 3   # 0 disables the fringe entirely
    fringe_rank: int = 4
    fringe_lr_mult: float = 20.0       # eager edge vs the steady core
    fringe_weight_decay: float = 0.05  # unused sprouts fade to silence
    fringe_merge_frac: float = 0.02    # utility vs ema_slow to harden inward
    fringe_min_probes: int = 4
    fringe_idle_age: int = 20_000      # ticks before an idle sprout recycles

    # ---- bloom (the growing core; elastic, measured, sleep-only) ----
    birth_anatomy: Optional[dict] = None   # None = default 8-block body
    bloom_enabled: bool = True
    bloom_window_s: float = 120.0
    bloom_loss_floor: float = 0.6     # stuck = mean loss > this x entry loss
    bloom_lp_eps: float = 0.01        # plateau = |mean lp| below this
    bloom_cooldown_ticks: int = 30_000
    bloom_widen_k: int = 256
    bloom_block_every: int = 3        # every Nth bloom adds depth, not width
    bloom_vram_headroom: float = 1.5

    # ---- reach (lifetime attention: the far past, attendable) ----
    reach_enabled: bool = True
    reach_capacity: int = 4096
    reach_topk: int = 8
    reach_write_every: int = 8        # plus every surprising moment
    reach_write_surprise: float = 1.5
    reach_exclude_recent: int = 24    # the window already holds the near past
    reach_min_sim: float = 0.10
    reach_probe_every: int = 64       # counterfactual: same tick, no memory

    # ---- gaze (active attention: looking is an action) ----
    gaze_enabled: bool = True
    gaze_zoom_min: float = 0.5        # 2x fovea at most; 1.0 = whole scene
    gaze_chase: float = 0.15          # how hard it chases the error's mass
    gaze_zoom_rate: float = 0.08      # concentration -> zoom-in pressure
    gaze_noise: float = 0.02          # OU exploration; the eye never freezes
    gaze_boredom_lp: float = 1e-3     # "staring taught me nothing" threshold
    gaze_release_s: float = 20.0      # the noisy-TV valve
    gaze_release_hold_s: float = 6.0  # stay wide after a release
    gaze_shift_emit: float = 0.08     # a shift this big is a real saccade
    gaze_cost: float = 0.3            # energy per unit of eye movement

    # ---- motor learning (actions judged by the reward that followed) ----
    w_motor: float = 0.05
    motor_lr_mult: float = 10.0   # sparse consequence needs a hot optimizer
    w_motor_entropy: float = 0.02  # anti-mutism: punishment must never
                                   # collapse the keyboard into silence
    w_motor_leash: float = 2e-3    # logit L2: certainty is never absolute,
                                   # so no policy state is absorbing

    # ---- tide (global modulation; authored controller constants,
    #      disclosed — the values move only from measured causes) ----
    tide_baseline_arousal: float = 0.1
    tide_baseline_reward: float = 0.2
    tide_baseline_calm: float = 0.5
    tide_drift: float = 0.15
    tide_calm_bump: float = 0.3

    # ---- watch (many cheap eyes; spikes are z vs each stream's history) ----
    watch_enabled: bool = True
    watch_spike_z: float = 3.5
    watch_min_n: int = 64             # watchers stay silent until they
                                      # have a history to defy

    # ---- wheel (slow prediction over summaries of the fast world) ----
    wheel_enabled: bool = True
    wheel_window_ticks: int = 16      # authored clock constant, ~2.4s
    wheel_spike_z: float = 3.0
    wheel_min_turns: int = 8

    # ---- grip (the hand made load-bearing: Era 6 MEASURED that paint
    #      efference was an input the world model ignored — ratio ~1.0.
    #      The grip trains the route: predict the canvas blindfolded, and
    #      infer the stroke from the change it left) ----
    grip_enabled: bool = True
    grip_every: int = 6               # blindfold pass cadence (extra fwd+bwd)
    grip_mask_groups: int = 4         # recent canvases hidden; older remain
    w_grip: float = 0.5
    w_hand: float = 1.0               # supervised stroke->delta forward model
    w_inv: float = 0.05               # inverse dynamics, every tick (cheap)
    grip_probe_every: int = 96        # counterfactual: same window, no hand

    # ---- pulse (honest event feed) ----
    pulse_capacity: int = 512

    # ---- misc ----
    device: str = "cuda"
    run_name: str = "eden"

    def apply_dev_fast(self) -> None:
        """Lower the developmental gates so a full infancy fits in ~20 min."""
        self.dev_fast = True
        self.stage_dwell_ticks = [1500, 3000]
        self.stage_window_s = 45.0
        self.setpoint_calibration_s = 120.0
        self.stage_loss_theta = 0.9
        self.stage_lp_eps = 0.05
        self.gate_caregiver_age_ticks = 3000
        self.bloom_cooldown_ticks = 2000

    def run_dir(self) -> Path:
        d = RUNS_DIR / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d
