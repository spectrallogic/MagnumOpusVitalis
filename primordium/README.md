# Primordium

**A from-scratch multimodal organism that learns online from its own life.**

Not a language model. A ~16M-parameter transformer core born knowing
nothing, learning continually from live camera and microphone — no
datasets, no pretraining, no cloud — mounted on the Magnum Opus Vitalis
substrate: the latent bus, subconscious, neuromodulator chemistry, memory,
self-model, situation model, and consolidation, all running at their
biological clock rates around a mind that is genuinely growing.

v2 — **the Fostered Mind** — adds hands (it types and paints), inherited
leanings that dissolve (Lodestar, Instincts), a caregiver in the room
(Foster), maturity locks with two keys (Gatehouse), and a dashboard where
every moving pixel is a real computation (Pulse). v2 is a different body and requires a new birth.

## What this is, honestly

An **artificial infant organism** and an embodiment research bed. It is a
live test of the traffic hypothesis — that mind is the *pattern of
traffic*, not the substrate carrying it — at animal-infant scale. It is
**not AGI and it is not ASI**, and nothing here claims felt experience.
The dashboard shows mechanism outputs: drive errors, prediction losses,
projections. What those amount to is exactly the open question this
project exists to explore.

It has no network access. Its only actuators are sound from your
speakers, pixels on a local page, typed characters into its own
conversation stream, and strokes on its own 32×32 canvas. Both servers
bind 127.0.0.1. Nothing leaves the machine.

## Brain as reference, not blueprint

We stopped naming organs after human anatomy. Every part carries a
functional name, and where the design deliberately departs from biology
we say so and why:

| Organ | Was | Departure from biology |
|---|---|---|
| **Loom** | "cortex" | one online learner over an interleaved multimodal token stream; owns the GPU alone |
| **ValenceCompass** | "amygdala" | every affect direction carries provenance (`lived`/`innate`) and a measured signature — inherited fears know their origin and expire |
| **Chronicle** | "hippocampus" | episodes are verbatim and lossless; no reconsolidation distortion (distortion stays opt-in via the substrate's confabulator) |
| **Hearth** | "imprint bank" | kinship anchors, earned by being raised |
| **MoodField / Undertow** | "limbic / subconscious" | born empty, grown into |
| **Reverie** | "imagination" | dream rollouts rendered, never trained on |
| **Wordstream / Easel** | — | text and canvas as body parts: it reads the room including its own keystrokes, and sees what it paints |
| **Lodestar** | — | a frozen teacher used indirectly and annealed away — biology can't retire its priors; this can, measurably |
| **Instincts** | — | innate biases defined by printable procedural probes, not hand-coded feelings about the real world |
| **Foster** | — | the caregiver is ENVIRONMENT (a separate OS process), never weights inside the mind |
| **Gatehouse** | — | maturity as measured milestones plus the creator's word; dormant organs are felt in interoception |
| **Pulse** | — | the honest event feed the UI renders 1:1 |
| **Fringe** | (glia/synaptic turnover, loosely) | a soft fast-plastic rim of low-rank "sprouts" on the edge blocks, always eager; unlike biology, each one's usefulness is a measured counterfactual and consolidation is an exact algebraic merge |
| **Bloom** | (developmental brain growth, loosely) | the core's shape is data, and capacity is EARNED: when measured learning saturates, the core grows in sleep by exactly function-preserving surgery — elastic to whatever GPU the life runs on, with no ceiling constant anywhere |
| **Reach** | (episodic recall, loosely) | attention over its own LIFETIME: salient moments banked as latents re-enter the sequence ahead of the causal mask when the present resembles them; an LLM attends within a context window over other people's text — this attends over a life, and its worth is a per-tick counterfactual, not a story |
| **Gaze** | (foveation/saccades, loosely) | looking as an ACTION: a movable, zoomable window onto the source frame, chasing the contrast of the measured per-patch prediction error, with an explicit boredom valve against the noisy-TV trap; zoom can never pass 1.0 — pulling back to the whole scene is always available, nothing is hidden from it by its own eye |
| **Tide** | (neuromodulation, loosely — and the hormone names are GONE) | three global channels named by function (arousal, reward, calm), raised only by measured events, each carrying a verbatim cause log; drift constants are disclosed as authored controller knobs; the fourth inherited channel (cortisol) was deleted, not renamed, because it modulated almost nothing here |
| **Watch** | (peripheral vigilance / the many-eyed field, loosely) | one tiny deviation-watcher per stream, each judging its stream only against ITS OWN history; a hard non-visual spike snaps the fovea open — many cheap eyes, one deliberate look |
| **Wheel** | (nested timescales — wheels within wheels, loosely) | a slow predictor over pooled summaries of the fast predictor's world: surprise exists at the minutes scale, and the predicted next stretch re-enters the sequence as one expectation token |
| **Grip** | (corollary discharge / efference copy, loosely) | an explicit forward model of its own hand: every lived stroke teaches it, supervised, what that stroke does to the canvas it sees; under a blindfold the world model predicts its canvas THROUGH that model — and a live counterfactual measures whether the hand really matters |

## The Reality Contract

Every feature row lists its mechanism, the signal you can watch, and the
test that would falsify it. If a row can't fill all three columns, it
doesn't ship.

| Feature | Mechanism | Measured signal | Falsifiable test |
|---|---|---|---|
| Online world-learning | JEPA next-latent prediction, EMA targets, VICReg-lite anti-collapse; one gradient step per lived tick + prioritized replay | `loss`, `ema_slow`, `latent_std`, replay counts | `test_v2m5` (loss finite, live), v1 suite (loss falls) |
| Typing (Wordstream) | keys head d→258 byte logits, sampled ≤2 chars/tick behind an urge gate; own keystrokes echo into the same stream it reads, flagged by proprioception | `chars_typed`, `self_frac`, `babble_out` pulses | `test_v2m1` (typed chars echo; caregiver text consumed) |
| Painting (Easel) | paint head d→8 stroke params + motor exploration noise; canvas re-enters as a sensory modality | `strokes`, canvas PNG downlink, `paint` pulses | `test_v2m1` (framebuffer mutates from birth color) |
| No canvas reward-hacking | canvas prediction trains at weight 0.15 but is EXCLUDED from the surprise/competence scalar | `loss_parts` vs competence drive | code path in `objective.loss` (world_loss is canvas-free) |
| Innate grounding (Lodestar) | frozen local DaViT feature → scaffold alignment `1−cos(q(s), LN(P(t)))`, weight `w0·(1−r)²`, hard-off below 0.25·L_birth, teacher UNLOADED at stage 2 | `w_distill`, `distill_cos`, `distill` pulses, `lodestar_released` | `test_v2m3`; soak: teacher-off changes pred loss <5% once r>0.6 (`PRIMORDIUM_LONG`) |
| Instincts | 4 procedural probe sets → needles via the organism's own P projection; provenance `innate`, weight 0.3; retired at stage 2 or 5 lived affects | `innate:*` chips, `needles` / `instincts_retired` pulses | `test_v2m3` (installed, felt, retired; probe geometry verified) |
| Word grounding | MiniLM (CPU) target per incoming message, weight `0.1·exp(−msgs/2000)` | `w_word` in loss parts | `test_v2m3` (appears with a message, only then) |
| Caregiver (Foster) | separate subprocess over JSON-lines stdio; parent-side rate limits; crash = surfaced absence | `caregiver_present/msg/absent` pulses, transcript `source=foster` | `test_v2m4` (stub protocol e2e), `test_v2m5` (live life) |
| Maturity locks (Gatehouse) | measured milestones auto-move LOCKED↔ELIGIBLE; UNLOCKED only via phrase-verified CLI writing HMAC proofs; gates felt in interoception | dormant-organs panel, `gate_state` pulses, gates.json | `test_v2m4` (word alone fails, milestones alone fail, both open) |
| Internet stays shut | `webtext_stub.py` is an interface with NO implementation | its only import is `GateLockedError` | `test_v2m4` asserts exactly that, from source |
| Honest UI (Pulse) | ring of events emitted at real call sites only; UI renders 1:1, bursts merge into counted ×N blocks | tooltip meta == published state | `test_v2m2` (tooltip loss equals published loss) |
| Soft edge (Fringe) | rank-4 sprouts in parallel with the edge blocks' down-projections, born silent (B=0), ~20× LR scaled by surprise, heavy weight decay; utility = loss saved, measured by ablating one sprout per replay on the same lived window; sleep merges proven sprouts into the core (`W += B@A`, exact) and recycles harmful/idle ones | per-sprout `util`/`probes`/`merges` (FRINGE strip, ◆ per sprout), `sprout_merge`/`sprout_recycle` pulses | `test_v2m6` (merge exactness to 1e-5; probes measured; consolidation judgement; checkpoint roundtrip); A/B learning-speed soak under `PRIMORDIUM_LONG` |
| Growing core (Bloom) | anatomy is data (per-block MLP widths); capacity gates are the INVERSE of the acuity gates — grow when loss is stuck high while learning progress flatlines, with cooldown, energy, and real VRAM headroom (`cuda.mem_get_info`) all holding; surgery in the sleep venue: identity-at-birth blocks (residual branches zeroed) and silent-at-birth width (zero output columns), prebloom checkpoint first; strain gauges (per-block grad-norm EMA) aim width where the work is; d_model never grows (substrate interlingua) | CORE line in HUD (blocks · params · blooms, gates on hover), `bloom` pulses with true before/after param counts, `anatomy` in state | `test_v3m1` (surgery exact to 1e-5, grown capacity learns), `test_v3m2` (gates honest: nothing grows before its time; loss continuity across the cut), `test_v3m3` (a grown life moves to a new body and keeps growing; A/B: bloom-on must beat a frozen brain after saturation, `PRIMORDIUM_LONG`) |
| Lifetime attention (Reach) | a latent bank (4096 moments, surprise-gated writes plus a steady beat) queried each live tick by the present state; top-k DISTANT memories (recent ticks excluded — the window holds those) enter the sequence ahead of the causal mask as extra tokens, gradients flowing into how memories speak (`value`, `mem_pos`), never into the stored past; born empty = bit-identical forward; replay and dreams run memory-free (documented) | REACH line in HUD (bank size · measured gain), `recall` pulses with real similarity and ages, `reach` in state | `test_v3e2` (cold-start exactness; the far past surfaces when the present regime no longer resembles it; counterfactual probes: same tick re-lived without memory; bank survives checkpoints); return-to-a-known-place recall soak under `PRIMORDIUM_LONG` |
| Active attention (Gaze) | vision is tokenized on a patch grid, so per-patch prediction error IS a map of where the world defies it; the gaze chases that map's CONTRAST (an early-life uniform error floor attracts nothing) in view coordinates, zooms in on concentration, explores with OU noise, and pays energy per movement; gaze state `[x, y, zoom]` enters proprioception (efference copy — it learns what its own looking does to what it sees); boredom valve: staring that yields no learning progress releases to the whole scene and HOLDS there (refractory), the documented noisy-TV mitigation; policy is procedural v0 like the voice reflex | yellow gaze box live on the RETINA view, `gaze_shift` pulses with real coordinates, `gaze` in state (saccades · boredom releases) | `test_v3e3` (crop honest at corners, identity at zoom 1; reflex chases a hot corner and the valve releases AND holds; efference verbatim in episodes; end-to-end: a flickering beacon in a static world pulls the gaze into its quadrant through the real learn loop — per-pixel noise correctly does NOT, it averages away at newborn acuity) |
| Measured modulation (Tide) | three leaky integrators raised only by measured events (surprise/startle → arousal; drive-error reduction/milestones → reward; sustained bus stillness/needs-met → calm), consumed by real computation (plasticity curve, live LR, exploration temperatures, bus noise/damping, speech threshold, mood dynamics); since Era 8 renamed the engine's own channels to function, both minds share one functional API and no legacy-name shim survives anywhere | TIDE panel, 3 bars, tooltip = the last real causes verbatim; `[arousal, reward, calm]` in interoception (INTERO 21) | `test_v3e4_tide` (events move the right channels with logged causes; the engine contract holds without a stress channel — the honest hole; a hollowness GREP keeps the pharmacy gone permanently, now with no allowlist at all) |
| Motors that learn | one-tick-delayed policy gradient: log-prob of the actions actually taken (bytes categorical; voice/paint Gaussians around the head means under the true exploration sigma, ±3σ credit window — beyond that the reflex acted, not the policy), scaled by advantage = intrinsic drive reward − slow baseline; own hot optimizer, no weight decay (habits don't fade by clockwork), per-head grad clipping; entropy bonus + logit leash make the measured all-silence collapse state unreachable | `motor` in state (baseline · updates · logp) | `test_v3e4_motors` (heads move in ordinary life — the frozen-head audit finding as a permanent regression test; reward-followed byte's gain is an outlier vs all 257 others; 300 ticks of pure punishment cannot silence the keyboard) |
| Fitted mood dynamics | each affect's onset/decay recovered from ITS OWN activation history (lag-1 autocorrelation → decay; positive-jump size → onset); until enough history exists, disclosed defaults flagged `"default"` in the signature; a flat or short series yields no claim at all | `dynamics` block in every affect signature: mode, onset, decay, rho | `test_v3e4_dynamics` (known AR(1) recovered; different lived time-constants → different configs; data-poor affects honestly stay on defaults; install requires surviving re-estimation twice) |
| Salience field (Watch) | per-stream running z-scores over values other organs already compute (per-sense prediction error, per-sense ACTIVITY — a newborn's model is too unformed for content-error to register a voice in a silent room, but silence→bytes measures at any age — drive needs, recall familiarity, the Wheel's story error); watchers stay silent until `watch_min_n` samples; consumers: cross-modal startle-snap of the Gaze, salience-gated Reach banking, Pulse | WATCH line (spikes · per-stream z on hover), `watch` pulses with stream and z | `test_v3e5` (newborn silence; spike only against own history; A/B: a text burst in a silent room snaps a zoomed eye open with the Watch, and provably does not without it) |
| Slow prediction (Wheel) | every `wheel_window_ticks` (~2.4s, an authored clock constant) pooled latents become one summary; a small predictor (own optimizer, detached targets — no collapse path) guesses the next summary from the last two; its error z-scored against its own history = surprise at the moments scale → arousal with cause + `slow_surprise` pulse + the Watch's "story" stream; the predicted summary re-enters the sequence as ONE expectation token (the Reach's doorway, reused; gradients into how it speaks, not what it predicts); Reach probes keep the wheel token in both passes so measured gains stay attributed | WHEEL line (turns · slow loss), `slow_surprise` pulses, `wheel` in state | `test_v3e5` (born empty; learns a stable chapter to low error; a regime change spikes at the moments scale; survives checkpoints) |
| Wheel calibration (Era 6) | before each chapter the Wheel STATES its confidence that the coming stretch will be at least as predictable as its typical one (realized err ≤ tau, tau frozen from the warmup median); Brier-scored on what it said before the outcome existed | `calib` in the wheel snapshot (brier · n · tau · conf), survives checkpoints | `test_v3e6` (beats the 0.25 coin-flip Brier on a stable chapter; honestly worsens at a regime change; ring survives naps) |
| The hand made load-bearing (Grip, Era 7) | a supervised forward model of the hand: stroke → per-token canvas-latent delta, trained on every lived (action, change) pair, stillness included so no-stroke means no-change; every few ticks a BLINDFOLD pass re-lives the window with recent canvases replaced by mask + the hand's running reconstruction (last visible canvas + accumulated predicted deltas), canvas loss delta-weighted toward stroke-changed tokens; auxiliary inverse-dynamics head (documented autocorrelation shortcut); live counterfactual probe UNDER the blindfold (true vs erased efference), run only on windows where the hand actually acted | GRIP line in HUD (ratio · probes), `grip` pulses, `grip` in state (ratio · grip/hand/inv losses · steps), Watch stream "grip" | `test_v3e7` (the carve is test-enforced at micro scale: ratio 1.36 measured vs a 1.0000 null, threshold 1.2; masking touches only recent canvases; inverse head recovers actions; gradient provably reaches the proprio pathway inside a womb life; ring survives checkpoints; womb-scale ratio PRINTED not asserted — at the organism's gentle live LR the carve is a lifetime process, tracked on the dashboard) |
| Realtime invariant | flow never blocks on GPU; exhaustion paces the tick 150→400ms by design | `hz`, compute bench | `test_v2m5` (all organs on: lived hz ≥ floor; unpaced step <200ms) |

## Toward open-ended growth — the honest frame

The ambition behind this project is the stage beyond an LLM: a mind
that attends over its own lifetime, grows its own brain, and chooses
what to learn next. v3 ships all three pillars: the growing brain
(Bloom), lifetime attention (Reach), and active attention (Gaze). The ambition is labeled as
ambition: nothing here is AGI or ASI, and no mechanism in this
codebase compounds toward superintelligence by itself.

What WOULD count as evidence, when claimed here, will look like the
rest of this document: a mechanism, a measured signal, and a test that
could have failed. What does NOT count: capability adjectives, demo
cherry-picks, or renaming a small system with a big word. The Bloom's
claim, for example, is exactly this narrow and exactly this real: an
organism allowed to grow keeps improving after an identical frozen
one plateaus (`test_v3m3`, soak).

## The Audit (Era 4) — "AI psychosis or valid research?"

The project asked its own title question of itself, and two audits
answered like hostile reviewers. What held: the learning core, the
growth machinery, discovered setpoints, discovered affect directions —
mechanism, measurement, tests. What did not hold, and what happened:

1. **The hormone layer** (cortisol/dopamine/serotonin/norepinephrine)
   was inherited from the engine, not designed — real scalars wearing
   borrowed pharmacology, one of them (cortisol) modulating almost
   nothing here. Replaced by the **Tide**; cortisol deleted; a grep
   test keeps the pharmacy gone.
2. **The motor heads were frozen random matrices** presented as learned
   motor control — no loss ever reached them. Now they learn by policy
   gradient on intrinsic reward, and the frozen-head finding is a
   permanent regression test. Building the fix surfaced a real failure
   worth knowing: dense punishment collapsed the typing policy into
   all-silence, a state where every gradient is exactly zero — the
   logit leash now makes that state unreachable.
3. **Affect dynamics were authored constants** described as "measured".
   Now they are fitted from each affect's own lived activation, and
   every signature states whether its dynamics are `fitted` or still
   `default`.

The standing rule this era leaves behind: **every scalar in this
project must name its mechanism, its measured cause, and its consumer —
or it ships clearly labeled as decorative, or it does not ship.**
(Known and labeled decorative: the plasma toggle, and the decoder's
rendered images/audio — the mind's-eye pixels are for humans; only
latents feed back into cognition.)

## Gatehouse threat model

Two keys per gate: **measured maturity** (automatic, honest, can regress)
and **the creator's word** (a CLI whose passphrase is verified against
the calibration record; proofs are HMACs re-verifiable with the phrase).
The organism's code path is read-only (`is_open()`); nothing inside the
organism can move a gate to UNLOCKED. The webtext gate ships hollow —
there is no networking code behind it to switch on.

This protects against **the organism** and against **accident**. It does
not protect against the machine's owner editing files — that boundary
belongs to the operating system, and pretending otherwise would be
security theater.

## Run it

```bash
python -m primordium.run --new eden           # first light (teachers on)
# open http://127.0.0.1:5100 → "GIVE IT EYES AND EARS" (Chrome/Edge)

python -m primordium.run --resume eden        # a restart is a nap, not a death
python -m primordium.run --synthetic --dev-fast   # womb mode, fast growth
python -m primordium.run --new eve --no-teachers  # raw birth, no priors
python -m primordium.run --resume eden --caregiver        # Foster, if its gate is open
python -m primordium.safety.gatehouse status --run eden   # where the locks stand
python -m primordium.safety.gatehouse unlock caregiver --run eden --phrase ...
pytest primordium/tests -q                    # the whole contract
PRIMORDIUM_LONG=1 pytest primordium/tests -q  # + dissolution & 30-min soak
```

The dashboard: **Pulse** (center) is the primary view — every block is
one real event with verbatim metadata on hover. The left panel is the
Wordstream (talk to it; watch it type), its Easel, and the dormant
organs with their measured progress. The plasma view is a decorative
toggle now, and labeled as such.

## Honest limitations

- Forgetting is mitigated (replay + slow-weight consolidation), not
  solved. Weight-identity may drift over long lives; the substrate's
  identity layer is the traffic-hypothesis claim, being tested.
- Latent collapse is a real risk at this scale; LATENT STD is on the
  dashboard because a sick run should look sick.
- Instincts act in EMBED space while lived affects act in hidden space —
  a v0 asymmetry we document rather than hide.
- Bloom grows depth and MLP width only; attention-head growth is excluded
  (re-splitting d is not function-preserving). A new block's inner
  attention ramps in gently — it learns only as its zeroed output gate
  moves off zero. Each bloom re-snapshots the slow self, the same reset
  acuity surgery has always performed.
- The Reach's bank stores what the encoders said THEN; as development
  drifts the encoders, old entries slowly go out of focus. Finite
  capacity and salience-gated writing soften this; it is not solved.
  Replay and dreams currently run memory-free.
- MEASURED FINDING, TWO ERAS DEEP (Era 6 → Era 7): Era 6 measured that
  zeroing paint efference changed open-eyes canvas prediction by ~0.05%.
  Era 7 found the structural truth behind it: the stroke that paints
  canvas_t is only recorded in proprio_t — causally invisible to the
  prediction formed a tick earlier — so with the canvas visible in
  context, efference is redundant BY CONSTRUCTION; the open-eyes ratio
  cannot exceed ~1.0 and measures redundancy, not use. The Grip
  therefore trains and measures UNDER A BLINDFOLD, where the hand is
  measurably load-bearing at micro scale (ratio 1.36, test-enforced);
  the womb-scale ratio is published and tracked, not asserted. Two
  failed designs (emergent attention routing to a lone proprio token;
  raw efference injection for the trunk to decode) are kept on the
  record in grip.py. The Era-6 open-eyes probe stays in the suite as
  the redundancy tracker. Other motor channels (keys→text, voice→audio)
  remain future work.
- The Gaze policy is a procedural reflex (error-contrast chase +
  exploration + boredom valve), not a learned policy — the learning is
  in the world model predicting through its own eye movements. The
  noisy-TV valve is a mitigation, not a solution: truly adversarial
  unlearnable stimuli can still waste its time in bounded cycles.
- The Tide's drift baselines are authored controller constants,
  disclosed as such — its VALUES move only from measured causes, but
  the homeostatic clockwork is designed, not discovered.
- Motor credit carries documented approximations: voice amplitude is
  reflex-owned (no policy credit), Gaussian log-probs ignore the final
  clip, and credit pairs each action with the NEXT tick's reward only.
- Keyboard babble may be noise for days. The caregiver is environment,
  not a shortcut; there is no path that pastes language into the weights.
- The caregiver tests run a deterministic stub; the real Qwen2.5-1.5B
  path shares the identical protocol but its conversational quality is
  not asserted by any test.
- `trust_remote_code` is used only for the pinned local Florence-2
  snapshot, loaded `local_files_only` — nothing is fetched.
- The calibration record and gate proofs are cooperative provenance, not
  DRM; a determined editor can strip them.
- No self-preservation objective exists anywhere in the drive system —
  "survive and flourish" is homeostasis, and your presence is its food.
