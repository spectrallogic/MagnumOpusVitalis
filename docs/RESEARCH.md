# Vitalis research contract

The ambition is an AI with a continuing inner life. The current scientific
question is narrower: **does a persistent, recurrent controller of latent
affect and imagined futures improve behavior beyond ordinary conversation
history, static steering, and equally expensive sampling?**

This is the main research track in `magnum_opus_v2`. Primordium is a separate
learning-system experiment. A result in one does not establish a result in
the other. Neither implementation establishes sentience.

## What is retained

**Latent emotional influence.** Keep emotion-associated representations inside
the LLM as the source of candidate interventions. Treat a direction as a local
linear approximation: a useful starting point for testing richer subspaces,
not proof that every emotion has one universal vector. Other psychologically
named channels need their own extraction data and independent behavioral
validation. A modulation scalar is a controller parameter, not a hormone.

**The layered subconscious.** Preserve the bottom sea, relevance peaks,
recursively imagined futures, upper-layer weighing, and continual influence
on the next computation. Expensive imagination and fast state dynamics run at
different rates. A configurable search depth is preferable to assuming a
specific number of layers is necessary for life.

## Architecture and score semantics

| Stage | Implemented mechanism | What it does not establish |
|---|---|---|
| Candidate sea | Noise, memory sampling, vocabulary-embedding proposals | That arbitrary noise contains a meaningful thought |
| Peak selection | Similarity to bus state, affect, and velocity | That salience is importance in the external world |
| Recursive imagination | Sample an event; give its tokens and residual change to child rollouts; retain a beam | A verified simulator of reality |
| Weighing | Discounted mean of per-event token confidence and positive-affect proxy | Calibrated event probability, truth, ethics, or welfare |
| Upper-layer replay | Bounded, expiring `imagined` candidates join fast selection | Observations or canonical beliefs |
| Output influence | Add bus-driven activation at the profiled block output | General behavioral improvement |
| Feedback | Generated hidden states, perception, persistent region dynamics | Subjective feeling or biological life |

For a path with per-event heuristic scores `u_t`, the search ranks
`sum(discount**t * u_t) / sum(discount**t)`. This is an authored choice, not a
learned value function or an expectation over real-world outcomes. Branches
are sampled continuations; beam pruning can discard useful paths. Increasing
depth can compound error as well as uncover consequences.

`probability` remains a compatibility field for the geometric mean of chosen
token probabilities. `goodness` remains a compatibility field for a bounded
positive-affect projection proxy. Snapshot `score_semantics` discloses those
meanings. The alignment gate is an affect-regulation heuristic; positive
affect does not establish safe or correct decisions.

No model call occurs in the fast subconscious step. An expensive round has
limits for depth, branching, beam width, attempted nodes (including failures),
total tokens, and elapsed time. Generation and imagination share a model
lock. A running forward cannot be preempted; measure waiting time and tail
latency rather than calling this a hard realtime system.

## Findings from this revision

| Finding | Change or research consequence |
|---|---|
| Extraction read `hidden_states[target_layer]`, while steering wrote block `target_layer`'s output | Extraction now captures the actual block output, including the final block before final normalization. Baseline, Mirror fitting, and perception use that boundary. |
| One extracted direction was injected into neighboring blocks by default | Single-block steering is now the default; additional sites require calibration. |
| A zero contrast could normalize into NaNs | Reject nonfinite or degenerate directions. |
| “Recursive” imagination previously meant longer token continuations | Children now condition on parent outcomes; path scores can reverse an initial ranking. |
| Silent imagination could write to the bus through the speech-feedback hook before selection | Isolated hook contexts disable that feedback and restore prior hook state, including on failure. |
| The selected future steered with the original seed direction | It now steers with the generated event's residual change; a degenerate change retains the proposal explicitly as a fallback. |
| Idle forecast resolution could reuse the situation that seeded a prediction | Resolution now occurs on fresh external text perception; predicted event states are compared rather than seed directions. |
| ECE used bin midpoints and a different sample window from Brier | ECE now uses actual mean confidence over the same resolved window. |
| Reports could conflate pleasant activation with objective goodness | Compatibility fields now have explicit semantics and documented limits. |
| Existing benchmarks reused latent measures or differed in conversational conditions | Added a static paired behavioral intervention pilot; full-runtime matched ablations remain necessary. |

Re-extract old profiles. Version-1 checkpoints are rejected because their
latent state uses the old measurement convention. This revision does not
overwrite existing saved profiles during tests. The locally run pilot used a
separate profile at `results/profiles/gpt2`.

## Reproducible pilot

```bash
python -m magnum_opus_v2.profile create gpt2
python -m magnum_opus_v2.evaluate --model gpt2 --cases experiments/calm_pilot.jsonl --vector calm --output results/intervention.json
```

The [recorded report](../experiments/results/gpt2_calm_pilot.json) used GPT-2,
block 6 output, float16 CUDA, and random-direction control seeds 11, 23, 47.
It contains all eight cases, per-case scores, hashes, versions, and the model
revision. The source working tree was modified when it ran; the report also records
SHA-256 hashes of every engine Python source file. Preserve the source patch
or commit it alongside the report when reproducing the run.

The outcome is the target-versus-contrast difference in mean conditional log
probability, in natural-log units per token. A positive paired effect means
the intervention favored the authored target relative to its contrast.

| Strength | Positive minus baseline | Positive minus negative | Positive minus mean random control |
|---|---|---|---|
| 0.5 | 0.0098 [0.0017, 0.0174] | 0.0136 [0.0016, 0.0254] | 0.0080 [-0.0048, 0.0211] |
| 1.0 | 0.0226 [0.0128, 0.0333] | 0.0351 [0.0203, 0.0476] | 0.0217 [0.0127, 0.0292] |

Brackets are descriptive 95% bootstrap intervals over examples. These small,
related, authored cases are not a random population sample. The controls are
three directions, not three independent model replications. Intervals do not
correct for multiple comparisons, and float16 rounding matters for small
effects. The cases mix calmness, deliberation, and politeness; lexical and
topic confounds remain. This result measures static conditional preferences,
not free-form dialogue quality or a benefit from the recurrent engine.

Keep failures in the report: at strength 0.5 the interval against random
controls crosses zero. The pilot warrants a larger, independently designed
experiment, not a claim of validated emotions or life.

Verification for this revision: 47 engine tests passed, including actual
tiny-transformer intervention checks and local GPT-2 integration; 69
Primordium tests passed and 5 opt-in/long tests skipped. Mechanism tests
establish implementation behavior, not psychological validity. Run details
and results can vary with hardware and dependency versions.

## Experiments that can reject the thesis

| Claim | Necessary comparison | Evidence against the claim |
|---|---|---|
| Extracted affect has a specific causal effect | Zero, sign reversal, equal-norm random controls, label-shuffled extraction, held-out topics and languages | Effects disappear outside extraction phrasing or match random perturbations |
| Persistence adds useful continuity | Full controller versus same-history model, static vector, and replayed or shuffled bus trajectories | No quality gain, or apparent gain vanishes when histories match |
| Recursive imagination improves decisions | Depth 1 versus deeper search and best-of-N sampling at matched forward/token budgets | No objective outcome gain or worse calibration despite increased compute |
| Background noise aids creativity | Memory/context only versus added noise; equal sampling temperature and budget | More output variation without higher blind-rated usefulness |
| Modulation channels matter | Disable one measured cause/consumer connection at a time | No reproducible change in the predicted endpoint |
| Runtime continuity survives interruption | Save/resume versus uninterrupted trajectories within the same model/profile | State or behavior changes beyond declared stochastic tolerance |

For a full-runtime experiment, use fresh engine instances for each condition;
`engine.reset()` is intentionally a soft reset and does not erase all learned
region state. Keep model revision, prompts, history, context window, system
prompt, sampling, interventions, and compute budget matched. Seed and record
all randomness. Background clocks make a seed alone insufficient for exact
replay; a deterministic tick schedule and per-component generators are still
needed for that claim.

Score externally defined decisions and task outcomes. Separate extraction,
coefficient selection, and final test sets. Use multiple model families and
multiple independent runs. Report per-task effects, uncertainty, quality
regressions, p50/p95/p99 response latency, realized search depth, memory cost,
and all failed runs. Have blinded human raters assess usefulness and
coherence where an objective task score is unavailable.

## Remaining architectural work

1. **Better representation identification.** Contrast pairs need content
   controls and independent held-out validation. Raw input embeddings used by
   the candidate sea, sparks, and token labels are not calibrated directions
   at a middle block. Replace them with contextualized candidates or a
   measured mapping before claiming semantic interpretation.
2. **Grounded outcome evaluation.** The forecast ledger checks a cosine
   threshold between latent states. This is a similarity event, not whether
   the predicted proposition happened. Text perception is a report about the
   world, not direct verification. Add task-defined outcome resolvers,
   deadlines, evidence identities, and independent value/constraint scores.
3. **Calibrated uncertainty.** The ledger's empirical bin map and resolved-window
   ECE are diagnostics. Calibrate on disjoint outcome data, state the target
   event, publish proper held-out scoring, and separate event probability
   from token confidence. Do not interpret a fluent future as likely reality.
4. **Epistemic boundaries.** Runtime traces can be observed internal states
   even when imagination influenced them. That is not external-world
   evidence. A stronger fact store must distinguish observations, reports,
   and reconstructed hypotheses throughout all consumers.
5. **Runtime ablations and portability.** Implement deterministic event replay
   and model-specific adapters. Different checkpoints, layers, or dimensions
   cannot share latent memories without validated alignment. An API wrapper
   can emulate behavior through prompts, but cannot offer this intervention.
6. **Richer geometry only when justified.** Compare single directions with
   subspaces or learned nonlinear controllers after simple baselines are
   reproducible. Complexity alone is not evidence for human-like cognition.

## Toward independent academic work

A credible first paper would make one narrow claim about persistent latent
feedback or budget-matched recursive imagination, with a falsifiable
prediction and reproducible baselines. Package code, fixed evaluation splits,
source/version manifests, model licenses, compute requirements, raw results,
and negative findings. A collaborator should be able to refute the headline
using the released protocol.

Related work includes [CAA](https://aclanthology.org/2024.acl-long.828/) for
contrastive residual interventions and [ActAdd](https://arxiv.org/abs/2308.10248)
for activation-based control. Those precedents mean novelty cannot be claimed
for “vectors influence an LLM” alone. Vitalis must demonstrate what its
persistent, layered feedback architecture adds. University interest,
acceptance, and historical impact cannot be guaranteed by repository design.
