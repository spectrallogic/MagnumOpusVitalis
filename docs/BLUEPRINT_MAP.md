# Blueprint coverage map

External audit: "Cognitive Runtime Evolution Blueprint" (baseline 30cbcff,
2026-07-10). Every backlog item mapped to its status here. Statuses:
**built** (existed before the blueprint, organ named) · **this-era**
(Era 6 the Ledger · Era 7 the Grip) · **deferred** (future era, reason
given).

## P0

| Item | Status | Where / why |
|---|---|---|
| ADR-001 where the mind lives | this-era | docs/adr/ADR-001 |
| ADR-002 epistemic types | this-era | docs/adr/ADR-002 |
| Baseline harness (prediction/latency/retention metrics) | partial-built + deferred | smoke_test (23 checks), primordium suite (62), bench_tick, lived-rate + compute-capability asserts (test_v2m5); a unified cognition-metrics harness is deferred to the benchmark era |
| Event journal + deterministic replay | partial-built + deferred | Pulse is an honest event feed and Tide/gates carry cause logs, but no replayable append-only journal; deferred — big surface, wants the CognitiveState era |
| Typed bus shim (source/type/confidence on writes) | this-era | bus provenance ledger (bus.py), all write surfaces |

## P1

| Item | Status | Where / why |
|---|---|---|
| Forecast ledger + Brier/ECE | this-era | magnum_opus_v2/forecast.py (speculation futures, resolved against the situation vector); Wheel calibration ring (primordium) |
| Action token audit | this-era | ADR-002 audit section (PROPRIO 26, one-tick delay) |
| Fast action-conditioned dynamics ensemble | this-era (Era 7) + deferred | the Grip (primordium/loom/grip.py): an explicit SUPERVISED forward model of the hand (stroke → canvas-latent delta, trained on every lived pair), consumed by the world model under a BLINDFOLD — recent canvases replaced by the hand's running reconstruction — with a live counterfactual probe, delta-weighted loss, and an auxiliary inverse-dynamics head. Era 6's negative finding got its structural explanation: the newest stroke is causally invisible to one-step prediction and visible canvas makes efference redundant BY CONSTRUCTION, so the open-eyes probe measures redundancy; USE is measured blindfolded, where the carve is test-enforced at micro scale (ratio 1.36 vs a 1.0000 null, test_v3e7) and lifetime-tracked in the womb (probe on the dashboard). Two failed designs (emergent attention routing; raw efference injection) are on the record in grip.py. Deferred: ensembles/uncertainty bands, and the other motor channels (keys→text, voice→audio) |
| Memory contamination guard | this-era | consolidation + self-model guards; test asserts block AND preserved undertow participation |
| Entity persistence benchmark | deferred | wants entity slots first (abstraction era); a raw occlusion probe without entity representations would measure pixels, not permanence |

## P2 (all deferred, ordering per the blueprint's own phasing)

| Item | Reason to wait |
|---|---|
| Event graph (entities/relations/temporal links) | abstraction era; requires entity slots + event segmentation (Watch surprise boundaries are a ready ingredient) |
| Goal/Skill/Plan/Action contracts | planner era; blueprint itself orders planner AFTER calibrated forecasts — calibration lands this era |
| MPC planner over skills | same |
| LLM cognitive packet + validator | language-organ era; ADR-001 fixes the boundary now |
| LLM swap benchmark | language-organ era; note: V2 profiles are per-model (vectors extracted per model), so "swap" must be defined over the runtime state, not the steering vectors |

## Blueprint claims that were ALREADY built (with their organ names)

| Blueprint requirement | Existing organ / mechanism |
|---|---|
| Mechanism + signal + falsifiable test discipline | the Reality Contract (primordium/README) |
| Counterfactual usefulness measurement | Reach ablation probes; Fringe sprout trials; Bloom A/B soak |
| Dreams never train the model | long-standing rule (reverie/minds_eye isolated; ADR-002 table) |
| Active sensing driven by prediction error | Gaze (error-contrast chase, boredom valve, startle snap) |
| Slow/multi-horizon prediction seed | Wheel (summaries at ~2.4s; hierarchical depth deferred) |
| Fast plasticity + conditional, reversible growth | Fringe (merge-by-proof), Bloom (measured gates, prebloom checkpoints, elastic anatomy) |
| Modulation with measured causes, not chemistry cosplay | Tide (cause-logged channels; hormone layer deleted in Era 4) |
| Salience across streams | Watch (per-stream z, cross-modal startle) |
| Provenance on affects | ValenceCompass provenance=lived/innate + fitted dynamics flags |
| Local-first safety, gated capabilities, no self-preservation objective | Gatehouse (dual-key), drives design rule, 127.0.0.1-only |

## Anti-pattern self-check (blueprint §Anti-patterns)

Era 6: no new named brain regions were added; no new untyped bus writes
(the opposite: all writes now carry provenance); no text rollouts
renamed as world models; dreams still untrained-on; the Ledger's
calibration replaces one scalar's pretensions with measured meaning.

Era 7 added ONE organ (the Grip) under the Reality Contract's full
discipline — mechanism, measured signal, consumers, falsifiable test —
and kept its two failed designs on the record instead of erasing them.

Era 8 (the Second Reckoning) extended Era 4's rule to the ENGINE: the
four modulation channels renamed to function (stress/reward/calm/
arousal — mechanisms preserved exactly, pharmacology gone repo-wide,
the Tide's legacy shim deleted); the face UIs now obey bind-or-remove
(pure-noise decoration deleted, base rates zeroed, all motion gated on
a live stream, presence motion disclosed in-page); the two unrendered
honesty instruments (bus provenance, forecast ledger) got dashboard
panels. Deliberate non-goal, still on the record: Limbic keeps its
name (anatomy metaphor, no claimed-absent mechanism, crosses the
substrate boundary).
