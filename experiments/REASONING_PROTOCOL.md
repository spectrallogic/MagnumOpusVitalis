# Objective overlay pilot, 2026-09-09

This protocol is written before running the scored model comparisons. It asks
whether Vitalis improves objective answer accuracy on a small fixed task set.
No tuning of engine coefficients, prompts, or scoring follows observed results.

- Models: locally cached Qwen/Qwen2.5-1.5B-Instruct and
  HuggingFaceTB/SmolLM2-1.7B-Instruct, sequentially on the same GPU.
- Data: 64 procedurally authored questions, 16 each in arithmetic, state
  tracking, ordering constraints, and conversation-memory updates. Generator
  seed 20260909. Save dataset and source hashes before scored runs.
- Conditions: unmodified base generation; full live overlay with seeds 11 and
  23; controller-on/steering-off control with seed 11 on the first four cases
  of each category (16 preselected control cases per model).
- Each condition sees the exact same chat-template input tokens. No persona
  or extra system instruction is added. Both arms see all supplied history.
  Earlier assistant acknowledgements are fixed, not model-generated.
- A fresh engine is constructed per case and repetition. Supplied history
  is replayed through perception and copied into conversation history. The
  overlay receives one second of background warmup before the scored turn.
  All default regions, rumination, and search budgets remain enabled.
  Autonomous speech is suppressed so it cannot insert unpaired extra turns.
- Greedy decoding, maximum 384 answer tokens in every condition. The controller
  retains its own stochastic dynamics. Wall-clock scheduling prevents exact
  deterministic replay; two overlay runs check some of that variability.
- Steering-off control suppresses all vectors returned by the hook while
  retaining the controller and its extra work. Its greedy answer should match
  the base model; disagreements are reported as a control-fidelity failure.
- Ground truth is computed without a model. The scorer accepts a FINAL line,
  a boxed answer, an explicit 'answer is' line, or a bare final line. It never
  searches for the expected answer inside an explanation. Format adherence,
  empty answers, truncations, and runtime errors are reported separately.
- Primary outcome: exact normalized answer accuracy, plus overlay-minus-base
  differences paired by question. Report both overlay seeds separately and
  their mean. Bootstrap intervals resample questions, keeping repetitions
  together; repeated templates limit population generalization.
- Record every response, model/config/profile/source/dataset identity,
  primary prompt hashes, generation latency, complete turn latency, forward
  calls, processed token positions, flow ticks, search rounds, and failures.
- The plain baseline receives less compute. These comparisons test whether
  the overlay helps at all with its additional work; they do not establish
  an advantage at matched compute. Steering-off is an ablation, not an
  independently optimized use of the same compute budget.
- Report all results, including regressions and inconclusive findings. This
  is an authored pilot, not a standardized intelligence benchmark. It tests
  brief episodes, not long-lived adaptation, broad intelligence, or sentience.

Harness smoke checks use a separate dataset seed. Calibration uses the existing
contrastive prompts, never these task answers. No model weights are trained.

Before the scored runs, the separate seed-901 plumbing smoke revealed that the
initial 128-token cap truncated the state-tracking and ordering responses in
all conditions. The cap was increased uniformly to 384 before any scored
64-question run. Task prompts, answers, and scoring were unchanged. Both smoke
runs are retained separately and excluded from the scored pilot.
