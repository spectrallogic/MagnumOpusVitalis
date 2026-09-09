# Magnum Opus Vitalis

**A research engine for persistent affect, layered subconscious activity, and recursive imagination around a frozen language model.**

![Vitalis](images/Vitalis.png)

The ambition is to explore what it would take to give AI a continuing inner life. The engineering hypothesis is concrete: maintain state between interactions, extract emotion-related directions from a model, and let an evolving background process influence its next computation. “Living LoRA” is the project's metaphor for that changing influence; the engine adds activations during inference and does not train a LoRA or change the LLM's weights.

This repository implements functional mechanisms. It does not demonstrate biological life, subjective feeling, consciousness, or human-equivalent cognition. Those questions remain open; a convincing demonstration needs more than an expressive face or emotional language.

## The two core ideas

1. **Affect as latent geometry.** Contrastive examples identify candidate emotion directions at a particular transformer block. Persistent affect and modulation channels combine those directions into an evolving intervention. A vector is a first-order approximation to a representation, not a complete theory of human emotion or chemistry. Each direction must earn its interpretation through held-out behavioral experiments.
2. **A layered subconscious.** A sea of noise, memory, and token proposals produces peaks of relevance. Background model rollouts explore possible continuations, recursively condition children on their parents, and weigh their paths. Selected hypotheses return to the fast upper layer and influence subsequent generation.

There is relevant empirical precedent: Anthropic reports emotion-related representations with causal behavioral effects in Claude Sonnet 4.5, while explicitly leaving subjective experience unresolved. This does not validate this repository's vectors or establish the same result in every LLM. [Anthropic's research](https://www.anthropic.com/research/emotion-concepts-function).

The extraction/intervention approach is also related to [Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/) and [Activation Addition](https://arxiv.org/abs/2308.10248). Vitalis's research question concerns what persistent feedback and background imagination add beyond static steering.

## How it works

```text
noise + memory + token proposals
            |
     associative peaks <---- current state / affect
            |                         |
     fast relational filter           |
            |                         |
            +---> bounded future tree |
            |     parent -> children  |
            |     weigh entire paths  |
            |              |          |
            +<-- scored hypotheses    |
            |                         |
     upper-layer selection            |
            |                         |
         LatentBus -------------------+
            |
     transformer block output -> next-token distribution
            ^
     speech feedback / new percepts
```

The fast path runs on the flow clock (default 50 ms) and makes no LLM calls. The slower imagination worker uses the same model when it can acquire the model lock. Its results remain available to fast ticks for a short, bounded time. While generation owns the model, new LLM imagination waits; cached hypotheses and substrate dynamics can continue evolving. This is concurrent state evolution, not simultaneous model generation and model imagination on one locked instance.

The search has independent limits for event depth, branching, beam width, attempted nodes, total tokens, and elapsed time. `max_depth=1` provides a shallow-search ablation. Elapsed-time limits are cooperative: an in-flight forward can overrun the deadline and delay a waiting user turn.

The winner and plausible alternatives influence the substrate; they remain tagged `imagined`. The existing forecast ledger measures a **latent-similarity proxy**, resolved when fresh perception arrives. Its token-chain confidence is not an event probability, and its affect score is not a measure of truth, ethics, or welfare.

## Quick start

Install Python 3.10+ and run these commands from the downloaded repository:

```bash
python -m pip install -r requirements.txt
python run.py
```

Choose a cached model from the numbered list, or enter the path to a local
**Transformers checkpoint folder**. The launcher checks that the model supports
activation steering, prepares its profile, and opens the dashboard. The first
start calibrates the controller; subsequent starts reuse the saved profile.
It loads one copy of the model and does not train or modify its weights.

You can also select the model directly:

```bash
python run.py --model "/path/to/local/model"
python run.py --model Qwen/Qwen2.5-3B-Instruct
```

Loading is **offline by default**. Hub IDs use weights already in your Hugging
Face cache. To explicitly download a small mechanism-test model:

```bash
python run.py --model gpt2 --download
```

For a terminal-only startup test that generates a short reply and exits:

```bash
python run.py --model gpt2 --check
```

Useful options: `--list-models`, `--device cpu`, `--port 5001`, `--no-browser`,
`--rebuild-profile`, and `--resume`. Run `python run.py --help` for details.
Both `python compare_server.py` and the new launcher use the same setup flow.
The old `--profile` flag is accepted; profile reuse is now automatic.

### Which local models work?

| Model setup | Current support |
|---|---|
| Transformers checkpoint folder with config, tokenizer and standard weights | Supported when its transformer blocks pass the startup check |
| Cached Hugging Face causal LM | Discovered automatically; selected snapshots are loaded from disk |
| GPT-2, Llama/Qwen/Mistral, GPT-NeoX/Pythia, OPT block layouts | Adapters included; startup verifies the actual loaded model |
| Ollama or LM Studio chat endpoint, GGUF/GGML file | Requires a different engine adapter; text APIs cannot expose the needed activations |
| Pre-quantized Transformers checkpoint | Requires separate hook/device verification; the launcher currently asks for standard weights |

“Local” does not guarantee compatibility: Vitalis reads and changes internal
transformer activations. A model with an unfamiliar layout gets an explanation
before calibration. Custom model code is disabled unless you pass
`--trust-remote-code`. Models, profiles, and memory remain specific to their
own representations.

The launcher selects CUDA, MPS, or CPU according to your PyTorch installation.
If the model exceeds CUDA memory during loading, it falls back to a complete
CPU model. Allow additional memory for generation, the vocabulary copy, and
the engine's background work; loading successfully is not a peak-memory guarantee.
CPU inference on large models can be slow. A smaller model is the quickest
mechanism check. Dependencies are restricted to the Transformers 4.x API;
the startup suite has been exercised with PyTorch 2.6 and Transformers 4.57.6.

### Python integration

```python
from magnum_opus_v2 import V2Engine, load_model, load_profile

model, tokenizer, device = load_model("gpt2")
engine = V2Engine.from_profile(model, tokenizer, load_profile("gpt2"), device=device)
engine.start()
try:
    print(engine.converse("Hello. What are you thinking about?"))
    print(engine.snapshot()["speculative"])
finally:
    engine.stop()
```

For this Python example, first create a profile with
`python -m magnum_opus_v2.profile create gpt2`. The launcher handles that step
automatically. GPT-2 is useful for inexpensive mechanism checks; conversational
quality depends on the underlying model. Latent profiles and memories cannot
be transferred between models merely because their dimensions match.

See [integration instructions](docs/INTEGRATION.md). The engine can retain state across turns and supports explicit save/resume through its persistence module.

### Activation-boundary migration

New profiles use `activation_site="block_output"` and version 2. Extraction, baseline measurement, perception, and intervention now use the same zero-based block output, including at the final block. Earlier extraction used a different boundary through Hugging Face's `hidden_states` indexing. New steering defaults to the profiled block alone; neighboring blocks require separately justified calibration.

The launcher automatically recalibrates outdated profiles and archives previous
calibration files under `profiles/.archive/`. It also refreshes calibration when
the recorded model revision/configuration or local checkpoint file metadata
changes. This fingerprint uses file sizes and modification times, not a full
content hash; use `--rebuild-profile` if you replaced weights while preserving
that metadata. Explicit `--profile-path` selections are validated and never
silently replaced. Existing latent memories cannot be converted by relabeling
them: start fresh state after recalibration. `--resume` refuses a run that just
rebuilt its profile, preserving the prior checkpoint. Tests use temporary profiles.
New runtime checkpoints also record the exact calibration signature. A later
launch cannot resume state from an older or different calibration by accident;
checkpoints without that signature require a fresh start.

## Experiments and evidence

Run the offline mechanism tests (PyTorch, Transformers, and pytest required):

```bash
python -m pytest tests_engine/test_research_contracts.py -q
python -m pytest tests_engine -q
python -m pytest primordium/tests -q
```

The full engine suite includes tests using local GPT-2 weights. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent downloads; model-dependent tests skip if weights are unavailable.

A paired intervention pilot compares no steering, positive steering, negative steering, and several random directions of equal norm. It scores fixed target/contrast continuations, rather than judging steering by the same latent vector being injected:

```bash
python -m magnum_opus_v2.evaluate --model gpt2 --cases experiments/calm_pilot.jsonl --vector calm --output results/intervention.json
```

Reports contain per-case effects, descriptive bootstrap intervals, control seeds, dataset/vector hashes, software versions, and model revision metadata. The included eight cases are an authored pilot, not a validated psychological instrument; they confound calmness with deliberation and politeness. Do not tune on them and then call them a held-out test.

[The recorded GPT-2 pilot](experiments/results/gpt2_calm_pilot.json) and [research plan](docs/RESEARCH.md) describe the current evidence and the experiments needed before stronger claims. `benchmark.py` and the A/B interface remain exploratory diagnostics, not proof that the full architecture outperforms matched controls.

## Interactive demonstrations

```bash
python run.py
# Research dashboard: http://127.0.0.1:5000/
# Voxel face and browser voice: http://127.0.0.1:5000/face
# Legacy face: http://127.0.0.1:5000/face2d
```

The launcher asks which local model to use; `--model gpt2` selects the smaller
smoke-test model. The dashboard exposes affect, modulation, memory, intrusive
candidates, and speculative futures. Browser voice and decorative face motion
are presentation features, not evidence of experience. Snapshot data includes
recursive search depth, parent/child records, budgets, and score semantics;
the existing visual future cards show only terminal alternatives.

## Two research tracks

- **`magnum_opus_v2/`**: the main engine-over-LLM architecture described here. This is the path for testing the two core ideas with pretrained language models.
- **[Primordium](primordium/README.md)**: a separate experimental learner with perception, online learning, memory, and growth. It shares substrate components but does not establish the LLM-engine thesis. Its tests and claims must be evaluated separately.

[PAPER.md](PAPER.md) preserves the motivating design essay. [The research plan](docs/RESEARCH.md) records falsifiable claims, limitations, and a route toward independent academic replication. The goal is a useful, reproducible research contribution; scientific acceptance and historical significance have to be earned by results.
