# Adding the Vitalis engine to your LLM bot

For the quickest local test, install the repository requirements and run
`python run.py`. It lists cached models, accepts a Transformers model folder,
checks activation access, prepares a profile, and opens the dashboard. Use
`--check` for a generation diagnostic, `--download` to allow model downloads,
and `--help` for the other options. Loading is offline by default.

This guide integrates the persistent latent controller into a model you
host. Hooks recognize GPT-2-style `transformer.h`, Llama-style `model.layers`,
GPT-NeoX `gpt_neox.layers`, and OPT `model.decoder.layers`.
Other architectures need an adapter and a boundary test;
API-only models do not expose the required activations. Verify the model's
license and hardware requirements for your deployment.

---

## The one idea

The engine maintains recurrent state alongside a frozen language model.
The model already conditions on supplied history and cached tokens; Vitalis
adds independently evolving affect, memory, and background computation.
Its experimental loop is:

```
  LLM  ──predicts──▶  the engine reads its hidden states (emotion geometry,
                       situation, imagined futures)
   ▲                        │
   │                        ▼
 steer  ◀──every token──  the engine runs a small realtime substrate on
                          those readings and writes a steering vector back
                          into the model's residual stream
```

That loop — extract → drive a compact realtime engine → steer the LLM →
repeat — is the whole thing. The base model never changes; you give it a
living state that rides on top.

---

## What you need (the honest requirement)

The engine reads and writes the model's **hidden states** (it adds a
steering vector to a mid-layer's residual stream on every forward pass, via
a forward hook). Integration requires a supported block adapter:

- ✅ **Open-weight, self-hosted models** (HuggingFace `transformers`, or
  a backend with a verified block adapter). This is the target.
- ❌ **Closed, API-only models** where you only get text out. You cannot hook
  a residual stream you can't reach, so the engine can't drive them. (You can
  still run the engine on a *local* model and use the closed model separately,
  but the two won't share a substrate.)

Use Python 3.10+, PyTorch, and Transformers. Budget memory for the model,
KV caches, rollout captures, and the shared float32 vocabulary copy as well
as region state; measure peak usage on your hardware.

---

## Three steps

### 1. Extract a profile from your model (once)

The engine reads the model's own emotion geometry and fits how those
feelings move over time (the "Mirror"). This is a one-time, per-model step —
nothing is hand-authored:

```bash
python -m magnum_opus_v2.profile create <your-hf-model-name>
```

This saves a small profile (direction vectors + a neutral baseline + fitted
dynamics) under `profiles/`. It is per-model because the geometry differs
per model — a profile for one model is not valid for another.

### 2. Mount the engine on your model

```python
from magnum_opus_v2 import (
    V2Engine, load_model, load_profile, create_profile, profile_exists,
)

model, tokenizer, device = load_model("<your-hf-model-name>")
profile = (load_profile("<your-hf-model-name>")
           if profile_exists("<your-hf-model-name>")
    else create_profile("<your-hf-model-name>", loaded_model=(model, tokenizer, device)))

engine = V2Engine.from_profile(model, tokenizer, profile, device=device)
engine.start()          # the realtime substrate begins ticking
```

`from_profile` accepts your own `system_prompt=` if you want one, but the
engine ships **no authored persona** — by design, whatever character shows
up is your model's own, surfaced by the steering, not a script we wrote.

### 3. Route your bot's turns through the engine

Replace your bot's `model.generate(...)` call with `engine.converse(...)`.
Everything else in your bot (transport, auth, UI) stays the same:

```python
reply = engine.converse(user_text, max_new_tokens=200)   # steered generation
# ... send `reply` to your user however your bot already does ...

state = engine.snapshot()   # optional: emotions, imagined futures, etc.
```

On shutdown, `engine.stop()` — and if you want the mind to persist across
restarts ("a nap, not a death"), `engine.save_state()` / `engine.load_state()`.

That's the whole integration. There's also a ready-made local dashboard/face
you can point at it (`python run.py`) if you want to
watch it think, but it's optional.

---

## What the integration adds

- Persistent affect variables influence the next-token distribution.
- A fast noise/memory/association stack supplies intrusive candidates.
- A bounded recursive search explores hypothetical event paths and returns
  selected hypotheses to the upper subconscious layer.
- Multi-rate clocks continue substrate evolution during generation; the
  shared model lock prevents new LLM rollouts while generation owns it.

These are functional mechanisms. Emotional coherence, creativity, and
improved decisions require evaluation against matched controls.

## Configuration and migration

```python
from magnum_opus_v2.config import V2Config

config = V2Config()
config.spec.max_depth = 2       # events in a recursively conditioned path
config.spec.branching_factor = 2
config.spec.max_nodes = 12      # attempted model rollouts per round
config.spec.max_total_tokens = 128
config.spec.round_budget_ms = 500
engine = V2Engine.from_profile(model, tokenizer, profile, device=device, config=config)
```

Depth 1 disables recursive expansion for a shallow baseline. Token and node
budgets apply across the round. Time limits cannot preempt a running model
forward, so benchmark response latency under realistic load.

Profiles now require `activation_site="block_output"`. The launcher archives
and recalibrates legacy profiles automatically, reusing the loaded model.
Python integrations can re-extract with `create_profile(...,
loaded_model=(model, tokenizer, device))`. Start fresh runtime state after
recalibration; old checkpoints retain their original representations. The old extraction and steering
sites differed; relabeling old vectors or memories cannot repair them.
Single-block steering is the default.

The engine has authored extraction prompts, constants, weights, and selection
rules. Mirror fitting estimates some dynamics from model representations;
it does not discover every controller choice from biology. Its current
held-affect set is positive/neutral by design. The `goodness` and alignment
fields regulate that affect proxy and do not certify safe, ethical, or
correct behavior.

Profiles and latent memories are model- and layer-specific. The same hidden
dimension does not establish compatible geometry. See [the research
contract](RESEARCH.md) for the intervention pilot, unresolved limitations,
and the experiments required before claiming broad improvements.
