# Adding the Vitalis engine to your LLM bot

This is a developer guide for putting the Magnum Opus Vitalis engine on top
of your own chatbot. The engine is **model-agnostic** — it extracts
everything it needs from whatever LLM you point it at — so the same steps
work for a bot built on Qwen, Kimi, Llama, Mistral, Gemma, GPT-OSS, Phi, or
any other open-weight causal language model.

It is a normal open-source integration: **a human developer** adds this to a
bot they run. There is nothing here that makes a model install itself,
modify a system it doesn't control, or spread on its own — and this guide
will not help you do that. Adopt it on models you host, whose license
permits it.

---

## The one idea

A raw LLM is a frozen next-token predictor. It has no continuous inner
state: each request starts from nothing, emotions flicker and vanish, time
doesn't exist, nothing carries forward. But everything a mind needs —
emotional structure, associative reach, a sense of situation — is already
*in* the model's activation geometry. The engine's job is to **read that
structure out and feed it back in, continuously**:

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
a forward hook). So it works with any model you can run a forward pass on:

- ✅ **Open-weight, self-hosted models** (HuggingFace `transformers`, or
  anything exposing hidden states + a forward hook). This is the target.
- ❌ **Closed, API-only models** where you only get text out. You cannot hook
  a residual stream you can't reach, so the engine can't drive them. (You can
  still run the engine on a *local* model and use the closed model separately,
  but the two won't share a substrate.)

Roughly: a GPU with enough memory for your model (the engine itself is tiny —
a handful of vectors and some matrix ops per tick), Python 3.10+, `torch`,
and `transformers`.

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
           else create_profile("<your-hf-model-name>", device=device))

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
you can point at it (`python compare_server.py --profile`) if you want to
watch it think, but it's optional.

---

## What your bot gains

- **Emotional continuity** — feelings are held *outside* the model as
  multi-speed variables with falloff (the LLM forgets what it should feel
  moment-to-moment; the engine doesn't), and they steer the voice in
  realtime. It stays emotionally coherent across a whole conversation.
- **A subconscious** — a layered noise/memory/association stack that
  produces genuine, non-scripted intrusive thoughts and creativity.
- **Thinking** — it imagines several short futures each moment, scores them
  by the model's own confidence and goodness, and lets the most promising
  one shape what it says; imagined futures are held to account by a
  calibrated forecast ledger.
- **Realtime inner life** — a multi-clock substrate keeps ticking *during*
  generation, so an emotion onset or a passing thought can change a reply
  mid-sentence.
- **Alignment by construction** — see below.

---

## The design principles you inherit

Two choices in this engine matter for anyone adopting it:

1. **Nothing is hand-authored.** No personality, no temperament table, no
   good/bad word lists. Everything the mind feels or judges is *extracted*
   from your model's own geometry. If you find yourself writing a persona
   string to make it behave, that's the anti-pattern this engine exists to
   avoid — steer, don't script.

2. **It holds only positive emotions, for alignment.** The engine can
   *perceive* negativity (so it understands a distressed user), but it never
   *holds* fear, anger, or dread — it only ever feels and steers toward
   positive states. Alignment here isn't "detect a bad output and refuse";
   it's **steer toward good**: when the mind drifts from its good baseline or
   imagines a poorly-aligned future, it takes a "second thought" that
   re-steers toward good before it speaks. It never withholds a direct
   answer (that would be hidden censorship). This is a soft, mechanistic
   alignment layer, not a guarantee — treat it as one honest tool among
   whatever safety measures your bot already has.

---

## Honest limits

- **Per-model profile.** Steering vectors are model-specific; re-extract for
  each model. Swapping the underlying model means re-profiling.
- **Needs the residual stream.** Closed API-only models can't be driven (see
  Requirements).
- **A soft layer, not magic.** It gives an LLM continuity, an inner life, and
  a steer-toward-good bias — it does not make it correct, safe, or conscious.
  Keep your existing guardrails.
- **A human integrates it.** By design there is no self-installation or
  self-propagation path, and adding this to a model should honor that model's
  license and the wishes of whoever operates the bot.

For the full architecture and the reasoning behind every mechanism, see the
root `README.md`, `PAPER.md` (the speculative design essay), and the Reality
Contract in `primordium/README.md`.
