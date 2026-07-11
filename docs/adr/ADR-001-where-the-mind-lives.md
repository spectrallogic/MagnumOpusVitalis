# ADR-001 — Where the mind lives

Status: accepted · Era 6 · applies to magnum_opus_v2 AND primordium

## Decision

The mind is the **persistent cognitive runtime** — the state that survives
turns, restarts, and language-model replacement. The LLM is a replaceable
**language organ**: a reader, verbalizer, semantic field, and hypothesis
generator. Nothing that constitutes identity, belief, memory, goals, or
learning may live only inside an LLM context window.

## Boundaries

| Layer | Owns | Must never own |
|---|---|---|
| Runtime (bus, regions, Loom, organs) | time, continuity, learning, memory writes, action authorization | prose style |
| World models (Loom predictor, Wheel, engine speculation) | expectation, surprise, imagined futures | canonical facts (imagination is not observation) |
| Memory (Chronicle, Reach, Memory pool) | what happened, with provenance | unverified narration |
| LLM (frozen model in V2; Foster as environment in primordium) | parsing, labeling grounded concepts, verbalizing state, proposing hypotheses | writing beliefs, authorizing actions, being the self |
| UI (Pulse, dashboards, faces) | rendering measured state 1:1 | fabricating values (standing audit rule) |

## The LatentBus is modulation, not a fact store

The bus carries low-dimensional continuous context — mood, urgency, drift,
steering. It is NOT the canonical container of beliefs, plans, or memories.
Every bus write is recorded with provenance (source, norm, clock, ts —
see the provenance ledger, Era 6 M3). Typed cognitive state (beliefs,
goals, forecasts) lives in structured stores as they are built, era by
era; the bus whispers, it does not testify.

## State before speech

Every generated utterance begins from runtime state (situation vector,
affect, memory, forecasts) — the LLM does not reconstruct the mind from
chat history. V2 already implements this partially (situation narrative,
steering, rumination); the full cognitive-packet adapter and the LLM-swap
test are DEFERRED to a later era (see docs/BLUEPRINT_MAP.md).

## Unification path (contracts first)

V2 and primordium remain two loops this era. They adopt the SAME
contracts: the epistemic tag set (ADR-002), the provenance record shape,
and the forecast-ledger record shape. A physical `CognitiveState` merge
behind adapters is a later era, attempted only when the contracts have
been lived in by both minds.
