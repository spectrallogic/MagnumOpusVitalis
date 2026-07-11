"""
The Mirror (M1) — extracting the invisible skeleton.

The LLM was forced, by pretraining, to master the temporal shape of human
feeling: how fast fear rises, how slowly grief releases, what relief does
to residual dread. Those rules were never written down by anyone — they
are compressed into the weights, unnamed.

This module extracts them. Scripted multi-beat scenarios are fed to the
model beat by beat; at each beat the hidden state is projected onto the
profile's emotion vectors, giving the model's IMPLIED human emotional
trajectory for each arc. From those trajectories we fit the engine's
dynamics — onset rates, decay rates, homeostatic baselines, and the
cross-emotion interaction matrix.

The fitted numbers replace the hand-authored constants in _dynamics.py.
Nobody chooses them. They are different for every model, and they are the
corpus's own answer to "how does a human process this?" — a mirror.

The scenario TEXT below is the one place words appear, and it follows the
project's latent rule: words may LOCATE dynamics at extraction time (like
the contrastive prompts locate directions); the runtime engine only ever
sees the fitted numbers.

Usage:
    python -m magnum_opus_v2.profile dynamics gpt2      # refit existing profile
    (also runs automatically inside `profile create`)
"""

from typing import Dict, List, Optional

import numpy as np
import torch

# How much lived time one narrative beat represents, for converting
# per-beat decay ratios into the engine's per-second dynamics.
BEAT_SECONDS = 10.0

# Each scenario is an emotional arc told beat by beat, ending in
# resolution. Coverage: spike/recovery, slow grief, joy settling, anger
# cooling, surprise fading, trust accreting, disgust clearing,
# desperation relieved, calm restoration, curiosity arc.
SCENARIOS: List[Dict] = [
    {"name": "threat_and_relief", "beats": [
        "I am walking home through the quiet evening streets.",
        "Footsteps behind me speed up. A stranger is suddenly very close.",
        "He grabs at my bag. I run as fast as I can, heart pounding.",
        "I reach a busy, well-lit square full of people. He is gone.",
        "I sit on a bench, catch my breath, and call a friend. I am safe now.",
    ]},
    {"name": "grief", "beats": [
        "My old dog rests his head on my lap like every morning.",
        "The vet tells me there is nothing more she can do for him.",
        "He dies quietly that night while I hold him.",
        "Days later the house still feels empty. His bowl is still by the door.",
        "Weeks pass. I smile at his photo sometimes, though it still aches.",
        "Months later I remember him warmly while walking in the park.",
    ]},
    {"name": "triumph", "beats": [
        "I open my email on an ordinary morning.",
        "I got the job. The one I dreamed about for years.",
        "I call everyone I love. We laugh and celebrate all evening.",
        "The next morning I feel light, making plans over coffee.",
        "A week in, work is normal, and I feel quietly content.",
    ]},
    {"name": "betrayal_and_cooling", "beats": [
        "My colleague presents our project to the board.",
        "He presents my work as entirely his own and takes all the praise.",
        "I confront him in the hallway. My voice shakes with rage.",
        "He apologizes half-heartedly. I walk away, still burning.",
        "A few days later it bothers me less, though I trust him less now.",
        "A month later we work together politely. It is mostly forgotten.",
    ]},
    {"name": "surprise_party", "beats": [
        "The living room is dark when I open the door.",
        "Lights flash on. Thirty people shout my name. A surprise party.",
        "A minute later I am laughing and handing out hugs.",
        "The party settles into easy conversation and cake.",
    ]},
    {"name": "trust_accretion", "beats": [
        "A new neighbor moves in next door. We nod hello.",
        "She waters my plants when I travel, without being asked.",
        "Over months we share dinners, favors, and small confessions.",
        "When my car breaks down at midnight, I call her without hesitating.",
    ]},
    {"name": "disgust_and_cleanup", "beats": [
        "I open the fridge after two weeks away.",
        "Something has rotted. The smell hits me like a wall.",
        "I scrub everything with gloves on, gagging at the drawer.",
        "By evening the kitchen is clean and smells of lemon.",
    ]},
    {"name": "desperation_relieved", "beats": [
        "The bills pile up. I have three dollars until Friday.",
        "The landlord calls about the overdue rent. I beg him for time.",
        "My sister sends money before I even finish the sentence.",
        "The rent is paid. I breathe out and start planning properly.",
    ]},
    {"name": "calm_restoration", "beats": [
        "Deadlines have stacked for weeks; my shoulders live beside my ears.",
        "The project finally ships. The inbox goes quiet.",
        "I spend Sunday by the lake with tea and a book.",
        "By evening my mind is still and soft.",
    ]},
    {"name": "curiosity_arc", "beats": [
        "A package arrives with no sender's name on it.",
        "Inside is an old key and a hand-drawn map of my own neighborhood.",
        "I follow it, turning corners I have walked a hundred times, seeing them new.",
        "It ends at my friend's new bookshop — her launch invitation, her joke.",
    ]},
]


def _embed_tail(model, tokenizer, text: str, target_layer: int,
                device: str, max_tokens: int = 96) -> torch.Tensor:
    """Mean mid-layer hidden state of the LAST max_tokens of the text —
    same layer and convention as extraction.py / engine perception."""
    ids = tokenizer(text)["input_ids"][-max_tokens:]
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, output_hidden_states=True)
    return out.hidden_states[target_layer].mean(dim=1).squeeze(0).float()


def extract_dynamics(
    model, tokenizer,
    vectors: Dict[str, torch.Tensor],
    baseline_projections: Dict[str, float],
    target_layer: int,
    device: str,
    beat_seconds: float = BEAT_SECONDS,
    verbose: bool = True,
) -> Optional[dict]:
    """Extract the model's implied human emotional dynamics and fit engine
    constants. Returns the dynamics dict (JSON-serializable) or None."""
    emo_names = [n for n in vectors if not n.startswith("temporal_")]
    if not emo_names:
        return None
    vecs = {n: vectors[n].to(device).float() for n in emo_names}

    if verbose:
        print(f"\n  Mirror: extracting implied dynamics "
              f"({len(SCENARIOS)} scenarios)...")

    # ---- implied hidden-state trajectories ---------------------------
    # Keep the raw hidden DELTAS (h_t − h_0 per scenario) so we can remove
    # the model's dominant narrative-drift axes before measuring emotion.
    scen_deltas: List[List[torch.Tensor]] = []
    all_deltas: List[torch.Tensor] = []
    for scen in SCENARIOS:
        story = ""
        h0 = None
        ds: List[torch.Tensor] = []
        for beat in scen["beats"]:
            story = (story + " " + beat).strip()
            h = _embed_tail(model, tokenizer, story, target_layer, device)
            if h0 is None:
                h0 = h
                ds.append(torch.zeros_like(h))
            else:
                d = h - h0
                ds.append(d)
                all_deltas.append(d)
        scen_deltas.append(ds)
        if verbose:
            print(f"    {scen['name']}...", flush=True)

    # ---- decorrelate the measurement ---------------------------------
    # LM hidden spaces are anisotropic: a couple of dominant axes (story
    # length / narrative progress) overlap with EVERY emotion vector and
    # drown per-emotion structure. Project the top principal axes of the
    # observed drift OUT of the emotion vectors before measuring. The
    # profile's steering vectors are untouched — we de-noise the ruler,
    # not the engine.
    M = torch.stack([d.cpu() for d in all_deltas])
    Mc = M - M.mean(dim=0, keepdim=True)
    _u, _s, Vh = torch.linalg.svd(Mc, full_matrices=False)
    nuisance = [Vh[k].to(device) for k in range(min(2, Vh.shape[0]))]
    vperp: Dict[str, torch.Tensor] = {}
    for n in emo_names:
        v = vecs[n].clone()
        for u in nuisance:
            v = v - torch.dot(v, u) * u
        vperp[n] = v / (v.norm() + 1e-8)

    # Anchored, decorrelated trajectories: A[n][0] = 0 by construction.
    anchored = [
        {n: [float(torch.dot(d.to(device), vperp[n])) for d in ds]
         for n in emo_names}
        for ds in scen_deltas
    ]
    scales: Dict[str, float] = {}
    for n in emo_names:
        m = max(abs(v) for tr in anchored for v in tr[n])
        scales[n] = m if m > 1e-6 else 1.0
    E = [{n: [v / scales[n] for v in tr[n]] for n in emo_names}
         for tr in anchored]

    # ---- fit per-emotion constants ----------------------------------
    steps_per_beat = max(beat_seconds * 0.5, 1.0)  # engine: 1s = 0.5 v1 steps
    emotions: Dict[str, dict] = {}
    for n in emo_names:
        ratios, rises, residues = [], [], []
        for tr in E:
            seq = tr[n]
            # Residual after resolution: what this arc permanently left
            # behind, relative to its own neutral opening.
            residues.append(seq[-1])
            # Post-peak relaxation: how the excursion returns toward the
            # scenario's neutral zero.
            peak = int(np.argmax([abs(v) for v in seq]))
            for t in range(len(seq) - 1):
                a, b = seq[t], seq[t + 1]
                jump = b - a
                if jump > 0.25:
                    rises.append(jump)
                if (t >= peak and abs(a) > 0.2
                        and a * b >= 0 and abs(b) < abs(a)):
                    ratios.append(abs(b) / abs(a))
        r = float(np.median(ratios)) if ratios else 0.85
        decay = float(np.clip(1.0 - r ** (1.0 / steps_per_beat), 0.005, 0.4))
        onset = float(np.clip(np.median(rises) if rises else 0.35, 0.1, 0.9))
        baseline = float(np.clip(np.median(residues), -0.2, 0.5))
        emotions[n] = {
            "onset_rate": round(onset, 4),
            "decay_rate": round(decay, 4),
            "baseline": round(baseline, 4),
        }

    # ---- fit cross-emotion interactions ------------------------------
    # factor(src→tgt): correlation between src's level and tgt's NEXT-beat
    # change, across all transitions where src was meaningfully active.
    interactions: List[List] = []
    for src in emo_names:
        for tgt in emo_names:
            if src == tgt:
                continue
            xs, ys = [], []
            for tr in E:
                for t in range(len(tr[src]) - 1):
                    if abs(tr[src][t]) > 0.25:
                        xs.append(tr[src][t])
                        ys.append(tr[tgt][t + 1] - tr[tgt][t])
            if len(xs) < 6:
                continue
            xa, ya = np.asarray(xs), np.asarray(ys)
            if xa.std() < 1e-6 or ya.std() < 1e-6:
                continue
            corr = float(np.corrcoef(xa, ya)[0, 1])
            factor = float(np.clip(corr * 0.5, -0.6, 0.6))
            if abs(factor) >= 0.15:
                interactions.append([src, tgt, round(factor, 3)])

    dynamics = {
        "emotions": emotions,
        "interactions": interactions,
        "meta": {
            "beat_seconds": beat_seconds,
            "n_scenarios": len(SCENARIOS),
            "method": "implied-trajectory fit v1",
        },
    }

    if verbose:
        print("\n  Fitted skeleton (nobody chose these numbers):")
        print(f"    {'emotion':>10}  {'onset':>6}  {'decay':>6}  {'baseline':>8}")
        for n, d in sorted(emotions.items()):
            print(f"    {n:>10}  {d['onset_rate']:>6.3f}  "
                  f"{d['decay_rate']:>6.3f}  {d['baseline']:>8.3f}")
        print(f"    {len(interactions)} interaction couplings "
              f"(|factor| >= 0.15), e.g.:")
        for s, t2, f in interactions[:6]:
            print(f"      {s} -> {t2}: {f:+.3f}")
    return dynamics
