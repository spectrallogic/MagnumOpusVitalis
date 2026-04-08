"""
Magnum Opus Vitalis - Blind Brain Discovery + Live Engine
==========================================================
Phase 1: The model discovers its own mind with zero guidance.
Phase 2: The discovered brain comes alive. Chat with it.
         Watch every direction react in real-time 3D.

Usage:
    python discover_server.py --model gpt2-medium
    Open http://localhost:5001
"""

import argparse, json, math, re, threading, time, random
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__, static_folder=".")

state = {
    "phase": "waiting", "phase_num": 0, "total_phases": 5,
    "message": "Waiting...", "progress": 0.0, "round": 0,
    "dimensions": {}, "interactions": [], "clusters": [],
    "dynamics": {}, "best_layer": None,
    "complete": False, "alive": False, "config": None, "log": [],
    "activations": {},  # live activation levels per direction
    "brain_activity": 0.0,
}
state_lock = threading.Lock()
model = None; tokenizer = None; device = "cpu"; args_model = "gpt2"
raw_vectors = {}; best_layer = 6; heartbeat_on = False

def log(msg):
    with state_lock:
        state["log"].append({"time": time.time(), "msg": msg})
        state["message"] = msg
    print(f"  [DISCOVER] {msg}")

def gen(prompt, max_tokens=60, temp=0.7):
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(inp["input_ids"], max_new_tokens=max_tokens,
                             temperature=temp, do_sample=True, top_p=0.9,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

def get_act(text, layer):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer].mean(dim=1).squeeze(0).float().cpu()

def project_to_3d(vecs):
    if len(vecs) < 2: return {n: [0, 0, 0] for n in vecs}
    names = list(vecs.keys()); mat = torch.stack([vecs[n] for n in names])
    mat = mat - mat.mean(dim=0)
    try:
        U, S, V = torch.svd(mat); proj = U[:, :3] * S[:3]
        mx = proj.abs().max()
        if mx > 0: proj = proj / mx * 0.85
        return {names[i]: proj[i].tolist() for i in range(len(names))}
    except: return {n: [random.uniform(-.5,.5) for _ in range(3)] for n in names}

CORPUS = [
    "She read the letter three times before the tears came.",
    "He couldn't stop laughing at the absurdity of it all.",
    "Her hands were shaking as she opened the door. Something was wrong.",
    "The moment his name was called he felt like he was floating.",
    "She slammed the phone down. How dare they do this to her family.",
    "He sat alone in the parking lot for an hour. Not crying. Just sitting.",
    "They had been best friends for twenty years but never talked about it.",
    "The argument wasn't about the dishes. It was never about the dishes.",
    "She could read a room in seconds. Who had power. Who was pretending.",
    "The solution was obvious in hindsight but took three weeks to see.",
    "He approached the problem from seventeen angles before asking for help.",
    "Something about the data didn't add up. The numbers were right but the story was wrong.",
    "She trusted her gut even when the spreadsheet said otherwise.",
    "The proof was elegant. Three lines where everyone else needed thirty.",
    "Stealing the bread was wrong. Watching his daughter starve was worse.",
    "She forgave him not because he deserved it but because the anger was killing her.",
    "The smell of cinnamon still brought him back to her kitchen thirty years later.",
    "She couldn't remember his face anymore. Just the sound of his laugh.",
    "The cold hit her lungs like broken glass with every breath.",
    "The melody came to him at 3 AM fully formed like it had been waiting.",
    "She stared at the blank canvas for six hours before making one mark.",
    "She had to choose and both options meant losing something she couldn't get back.",
    "Confidence and correctness are different things. She was confident. Also wrong.",
    "She explained it five different ways before his eyes lit up.",
    "The system wasn't broken. It was working as designed. That was the problem.",
    "Tuesday. Coffee. Emails. The same parking spot. The slow erosion of urgency.",
    "The map is not the territory. The menu is not the meal.",
    "Nothing happened today. Something would happen eventually.",
    "The first bite after three days without food tasted like the entire concept of mercy.",
    "His muscles remembered the motion even though his mind had forgotten.",
    "The creative breakthrough came from letting go, not trying harder.",
    "The conversation shifted into something deeper when she stopped performing.",
    "Time seemed to slow as the critical moment approached.",
    "He made the decision knowing he would never know if it was right.",
    "The student asked a question that made the professor realize she'd been wrong for years.",
    "Some things can only be understood by living through them.",
    "The universe doesn't owe us meaning. We make it and project it outward.",
    "Twenty years from now this would be a dinner party story. Right now it was everything.",
    "He kept the ticket stub in his wallet for fifteen years. He couldn't explain why.",
    "The right thing and the kind thing were different things today.",
    "Her apology was perfect. Every word calculated. He accepted it without believing any of it.",
    "Trust takes years to build and seconds to destroy.",
    "The child didn't understand why adults said things they didn't mean.",
    "He had the authority to change it. He didn't have the courage.",
    "Writer's block isn't the absence of ideas. It's the presence of too many bad ones.",
    "She couldn't fight the machine alone. But she started anyway.",
    "If you replaced every part of a ship one plank at a time when is it a different ship?",
    "The worst part wasn't the uncertainty. It was knowing that waiting was also a choice.",
    "The diagnosis was wrong. She was sure. But proving it meant going against everyone.",
    "Something felt fundamentally wrong. A sense that everything was about to change.",
]

COLORS = ["#E8A838","#D4842A","#F0B848","#C07020","#E89830","#D09038","#F0C060","#B87828",
          "#E0A040","#C88830","#F0D070","#A87020","#D09840","#E8B050","#C08028"]

def extract_vec(pos, neg, layer):
    def ms(prompts):
        ss = []
        for p in prompts:
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad(): out = model(**inp, output_hidden_states=True)
            ss.append(out.hidden_states[layer].mean(dim=1).squeeze(0).float().cpu())
        return torch.stack(ss).mean(dim=0)
    p, n = ms(pos), ms(neg)
    d = p - n; norm = d.norm().item()
    if norm > 0: d = d / d.norm()
    return d, norm


# ═══════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def run_discovery(num_rounds=3, dirs_per_round=15):
    global raw_vectors, best_layer, heartbeat_on

    if hasattr(model.config, "n_layer"): n_layers = model.config.n_layer
    else: n_layers = model.config.num_hidden_layers

    norms = {}; chars = {}; dynamics = {}; interactions = []
    dim_cluster = {}

    # Phase 1: Find best layer
    with state_lock: state.update(phase="finding_layer", phase_num=1, progress=0)
    log(f"Finding most responsive layer across {n_layers}...")
    t1, t2 = "She was overwhelmed with joy, laughing and dancing.", "He sat motionless in the dark, contemplating silence."
    scores = {}
    for L in range(n_layers):
        s1, s2 = get_act(t1, L), get_act(t2, L)
        scores[L] = (s1 - s2).norm().item() * (1 - F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item())
        with state_lock: state["progress"] = (L+1)/n_layers
    best_layer = max(scores, key=scores.get)
    with state_lock: state["best_layer"] = best_layer
    log(f"Best layer: {best_layer}")

    # Iterative rounds
    for rnd in range(1, num_rounds + 1):
        with state_lock: state.update(phase=f"round_{rnd}", phase_num=1+rnd, round=rnd, progress=0)
        log(f"{'='*30} Round {rnd}/{num_rounds} {'='*30}")

        if rnd == 1:
            probes = CORPUS
        else:
            probes = []
            log(f"Generating probes from {len(raw_vectors)} known directions...")
            for dn, dv in list(raw_vectors.items())[:20]:
                hook_v = [dv * 3.0]
                def mkhook():
                    def hfn(m, i, o):
                        h = o[0]; sv = hook_v[0].to(h.device).to(h.dtype)
                        return (h + sv.unsqueeze(0).unsqueeze(0),) + o[1:]
                    return hfn
                hfn = mkhook()
                if hasattr(model, "transformer"): handle = model.transformer.h[best_layer].register_forward_hook(hfn)
                else: handle = model.model.layers[best_layer].register_forward_hook(hfn)
                probes.append(gen("The person experienced something that made them", max_tokens=40, temp=0.9))
                handle.remove()
                hook_v[0] = dv * -3.0
                hfn2 = mkhook()
                if hasattr(model, "transformer"): handle = model.transformer.h[best_layer].register_forward_hook(hfn2)
                else: handle = model.model.layers[best_layer].register_forward_hook(hfn2)
                probes.append(gen("The person experienced something that made them", max_tokens=40, temp=0.9))
                handle.remove()

        # Collect activations
        acts = []
        for i, t in enumerate(probes):
            acts.append(get_act(t, best_layer))
            with state_lock: state["progress"] = (i+1)/len(probes) * .3
        mat = torch.stack(acts); mat = mat - mat.mean(dim=0)

        # SVD
        log("Running SVD...")
        try: U, S, V = torch.svd(mat)
        except: continue
        n_d = min(dirs_per_round, len(S))
        candidates = V[:, :n_d].T

        # Filter for novelty
        new = 0
        for idx in range(n_d):
            d = candidates[idx]; d = d / max(d.norm().item(), 1e-8)
            if raw_vectors:
                ma = max(abs(F.cosine_similarity(d.unsqueeze(0), ev.unsqueeze(0)).item()) for ev in raw_vectors.values())
            else: ma = 0
            if ma < 0.7:
                name = f"d{len(raw_vectors)+1}"; raw_vectors[name] = d
                norms[name] = S[idx].item() if idx < len(S) else 1.0; new += 1
            with state_lock: state["progress"] = .3 + (idx+1)/n_d * .2

        log(f"Found {new} novel directions ({len(raw_vectors)} total)")

        # Characterize
        log("Characterizing by behavioral effect...")
        for di, (dn, dv) in enumerate(list(raw_vectors.items())):
            if dn in chars: continue
            tp = "The person looked around and then"
            inp = tokenizer(tp, return_tensors="pt").to(device)
            with torch.no_grad():
                base_ids = model.generate(inp["input_ids"], max_new_tokens=25, temperature=.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            base_t = tokenizer.decode(base_ids[0], skip_special_tokens=True)[len(tp):].strip()

            for sign, label in [(4.0, "positive"), (-4.0, "negative")]:
                hv = [dv * sign]
                def mkhook2():
                    def hfn(m, i, o):
                        h = o[0]; sv = hv[0].to(h.device).to(h.dtype)
                        return (h + sv.unsqueeze(0).unsqueeze(0),) + o[1:]
                    return hfn
                hfn = mkhook2()
                if hasattr(model, "transformer"): handle = model.transformer.h[best_layer].register_forward_hook(hfn)
                else: handle = model.model.layers[best_layer].register_forward_hook(hfn)
                with torch.no_grad():
                    sid = model.generate(inp["input_ids"], max_new_tokens=25, temperature=.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                handle.remove()
                st = tokenizer.decode(sid[0], skip_special_tokens=True)[len(tp):].strip()
                if dn not in chars: chars[dn] = {"baseline": base_t[:80]}
                chars[dn][label] = st[:80]

            projs = [torch.dot(a, dv).item() for a in acts[:30]]
            dynamics[dn] = {
                "mean": round(float(np.mean(projs)), 3),
                "std": round(float(np.std(projs)), 3),
                "onset": round(min(1.0, float(np.std(projs))/20), 4),
                "decay": round(max(.01, min(.2, 1/(float(np.std(projs))+1))), 4),
            }
            with state_lock: state["progress"] = .5 + (di+1)/len(raw_vectors) * .3

        # Interactions
        log("Measuring interactions...")
        base_a = get_act("The person thought carefully about what to do.", best_layer)
        base_p = {n: torch.dot(base_a, v).item() for n, v in raw_vectors.items()}
        vn = list(raw_vectors.keys()); srcs = vn[:15]
        for si, src in enumerate(srcs):
            ct = chars.get(src, {}).get("positive", "The person decided to")
            sa = get_act(ct, best_layer)
            for tgt in vn:
                if tgt == src: continue
                sp = torch.dot(sa, raw_vectors[tgt]).item()
                rel = (sp - base_p.get(tgt, 0)) / max(abs(base_p.get(tgt, .01)), .01)
                if abs(rel) > .15:
                    interactions.append({"source": src, "target": tgt, "strength": round(max(-1, min(1, rel*.3)), 3)})
            with state_lock: state["progress"] = .8 + (si+1)/len(srcs) * .15
        # Deduplicate
        seen = set(); ui = []
        for ix in interactions:
            k = f"{ix['source']}|{ix['target']}"
            if k not in seen: seen.add(k); ui.append(ix)
        interactions = ui

        # Cluster
        clusters = []; assigned = set()
        vn_all = list(raw_vectors.keys())
        for i, n1 in enumerate(vn_all):
            if n1 in assigned: continue
            cl = [n1]; assigned.add(n1)
            for j, n2 in enumerate(vn_all):
                if n2 in assigned: continue
                sim = F.cosine_similarity(raw_vectors[n1].unsqueeze(0), raw_vectors[n2].unsqueeze(0)).item()
                if sim > .4: cl.append(n2); assigned.add(n2)
            clusters.append(cl)
        for ci, cl in enumerate(clusters):
            for m in cl: dim_cluster[m] = ci

        # Update 3D
        pos = project_to_3d(raw_vectors)
        with state_lock:
            state["dimensions"] = {
                n: {"x": pos[n][0], "y": pos[n][1], "z": pos[n][2],
                    "strength": round(norms.get(n, 1), 2),
                    "color": COLORS[dim_cluster.get(n, 0) % len(COLORS)],
                    "cluster": dim_cluster.get(n, 0),
                    "char": chars.get(n, {})}
                for n in raw_vectors}
            state["interactions"] = interactions
            state["clusters"] = [{"members": c, "color": COLORS[i % len(COLORS)]} for i, c in enumerate(clusters)]
            state["dynamics"] = dynamics

    # Save
    config = {
        "version": "blind-1.0", "model": args_model, "target_layer": best_layer,
        "total": len(raw_vectors), "vectors": {n: v.tolist() for n, v in raw_vectors.items()},
        "chars": chars, "dynamics": dynamics,
        "interactions": {f"{x['source']}->{x['target']}": x["strength"] for x in interactions},
    }
    with open("discovered_brain.json", "w") as f: json.dump(config, f, indent=2)
    with state_lock:
        state.update(config=config, complete=True, phase="complete", progress=1.0)
    log(f"Brain complete: {len(raw_vectors)} directions, {len(clusters)} clusters")
    log("Bringing brain online...")

    # START LIVE ENGINE
    heartbeat_on = True
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    with state_lock: state["alive"] = True
    log("Brain is alive. Start talking.")


# ═══════════════════════════════════════════════════════════════════════════
# LIVE ENGINE (post-discovery)
# ═══════════════════════════════════════════════════════════════════════════

current_activations = {}
steering_hook_handle = None

def heartbeat_loop():
    """Keep the brain alive. Measure activations. Background activity."""
    global current_activations
    while heartbeat_on:
        time.sleep(0.5)
        # Natural drift: activations decay toward 0
        for n in list(current_activations.keys()):
            current_activations[n] *= 0.93
        # Subconscious noise
        for n, v in list(raw_vectors.items())[:20]:
            jitter = np.random.normal(0, 0.02)
            current_activations[n] = current_activations.get(n, 0) + jitter

        total = sum(abs(v) for v in current_activations.values())
        with state_lock:
            state["activations"] = {k: round(v, 4) for k, v in current_activations.items()}
            state["brain_activity"] = round(total / max(len(current_activations), 1), 4)


def chat_with_brain(message, max_tokens=80):
    """Full conversation with the discovered brain steering in real-time."""
    global current_activations

    # Measure what the input activates
    inp_act = get_act(message, best_layer)
    for n, v in raw_vectors.items():
        proj = torch.dot(inp_act, v).item()
        current_activations[n] = current_activations.get(n, 0) + proj * 0.1

    # Build combined steering vector from all active directions
    steering = torch.zeros_like(list(raw_vectors.values())[0])
    for n, v in raw_vectors.items():
        activation = current_activations.get(n, 0)
        if abs(activation) > 0.01:
            steering = steering + v * activation * 0.5
    # Clamp
    sn = steering.norm().item()
    if sn > 3.0: steering = steering / sn * 3.0

    # Hook
    hook_v = [steering]
    def hfn(m, i, o):
        h = o[0]; sv = hook_v[0].to(h.device).to(h.dtype)
        return (h + sv.unsqueeze(0).unsqueeze(0),) + o[1:]

    if hasattr(model, "transformer"): handle = model.transformer.h[best_layer].register_forward_hook(hfn)
    else: handle = model.model.layers[best_layer].register_forward_hook(hfn)

    prompt = f"User: {message}\nAssistant:"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(inp["input_ids"], max_new_tokens=max_tokens,
                             temperature=0.7, do_sample=True, top_p=0.9,
                             pad_token_id=tokenizer.eos_token_id)
    handle.remove()

    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    if "Assistant:" in text: response = text.split("Assistant:")[-1].strip()
    else: response = text[len(prompt):].strip()

    # Measure output activations
    out_act = get_act(response, best_layer)
    for n, v in raw_vectors.items():
        proj = torch.dot(out_act, v).item()
        current_activations[n] = current_activations.get(n, 0) * 0.7 + proj * 0.05

    top = sorted(current_activations.items(), key=lambda x: -abs(x[1]))[:5]
    return {"response": response, "top_active": [{"dim": n, "val": round(v, 3)} for n, v in top]}


# ═══════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index(): return send_from_directory("..", "discover_dashboard.html")

@app.route("/api/state")
def api_state():
    with state_lock: return jsonify(state)

@app.route("/api/start", methods=["POST"])
def api_start():
    body = request.json or {}
    t = threading.Thread(target=run_discovery, args=(body.get("rounds", 3), body.get("dirs", 15)), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.json or {}
    msg = body.get("message", "")
    if not msg: return jsonify({"error": "No message"})
    if not state.get("alive"): return jsonify({"error": "Brain not alive yet"})
    result = chat_with_brain(msg)
    return jsonify(result)

args_model = "gpt2"
def main():
    global model, tokenizer, device, args_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args(); args_model = args.model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60); print("  BLIND BRAIN DISCOVERY + LIVE ENGINE"); print("="*60)
    print(f"\n  Loading '{args.model}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    model.eval(); torch.manual_seed(42); np.random.seed(42)
    print(f"\n  http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

if __name__ == "__main__": main()
