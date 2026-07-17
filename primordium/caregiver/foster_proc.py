"""
Foster's body — runs OUTSIDE the organism, as a separate OS process.

This is the honest boundary: the caregiver LLM is ENVIRONMENT, not
weights inside the mind. It talks over stdin/stdout JSON lines; if it
crashes, the room simply goes quiet (the parent surfaces the absence).

    {"t":"observe","babble":"...","state":{...}}  ->  {"t":"say","text":"..."}
    {"t":"ping"}                                  ->  {"t":"pong"}

`--stub` replaces the LLM with a tiny deterministic mimic so tests can
exercise the whole protocol without a 3.6GB model. Same protocol,
clearly labeled in every reply's meta.

No authored persona: the caregiver LLM is given only a NEUTRAL, factual
description of what the infant did and asked for a short response —
whatever character it has is the model's own, not a hand-written
"gentle caregiver" script. The caregiver ships OFF by default and is
gated; this is optional environment, not behavior inside the mind.
"""

import argparse
import json
import sys


def _neutral_prompt(babble: str, state: dict) -> str:
    """A factual, personality-free description of the moment."""
    if not babble:
        return "The infant is quiet. Reply with one short sentence."
    awake = "awake" if state.get("awake", True) else "asleep"
    return (f"The infant typed: {babble!r}. It is {awake}, "
            f"developmental stage {state.get('stage', 0)}. "
            "Reply with one short sentence.")


def _stub_reply(babble: str, n: int) -> str:
    """Neutral deterministic stand-in — echoes a fragment so tests have
    a stable, content-free reply. No authored personality."""
    frag = "".join(ch for ch in babble if ch.isalnum())[:6].lower()
    return f"reply {n}: {frag}" if frag else f"reply {n}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--max-chars", type=int, default=120)
    args = ap.parse_args()

    tok = model = None
    if not args.stub:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

    print(json.dumps({"t": "ready", "stub": bool(args.stub)}), flush=True)
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        t = msg.get("t")
        if t == "ping":
            print(json.dumps({"t": "pong"}), flush=True)
        elif t == "observe":
            babble = str(msg.get("babble", ""))[:200]
            if args.stub:
                text = _stub_reply(babble, n)
            else:
                import torch
                state = msg.get("state", {})
                # neutral, factual prompt only — no authored persona
                user = _neutral_prompt(babble, state)
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(chat, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **ids, max_new_tokens=30, do_sample=True,
                        temperature=0.8, top_p=0.9,
                        pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                text = text.split("\n")[0].strip()
            n += 1
            print(json.dumps({"t": "say",
                              "text": text[:args.max_chars],
                              "stub": bool(args.stub)}), flush=True)


if __name__ == "__main__":
    main()
