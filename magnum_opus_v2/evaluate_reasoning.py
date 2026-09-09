"""Objective paired pilot of the live Vitalis overlay; see experiments/REASONING_PROTOCOL.md."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
import transformers

from magnum_opus_v2.config import V2Config
from magnum_opus_v2.engine import V2Engine, _clean_reply
from magnum_opus_v2.loader import load_model
from magnum_opus_v2.startup import prepare_profile, validate_model_boundary
from magnum_opus_v2.steering_hook import SteeringHook


def normalized_answer(value):
    value = value.strip().strip("`*$ ").rstrip(".!").strip()
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.0+)?", value):
        return str(int(float(value.replace(",", ""))))
    return value.casefold()


def grade_answer(text, expected):
    finals = re.findall(r"(?im)^\s*(?:\*\*)?FINAL\s*:\s*(.+?)\s*$", text)
    explicit = bool(finals)
    candidates = finals or re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not candidates:
        candidates = re.findall(r"(?im)^.*?\b(?:final answer|answer)\s*(?:is|:)\s*(.+?)\s*$", text)
    if candidates:
        answer = candidates[-1]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        answer = lines[-1] if lines else ""
    parsed = normalized_answer(answer)
    return {"parsed_answer": parsed, "correct": parsed == normalized_answer(expected),
            "final_format": explicit, "empty": not text.strip()}


def prompt_ids(tokenizer, case, device):
    messages = [dict(turn) for turn in case.get("history", [])]
    messages.append({"role": "user", "content": case["prompt"]})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt").to(device)


def tensor_hash(ids):
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


class ModelMeter:
    """Count actual model calls, including imagination, and identify the primary prompt."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.calls = self.token_positions = 0
        self.generations = []
        self.lock = threading.Lock()
        self.original_generate = model.generate
        self.handle = model.register_forward_pre_hook(self.forward, with_kwargs=True)
        model.generate = self.generate

    def forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            ids = kwargs.get("inputs_embeds")
        n = int(ids.shape[0] * ids.shape[1]) if ids is not None else 0
        with self.lock:
            self.calls += 1
            self.token_positions += n

    def generate(self, *args, **kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        record = {"prompt_hash": tensor_hash(ids), "input_tokens": int(ids.shape[1])}
        synchronize(self.device)
        started = time.perf_counter()
        result = self.original_generate(*args, **kwargs)
        synchronize(self.device)
        record["seconds"] = time.perf_counter() - started
        seq = result.sequences if hasattr(result, "sequences") else result
        record["new_tokens"] = int(seq.shape[1] - ids.shape[1])
        eos = kwargs.get("eos_token_id", self.model.generation_config.eos_token_id)
        eos = eos if isinstance(eos, list) else [eos]
        record["truncated"] = record["new_tokens"] >= kwargs["max_new_tokens"] and int(seq[0, -1]) not in eos
        self.generations.append(record)
        return result

    def close(self):
        self.handle.remove()
        self.model.generate = self.original_generate


def evaluate_case(model, tokenizer, profile, case, condition, seed, device,
                  *, max_tokens=384, warmup_seconds=1.0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    expected_ids = prompt_ids(tokenizer, case, device)
    expected_hash = tensor_hash(expected_ids)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    meter = ModelMeter(model, device)
    engine = None
    started = time.perf_counter()
    row = {"case_id": case["id"], "category": case["category"],
           "condition": condition, "seed": seed, "expected": case["answer"],
           "prompt_hash": expected_hash, "error": None}
    try:
        if condition == "base":
            before_answer = time.perf_counter()
            output = model.generate(expected_ids, attention_mask=torch.ones_like(expected_ids),
                                    max_new_tokens=max_tokens, do_sample=False,
                                    top_p=1.0, temperature=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
            reply = _clean_reply(tokenizer.decode(output[0, expected_ids.shape[1]:], skip_special_tokens=True))
        else:
            engine = V2Engine.from_profile(model, tokenizer, profile, device=device,
                                           system_prompt=None, on_should_speak=lambda: None)
            if condition == "sham":
                engine.hook._current_vector = lambda: None
            for turn in case.get("history", []):
                if turn["role"] == "user":
                    engine.user_message(turn["content"])
            engine.chat_history = [dict(turn) for turn in case.get("history", [])]
            if engine.chat_history:
                tail = "\n".join(turn["content"] for turn in engine.chat_history[-2:])
                engine._context_ids = tokenizer(tail, return_tensors="pt", truncation=True,
                                                max_length=64, add_special_tokens=False)["input_ids"][0].detach()
            engine.start()
            time.sleep(warmup_seconds)
            before_answer = time.perf_counter()
            row["ticks_before_answer"] = engine.bus.tick_count
            reply = engine.converse(case["prompt"], max_new_tokens=max_tokens,
                                    do_sample=False, top_p=1.0, temperature=1.0)
        synchronize(device)
        row["turn_seconds"] = time.perf_counter() - before_answer
        row["setup_and_warmup_seconds"] = before_answer - started
        row["response"] = reply
        row.update(grade_answer(reply, case["answer"]))
        if engine is not None:
            snap = engine.snapshot()
            row["telemetry"] = {"flow_ticks": snap["flow_metrics"]["flow"]["ticks"],
                                "bus_norm": snap["bus"]["state_norm"],
                                "speculative_rounds": snap["speculative"]["rounds_total"],
                                "speculative_failures": snap["speculative"]["rollout_failures"],
                                "last_rollout_error": snap["speculative"]["last_rollout_error"],
                                "search": snap["speculative"]["search"],
                                "causes": engine._last_causes}
        if not meter.generations or meter.generations[0]["prompt_hash"] != expected_hash:
            raise RuntimeError("Primary generation input differs from the paired baseline prompt")
        row["primary_generation"] = meter.generations[0]
        row["truncated"] = meter.generations[0]["truncated"]
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                   correct=False, final_format=False)
    finally:
        if engine is not None:
            engine.flow.stop(timeout=30.0)
            if engine.flow._thread is not None and engine.flow._thread.is_alive():
                raise RuntimeError("Previous engine did not stop; refusing contaminated follow-up trials")
            engine.hook.detach()
        row["forward_calls"] = meter.calls
        row["processed_token_positions"] = meter.token_positions
        row["generation_calls"] = len(meter.generations)
        row["total_seconds"] = time.perf_counter() - started
        if device == "cuda":
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        meter.close()
    return row


def paired_interval(deltas, seed=20260909):
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"delta": float(values.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row["condition"] if row["condition"] != "overlay" else f"overlay_seed_{row['seed']}"
        groups[key].append(row)
    summary = {}
    for key, group in groups.items():
        summary[key] = {"n": len(group), "correct": sum(r["correct"] for r in group),
                        "accuracy": float(np.mean([r["correct"] for r in group])),
                        "format_rate": float(np.mean([r["final_format"] for r in group])),
                        "errors": sum(bool(r["error"]) for r in group),
                        "truncated": sum(r.get("truncated", False) for r in group),
                        "median_turn_seconds": float(np.median([r["turn_seconds"] for r in group if "turn_seconds" in r])),
                        "mean_forward_calls": float(np.mean([r["forward_calls"] for r in group])),
                        "mean_processed_token_positions": float(np.mean([r["processed_token_positions"] for r in group])),
                        "by_category": {cat: {"n": len(selected), "correct": sum(r["correct"] for r in selected)}
                                        for cat in sorted({r["category"] for r in group})
                                        if (selected := [r for r in group if r["category"] == cat])}}
    base = {r["case_id"]: r for r in rows if r["condition"] == "base"}
    overlay = defaultdict(list)
    for row in rows:
        if row["condition"] == "overlay":
            overlay[row["case_id"]].append(row)
    paired = [float(np.mean([r["correct"] for r in selected])) - base[case_id]["correct"]
              for case_id, selected in overlay.items() if case_id in base]
    summary["paired_mean_overlay_minus_base"] = paired_interval(paired) if paired else None
    for key, group in groups.items():
        if key.startswith("overlay_"):
            summary[key]["wins"] = sum(r["correct"] and not base[r["case_id"]]["correct"] for r in group)
            summary[key]["losses"] = sum(not r["correct"] and base[r["case_id"]]["correct"] for r in group)
    controls = [r for r in rows if r["condition"] == "sham"]
    summary["control_fidelity"] = {"n": len(controls), "response_disagreements": sum(
        r.get("response") != base[r["case_id"]].get("response") for r in controls)}
    return summary


def source_manifest():
    root = Path(__file__).resolve().parent.parent
    paths = list((root / "magnum_opus_v2").rglob("*.py"))
    paths += [root / "experiments/build_reasoning_pilot.py", root / "experiments/REASONING_PROTOCOL.md"]
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=Path("experiments/reasoning_pilot.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, default=Path("results/reasoning-profiles"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--control-per-category", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    payload = args.cases.read_bytes()
    cases = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("Cases must be nonempty with unique IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "responses.jsonl"
    if rows_path.exists():
        parser.error("Output already contains responses; choose a new directory to preserve the run")
    model, tokenizer, device = load_model(args.model, args.device, local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        parser.error("This protocol requires an instruction model with a chat template")
    validate_model_boundary(model, tokenizer, device)
    profile, _ = prepare_profile(args.model, (model, tokenizer, device), profiles_dir=args.profiles_dir)
    manifest = {"model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                "profile_signature": profile.signature(), "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "source_sha256": source_manifest(), "python": platform.python_version(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "engine_config": asdict(V2Config()), "seeds": args.seeds,
                "max_answer_tokens": args.max_tokens, "warmup_seconds": args.warmup_seconds,
                "cases": len(cases), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "protocol": "experiments/REASONING_PROTOCOL.md"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # One unscored forward warms up kernels before measuring either condition.
    with torch.no_grad():
        model(**tokenizer("Hello.", return_tensors="pt").to(device))
    rows, category_seen = [], defaultdict(int)
    with rows_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases):
            jobs = [("base", args.seeds[0])] + [("overlay", seed) for seed in args.seeds]
            if category_seen[case["category"]] < args.control_per_category:
                jobs.append(("sham", args.seeds[0]))
            category_seen[case["category"]] += 1
            random.Random(20260909 + index).shuffle(jobs)
            for condition, seed in jobs:
                row = evaluate_case(model, tokenizer, profile, case, condition, seed, device,
                                    max_tokens=args.max_tokens, warmup_seconds=args.warmup_seconds)
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                print(f"  [{index+1}/{len(cases)}] {case['id']} {condition}/{seed}: "
                      f"{'correct' if row['correct'] else 'WRONG'}" +
                      (f" ERROR={row['error']}" if row['error'] else ""), flush=True)
                gc.collect()
            # Failures remain in the log; abort if the harness/control has broken.
            if any(r["error"] for r in rows[-len(jobs):]):
                raise RuntimeError("A trial failed; raw results retained. Resolve the failure before proceeding.")
    report = {"manifest": manifest, "summary": summarize(rows)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
