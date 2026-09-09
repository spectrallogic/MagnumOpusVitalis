"""Build an auditable Markdown report from completed, immutable pilot runs."""

import argparse
import json
from pathlib import Path
import statistics


def pct(value):
    return f"{100 * value:.1f}%"


def build_report(root):
    runs = []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and (directory / "report.json").is_file():
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (directory / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
            runs.append((directory.name, report, rows))
    if not runs:
        raise ValueError("No completed run reports")
    lines = ["# Does the Vitalis overlay improve objective task accuracy?", "",
             "This is a local, authored pilot with 64 fixed questions per model. Each base answer is paired with two live-overlay runs using the same model, prompt tokens, supplied conversation history, and greedy answer decoding. All answers and failures are retained.", "",
             "| Model | Base | Overlay seed 11 | Overlay seed 23 | Mean change | Paired descriptive 95% interval |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, report, rows in runs:
        s = report["summary"]
        delta = s["paired_mean_overlay_minus_base"]
        counts = [f"{s[key]['correct']}/{s[key]['n']} ({pct(s[key]['accuracy'])})"
                  for key in ("base", "overlay_seed_11", "overlay_seed_23")]
        lines.append(f"| {report['manifest']['model']} | " + " | ".join(counts) +
                     f" | {delta['delta']*100:+.2f} pp | [{delta['ci95'][0]*100:+.2f}, {delta['ci95'][1]*100:+.2f}] pp |")
    lines += ["", "Intervals resample questions, keeping the two overlay repetitions together. They are descriptive intervals for this authored case set, not population-level guarantees. Sixteen variants of a task template do not provide sixteen independent task families. An interval containing zero does not establish a reliable improvement or prove that no effect exists.", "",
              "## Accuracy by task", "", "| Model | Task | Base correct / 16 | Overlay seed 11 / 16 | Overlay seed 23 / 16 |", "|---|---|---:|---:|---:|"]
    for name, report, rows in runs:
        s = report["summary"]
        for category in s["base"]["by_category"]:
            counts = [str(s[key]["by_category"][category]["correct"]) for key in ("base", "overlay_seed_11", "overlay_seed_23")]
            lines.append(f"| {report['manifest']['model']} | {category} | " + " | ".join(counts) + " |")
    lines += ["", "## Time and additional computation", "",
              "| Model | Median base turn | Median overlay turn | Mean base forward calls | Mean overlay forward calls | Mean processed-position ratio |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, report, rows in runs:
        base = [r for r in rows if r["condition"] == "base"]
        overlay = [r for r in rows if r["condition"] == "overlay"]
        mean = lambda group, key: statistics.mean(r[key] for r in group)
        lines.append(f"| {report['manifest']['model']} | {statistics.median(r['turn_seconds'] for r in base):.2f} s | {statistics.median(r['turn_seconds'] for r in overlay):.2f} s | {mean(base, 'forward_calls'):.1f} | {mean(overlay, 'forward_calls'):.1f} | {mean(overlay, 'processed_token_positions') / mean(base, 'processed_token_positions'):.2f}× |")
    lines += ["", "Turn latency includes perception, rumination, the answer, and post-answer bookkeeping for the overlay. It excludes construction and the explicit one-second warmup. Forward/position counts include setup, warmup, and background work until shutdown. Processed token positions are a workload proxy, not FLOPs. The baseline has less compute; this study does not test the best use of an equal compute budget.", "",
              "## Controls, cutoffs, and runtime health", ""]
    for name, report, rows in runs:
        s = report["summary"]
        f = s["control_fidelity"]
        overlay = [r for r in rows if r["condition"] == "overlay"]
        rounds = [r.get("telemetry", {}).get("speculative_rounds", 0) for r in overlay]
        failures = sum(r.get("telemetry", {}).get("speculative_failures", 0) for r in overlay)
        lines.append(f"- **{report['manifest']['model']}**: steering-off control had {f['response_disagreements']}/{f['n']} exact-response disagreements with baseline. Overlay search completed at least one round in {sum(n > 0 for n in rounds)}/{len(rounds)} trials; recorded rollout failures: {failures}. Trial errors: {sum(bool(r['error']) for r in rows)}.")
        for key in ("base", "overlay_seed_11", "overlay_seed_23"):
            lines.append(f"  - {key}: {s[key]['truncated']}/{s[key]['n']} answer cutoffs; FINAL-format rate {pct(s[key]['format_rate'])}.")
        base = {r["case_id"]: r for r in rows if r["condition"] == "base"}
        for seed in (11, 23):
            selected = [r for r in overlay if r["seed"] == seed]
            different = sum(r["response"] != base[r["case_id"]]["response"] for r in selected)
            lines.append(f"  - Overlay seed {seed}: {s[f'overlay_seed_{seed}']['wins']} incorrect→correct changes, {s[f'overlay_seed_{seed}']['losses']} correct→incorrect changes; {different}/{len(selected)} response strings changed.")
    lines += ["", "## Every change in correctness", "", "| Model | Case | Seed | Expected | Base parsed answer | Overlay parsed answer | Change |", "|---|---|---:|---|---|---|---|"]
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")[:100]
    for name, report, rows in runs:
        base = {r["case_id"]: r for r in rows if r["condition"] == "base"}
        for row in rows:
            if row["condition"] == "overlay" and row["correct"] != base[row["case_id"]]["correct"]:
                b = base[row["case_id"]]
                lines.append(f"| {report['manifest']['model']} | {row['case_id']} | {row['seed']} | {escape(row['expected'])} | {escape(b['parsed_answer'])} | {escape(row['parsed_answer'])} | {'gain' if row['correct'] else 'regression'} |")
    lines += ["", "## Scope and reproducibility", "",
              "The four task families test arithmetic, six-step state tracking, ordering constraints, and updates recalled from supplied conversation history. Each episode starts with a fresh controller. History is replayed through perception using fixed assistant acknowledgements; the base model sees exactly the same history. Default regions and search budgets run with autonomous speech suppressed to prevent extra unmatched turns.", "",
              "The fixed 384-token answer limit can still truncate long reasoning. Parse errors and output-format failures can lower accuracy, so this is not a pure intelligence score. Two seeds sample some controller variability; asynchronous scheduling prevents exact replay. There is no held-out public benchmark, long-duration adaptation study, blinded human evaluation, or matched-compute optimized baseline here.", "",
              "The separate seed-901 harness smoke caused a uniform increase from 128 to 384 tokens before the scored runs; those smoke trials are excluded. No scoring rule, task prompt, answer, or engine coefficient was tuned on scored outcomes.", "",
              "Manifest note: `engine_config` serializes the default configuration template, including its placeholder hidden dimension and CPU device. `device` in the manifest records the actual device; actual dimensions come from each saved model profile and are supplied automatically by `V2Engine.from_profile`. All other listed clock, bus, and search defaults apply. Profile files and an evaluator source snapshot accompany this report.", "",
              "Files:", "", "- [Frozen protocol](../../REASONING_PROTOCOL.md)", "- [All questions and computed answers](../../reasoning_pilot.jsonl)"]
    for name, report, rows in runs:
        lines += [f"- {report['manifest']['model']}: [summary and manifest]({name}/report.json), [every raw response]({name}/responses.jsonl)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    destination = args.root / "REPORT.md"
    destination.write_text(build_report(args.root), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
