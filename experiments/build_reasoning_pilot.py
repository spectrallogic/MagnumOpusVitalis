"""Generate a fixed, objective reasoning pilot without querying any model."""

import argparse
import json
import random
from pathlib import Path


SUFFIX = " Reason briefly, then finish with a line in the form FINAL: <answer>."


def build_cases(seed=20260909, per_category=16):
    rng = random.Random(seed)
    cases = []
    for i in range(per_category):
        packs, size, extra = rng.randint(7, 29), rng.randint(6, 24), rng.randint(11, 79)
        sold, donated = rng.randint(10, 35), rng.randint(2, 9)
        count = packs * size + extra - sold - donated
        cases.append({"id": f"arithmetic-{i:02}", "category": "arithmetic", "history": [],
                      "prompt": f"A shop starts with {packs} sealed packs of {size} pencils each, plus {extra} loose pencils. It sells {sold} pencils and then donates {donated} pencils. How many pencils remain? Give an integer." + SUFFIX,
                      "answer": str(count), "facts": [packs, size, extra, sold, donated]})

    for i in range(per_category):
        initial = {label: rng.randint(10, 30) for label in ("amber", "blue", "green")}
        state = dict(initial)
        events = []
        moves = []
        for _ in range(6):
            source, target = rng.sample(list(state), 2)
            while state[source] == 0:
                source, target = rng.sample(list(state), 2)
            amount = rng.randint(1, min(8, state[source]))
            state[source] -= amount
            state[target] += amount
            events.append(f"Move {amount} tokens from {source} to {target}.")
            moves.append([source, target, amount])
        query = rng.choice(list(state))
        cases.append({"id": f"state-{i:02}", "category": "state_tracking", "history": [],
                      "prompt": "Three boxes initially contain " + ", ".join(f"{k}: {v} tokens" for k, v in initial.items()) + ". Perform these instructions in order: " + " ".join(events) + f" How many tokens are now in the {query} box? Give an integer." + SUFFIX,
                      "answer": str(state[query]), "facts": {"initial": initial, "moves": moves, "query": query}})

    for i in range(per_category):
        names = rng.sample(["Mira", "Owen", "Pia", "Theo", "Zara", "Luca", "Nora", "Arun", "Iris"], 6)
        constraints = [f"{left} stands somewhere before {right}." for left, right in zip(names, names[1:])]
        rng.shuffle(constraints)
        position = rng.randint(2, 5)
        cases.append({"id": f"ordering-{i:02}", "category": "ordering", "history": [],
                      "prompt": "Six people stand in a single line, facing the same direction. " + " ".join(constraints) + f" Who is in position {position}, counting from the front as position 1? Give only that person's name as your final answer." + SUFFIX,
                      "answer": names[position - 1], "facts": {"front_to_back": names, "position": position}})

    for i in range(per_category):
        projects = rng.sample(["Cedar", "Juniper", "Maple", "Birch", "Willow", "Aster"], 3)
        codes = rng.sample(range(1000, 9999), 4)
        history = [
            {"role": "user", "content": f"Remember this dispatch list: {projects[0]} uses code {codes[0]}, {projects[1]} uses code {codes[1]}, and {projects[2]} uses code {codes[2]}."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": f"Update: {projects[0]}'s code is now {codes[3]}; its previous code is cancelled. The other project codes stay the same."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": f"Unrelated details: the office has {rng.randint(3,9)} chairs, and delivery will arrive on Thursday. Keep the dispatch update in mind."},
            {"role": "assistant", "content": "Understood."},
        ]
        target = 0 if i % 2 == 0 else rng.choice([1, 2])
        answer = codes[3] if target == 0 else codes[target]
        cases.append({"id": f"memory-{i:02}", "category": "context_memory", "history": history,
                      "prompt": f"What is the current dispatch code for {projects[target]}? Give the four-digit code." + SUFFIX,
                      "answer": str(answer), "facts": {"projects": projects, "codes": codes, "query": target}})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/reasoning_pilot.jsonl"))
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--per-category", type=int, default=16)
    args = parser.parse_args()
    cases = build_cases(args.seed, args.per_category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
