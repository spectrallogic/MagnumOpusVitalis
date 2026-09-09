"""Check pilot ground truth, answer extraction, and paired aggregation."""

import json
from pathlib import Path

import pytest

from experiments.build_reasoning_pilot import build_cases
from magnum_opus_v2.evaluate_reasoning import grade_answer, summarize


def test_frozen_dataset_regenerates_and_answers_satisfy_task_constraints():
    stored = [json.loads(line) for line in Path("experiments/reasoning_pilot.jsonl").read_text().splitlines()]
    assert stored == build_cases()
    assert len(stored) == 64 and len({c["id"] for c in stored}) == 64
    for case in stored:
        facts = case["facts"]
        if case["category"] == "arithmetic":
            packs, size, extra, sold, donated = facts
            assert int(case["answer"]) + sold + donated == packs * size + extra
        elif case["category"] == "state_tracking":
            state = dict(facts["initial"])
            total = sum(state.values())
            for source, target, amount in facts["moves"]:
                state[source] -= amount
                state[target] += amount
                assert min(state.values()) >= 0 and sum(state.values()) == total
            assert int(case["answer"]) == state[facts["query"]]
        elif case["category"] == "ordering":
            assert facts["front_to_back"].index(case["answer"]) + 1 == facts["position"]
        else:
            current = dict(zip(facts["projects"], facts["codes"]))
            current[facts["projects"][0]] = facts["codes"][3]
            assert str(current[facts["projects"][facts["query"]]]) == case["answer"]


@pytest.mark.parametrize("reply,expected,correct", [
    ("I considered 24, but it was wrong.\nFINAL: 31", "24", False),
    ("Initially 24.\nFINAL: 31", "31", True),
    ("**FINAL: 1,234**", "1234", True),
    (r"The result is \boxed{42}.", "42", True),
    ("The answer is Mira.", "Mira", True),
    ("Pia", "pia", True),
    ("", "12", False),
])
def test_scorer_uses_answer_not_incidental_ground_truth_mentions(reply, expected, correct):
    assert grade_answer(reply, expected)["correct"] is correct


def test_summary_keeps_repeated_trials_paired_with_their_case():
    rows = []
    for case_id, base, first, second in [("a", True, False, True), ("b", False, True, True)]:
        for condition, seed, correct in [("base", 11, base), ("overlay", 11, first), ("overlay", 23, second)]:
            rows.append(dict(case_id=case_id, category="fixture", condition=condition, seed=seed,
                             correct=correct, final_format=True, error=None, turn_seconds=1,
                             forward_calls=3, processed_token_positions=8, response="fixture"))
    report = summarize(rows)
    assert report["base"]["accuracy"] == 0.5
    assert report["overlay_seed_11"]["wins"] == report["overlay_seed_11"]["losses"] == 1
    assert report["paired_mean_overlay_minus_base"]["delta"] == 0.25
