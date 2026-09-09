"""Bounded branch-and-reconsider search over hypothetical continuations.

This is a search controller, not a validated simulator of the external world.
The rollout callback supplies model predictions; evaluate supplies an explicit
heuristic. No node is an observation and no search step writes into memory.
"""

from dataclasses import dataclass
import time
from typing import Callable


@dataclass
class SearchResult:
    nodes: list
    leaves: list
    attempts: int
    tokens: int
    budget_exhausted: bool


def search_futures(roots: list, rollout: Callable, evaluate: Callable, *,
                   max_depth=2, branching_factor=2, beam_width=2,
                   max_nodes=12, max_tokens=128, budget_s=0.5,
                   discount=0.8, clock=time.monotonic) -> SearchResult:
    """Breadth first beam search, then rank terminal paths by discounted utility.

    rollout(seed, parent, deadline, remaining_tokens) returns an outcome dict
    or None. A child receives its parent's actual outcome and context. Only
    the strongest beam is expanded; unexpanded nodes remain alternatives.
    Limits count attempted rollouts, including failures. The time limit is
    cooperative: an in-flight model forward cannot be preempted.
    """
    if min(max_depth, branching_factor, beam_width, max_nodes, max_tokens) < 1:
        raise ValueError("Search count limits must be positive")
    if budget_s <= 0 or not 0 < discount <= 1:
        raise ValueError("budget_s must be positive and discount in (0, 1]")
    deadline = clock() + budget_s
    nodes, frontier = [], [(seed, None) for seed in roots]
    attempts = tokens = 0
    exhausted = False
    for depth in range(1, max_depth + 1):
        level = []
        for seed, parent in frontier:
            if attempts >= max_nodes or tokens >= max_tokens or clock() >= deadline:
                exhausted = True
                break
            attempts += 1
            result = rollout(seed, parent, deadline, max_tokens - tokens)
            if result is None:
                continue
            used = int(result["tokens_used"])
            if used < 0 or used > max_tokens - tokens or (used == 0 and not result.get("failed")):
                raise ValueError("Rollout violated its token budget")
            tokens += used
            if result.get("failed"):
                continue
            local = float(evaluate(result, seed))
            weight = discount ** (depth - 1)
            total = weight * local + (parent["utility_sum"] if parent else 0.0)
            mass = weight + (parent["utility_mass"] if parent else 0.0)
            node = dict(result, id=len(nodes), parent_id=parent["id"] if parent else None,
                        depth=depth, seed=seed, local_utility=local,
                        utility_sum=total, utility_mass=mass, utility=total / mass,
                        epistemic_type="hypothesis")
            nodes.append(node)
            level.append(node)
        if exhausted or not level or depth == max_depth:
            break
        level.sort(key=lambda node: -node["utility"])
        frontier = [(node["seed"], node) for node in level[:beam_width]
                    for _ in range(branching_factor)]
    parents = {node["parent_id"] for node in nodes if node["parent_id"] is not None}
    leaves = sorted((node for node in nodes if node["id"] not in parents),
                    key=lambda node: -node["utility"])
    return SearchResult(nodes, leaves, attempts, tokens, exhausted)
