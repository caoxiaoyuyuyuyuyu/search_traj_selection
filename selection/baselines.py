"""Baseline trajectory selection methods.

Implements selection strategies for comparison with RSR and SD-RSR:
1. Random selection
2. Correctness-only filtering (SimpleDeepSearcher/OpenSeeker style)
3. TopoCurate-style structural filtering adapted to search-reasoning
"""

import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of trajectory selection."""
    trajectory_id: str
    selected: bool
    score: float
    method: str
    metadata: dict[str, Any] | None = None


def random_selection(
    trajectories: list[dict],
    k: int,
    seed: int = 42,
) -> list[dict]:
    """Random trajectory selection (baseline).

    Args:
        trajectories: List of trajectory dicts with at least "id" key.
        k: Number to select.
        seed: Random seed.

    Returns:
        Selected trajectories.
    """
    rng = random.Random(seed)
    return rng.sample(trajectories, min(k, len(trajectories)))


def correctness_selection(
    trajectories: list[dict],
    k: int,
    seed: int = 42,
) -> list[dict]:
    """Correctness-only filtering (SimpleDeepSearcher/OpenSeeker baseline).

    Keeps only trajectories where the final answer is correct, then samples k.

    Args:
        trajectories: List of trajectory dicts with "id", "answer_correct" keys.
        k: Number to select.
        seed: Random seed.

    Returns:
        Selected trajectories.
    """
    correct = [t for t in trajectories if t.get("answer_correct", False)]
    logger.info("Correctness filter: %d/%d correct", len(correct), len(trajectories))

    if len(correct) <= k:
        return correct

    rng = random.Random(seed)
    return rng.sample(correct, k)


def topocurate_style_selection(
    trajectories: list[dict],
    k: int,
    diversity_weight: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """TopoCurate-style structural selection adapted for search-reasoning.

    Scores trajectories on structural quality metrics:
    1. Search depth: number of search steps (more = deeper exploration)
    2. Query diversity: unique query patterns (avoid repetitive searches)
    3. Evidence coverage: fraction of supporting facts retrieved
    4. Reflective recovery: presence of query refinement after failed searches

    This is student-agnostic — it evaluates trajectory quality independent of
    which student will learn from it.

    Args:
        trajectories: List of trajectory dicts with structural metadata.
        k: Number to select.
        diversity_weight: Weight for diversity-based reranking.
        seed: Random seed for tie-breaking.

    Returns:
        Selected trajectories.
    """
    scored: list[tuple[float, dict]] = []

    for traj in trajectories:
        meta = traj.get("metadata", {})

        # Structural quality components
        search_depth = meta.get("num_search_steps", 0)
        num_unique_queries = meta.get("num_unique_queries", 0)
        total_queries = meta.get("num_queries", max(1, num_unique_queries))
        query_diversity = num_unique_queries / total_queries if total_queries > 0 else 0
        evidence_coverage = meta.get("evidence_coverage", 0.0)
        has_refinement = 1.0 if meta.get("has_query_refinement", False) else 0.0

        # Composite score (higher = better structural quality)
        score = (
            0.25 * min(search_depth / 5.0, 1.0)  # normalize search depth
            + 0.25 * query_diversity
            + 0.30 * evidence_coverage
            + 0.20 * has_refinement
        )

        scored.append((score, traj))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Diversity-aware top-k selection (greedy MMR-style)
    if diversity_weight > 0 and k < len(scored):
        selected = _mmr_select(scored, k, diversity_weight, seed)
    else:
        selected = [t for _, t in scored[:k]]

    return selected


def _mmr_select(
    scored: list[tuple[float, dict]],
    k: int,
    lambda_div: float,
    seed: int,
) -> list[dict]:
    """Maximal Marginal Relevance selection for diversity.

    Simple version: uses dataset source as diversity dimension.
    """
    selected: list[dict] = []
    remaining = list(scored)
    rng = random.Random(seed)

    while len(selected) < k and remaining:
        if not selected:
            # First pick: highest score
            best_idx = 0
        else:
            # Balance quality vs diversity
            selected_datasets = {s.get("dataset", "") for s in selected}
            best_score = -float("inf")
            best_idx = 0
            for i, (score, traj) in enumerate(remaining):
                diversity_bonus = 1.0 if traj.get("dataset", "") not in selected_datasets else 0.0
                combined = (1 - lambda_div) * score + lambda_div * diversity_bonus
                if combined > best_score:
                    best_score = combined
                    best_idx = i

        _, chosen = remaining.pop(best_idx)
        selected.append(chosen)

    return selected
