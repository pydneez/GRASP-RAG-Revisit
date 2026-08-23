#!/usr/bin/env python3
"""
Compute Wilson score 95% confidence intervals for all reported win rates.

Reads win/loss counts directly from per-question records (gt_comparison=True),
so results are correct even when the eval file has been partially overwritten
by subsequent ablation runs.

Reads: data/results/{domain}_ultradomain_eval.json
Usage: python experiments/compute_ci.py
"""

import json
import math
import os
from collections import defaultdict

_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOMAINS    = ["agriculture", "cs", "legal", "health"]
BASELINES  = ["naive_rag", "lightrag", "pathrag"]
DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point_estimate, lower_bound, upper_bound) as fractions."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom  = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def load_win_counts(path: str) -> dict[str, dict[str, int]]:
    """
    Aggregate Win/Loss counts per baseline directly from per-question records.

    Using per-question records instead of summary records means we get correct
    counts even when evaluate.py was re-run with different baselines (each run
    overwrites the file, but the per-question data from each run is preserved
    within that run's output).
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"Win": 0, "Loss": 0})
    for r in records:
        if not r.get("question") or not r.get("outcome"):
            continue
        if not r.get("gt_comparison"):
            continue
        bl = r.get("baseline")
        if bl not in BASELINES:
            continue
        for dim, verdict in r["outcome"].items():
            if dim not in DIMENSIONS:
                continue
            if verdict == "Win":
                counts[bl]["Win"] += 1
            elif verdict == "Loss":
                counts[bl]["Loss"] += 1
    return counts


def main():
    print(f"\n{'═' * 76}")
    print("  GRASP-RAG win rates with 95% Wilson confidence intervals")
    print(f"{'═' * 76}")
    print(f"  {'Domain':<12} {'vs Baseline':<24} {'Win%':>7}  {'95% CI':>17}  {'n decided':>10}")
    print(f"  {'-' * 73}")

    for domain in DOMAINS:
        path = os.path.join(_ROOT, "data", "results",
                            f"{domain}_ultradomain_eval.json")
        if not os.path.exists(path):
            print(f"  {domain:<12}  (no results file found)")
            continue

        counts = load_win_counts(path)

        first = True
        for baseline in BASELINES:
            c = counts.get(baseline)
            if c is None or (c["Win"] + c["Loss"]) == 0:
                continue
            total_wins    = c["Win"]
            total_decided = c["Win"] + c["Loss"]

            p, lo, hi = wilson_ci(total_wins, total_decided)
            domain_label = domain if first else ""
            first = False
            print(f"  {domain_label:<12} vs {baseline:<20} {100*p:>6.1f}%  "
                  f"[{100*lo:>5.1f}%, {100*hi:>5.1f}%]  {total_decided:>10}")

        if not first:
            print(f"  {'-' * 73}")

    print()


if __name__ == "__main__":
    main()
