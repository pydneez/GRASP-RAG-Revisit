#!/usr/bin/env python3
"""
Per-judge win rate breakdown — confound-resolution check for the redone judge panel.

Each judge's verdict is bias-corrected independently (same logic as evaluate.py):
  Win  = forward says Answer 1  AND  reverse says Answer 2
  Loss = forward says Answer 2  AND  reverse says Answer 1
  Tie  = inconsistent orderings (position bias — excluded from win rate)

The original paper found GPT-4o-mini (which was both the answer generator AND a
judge) awarded GRASP-RAG a substantially higher win rate than Claude Haiku or
Gemini Flash independently reported — self-preference inflation. This redo
removes OpenAI from the judge panel entirely specifically to fix that. This
script now checks whether that fix worked: do the three remaining judges
(none of which generate any answers) agree with each other within a tight
band? A large residual spread here would be a *different* finding (e.g. one
of the three judges has its own systematic bias) worth investigating on its
own terms, not assumed away just because OpenAI left the panel.

Reads: data/results/{domain}_ultradomain_eval.json
       data/results/{domain}_typed_balanced_eval.json  (once Part B exists)
Usage: python experiments/judge_bias.py
       python experiments/judge_bias.py --domain agriculture
"""

import argparse
import json
import os
from collections import defaultdict

_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOMAINS    = ["agriculture", "cs", "legal", "health"]
BASELINES  = ["naive_rag", "lightrag", "pathrag", "grasp_rag_fixed_ego",
              "grasp_rag_fixed_bridge", "grasp_rag_fixed_steiner", "grasp_rag_fixed_broad"]
DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]
JUDGES     = ["claude", "gemini", "deepseek"]
JUDGE_LABELS = {
    "claude":   "Claude Haiku 4.5   (judge only)",
    "gemini":   "Gemini 3.6 Flash   (judge only)",
    "deepseek": "DeepSeek V4 Flash  (judge only)",
}
EVAL_SUFFIXES = ["_ultradomain", "_typed_balanced"]


def bias_correct(fwd: dict, rev: dict) -> dict[str, str]:
    """Return per-dimension bias-corrected verdict for GRASP-RAG."""
    result = {}
    for dim in DIMENSIONS:
        f = fwd.get(dim, "")
        r = rev.get(dim, "")
        if f == "Answer 1" and r == "Answer 2":
            result[dim] = "Win"
        elif f == "Answer 2" and r == "Answer 1":
            result[dim] = "Loss"
        else:
            result[dim] = "Tie"
    return result


def analyze(domains: list[str]):
    for domain in domains:
        for suffix in EVAL_SUFFIXES:
            path = os.path.join(_ROOT, "data", "results", f"{domain}{suffix}_eval.json")
            if not os.path.exists(path):
                continue

            with open(path, encoding="utf-8") as f:
                records = json.load(f)

            # counts[baseline][judge] = {Win, Loss, Tie}
            counts: dict[str, dict[str, dict[str, int]]] = {
                bl: {j: {"Win": 0, "Loss": 0, "Tie": 0} for j in JUDGES}
                for bl in BASELINES
            }

            for r in records:
                if not r.get("question") or not r.get("per_judge") or not r.get("outcome"):
                    continue
                baseline = r.get("baseline")
                if baseline not in BASELINES:
                    continue
                # Only count GT-anchored comparisons (ultradomain / typed_balanced)
                if not r.get("gt_comparison"):
                    continue
                pj = r["per_judge"]
                for judge in JUDGES:
                    fwd = pj.get(f"{judge}_fwd", {})
                    rev = pj.get(f"{judge}_rev", {})
                    if not fwd and not rev:
                        continue
                    bc = bias_correct(fwd, rev)
                    for verdict in bc.values():
                        counts[baseline][judge][verdict] += 1

            has_data = any(
                counts[bl][j]["Win"] + counts[bl][j]["Loss"] > 0
                for bl in BASELINES for j in JUDGES
            )
            if not has_data:
                continue

            part_label = "Part A (natural)" if suffix == "_ultradomain" else "Part B (balanced)"
            print(f"\n{'═' * 72}")
            print(f"  {domain.upper()} — {part_label} — per-judge bias-corrected win rates")
            print(f"{'═' * 72}")

            for baseline in BASELINES:
                if not any(counts[baseline][j]["Win"] + counts[baseline][j]["Loss"] > 0
                           for j in JUDGES):
                    continue
                print(f"\n  vs {baseline}")
                print(f"  {'Judge':<36} {'Win%':>7} {'Loss%':>7} {'Tie%':>7} {'n decided':>10}")
                print(f"  {'-' * 68}")

                for judge in JUDGES:
                    c = counts[baseline][judge]
                    n_decided = c["Win"] + c["Loss"]
                    n_total   = n_decided + c["Tie"]
                    if n_total == 0:
                        continue
                    w = 100 * c["Win"]  / n_decided if n_decided else 0.0
                    l = 100 * c["Loss"] / n_decided if n_decided else 0.0
                    t = 100 * c["Tie"]  / n_total
                    print(f"  {JUDGE_LABELS[judge]}  {w:>6.1f}%  {l:>6.1f}%  "
                          f"{t:>6.1f}%  {n_decided:>9}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default=None,
                        help="Single domain (default: all four)")
    args = parser.parse_args()
    domains = [args.domain] if args.domain else DOMAINS
    analyze(domains)


if __name__ == "__main__":
    main()
