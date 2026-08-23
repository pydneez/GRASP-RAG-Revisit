#!/usr/bin/env python3
"""
Analysis of undecided outcomes in the multi-judge evaluation (Issue #3).

Undecided outcomes used to collapse into one generic "Undefined" bucket. They're
now split into three distinct causes (see evaluate.py::majority_vote() docstring):
  - CannotEvaluate: >=2 judges agreed neither answer meaningfully addressed the question
  - PositionBias:   a judge's verdict flipped between forward/reversed ordering
  - NoMajority:     judges gave real verdicts but disagreed, no 2/3 majority
All three are still excluded from win-rate denominators, same as before.

Reports:
  1. Undecided rate by oracle query type  — does any motif systematically fail?
  2. Undecided rate by domain and baseline
  3. Whether undecided cases cluster in any type vs. being uniform noise,
     and which of the three causes dominates

Reads: data/results/{domain}_ultradomain_eval.json
       data/labels/{domain}_gt_labels.json
Usage: python experiments/analyze_undefined.py
       python experiments/analyze_undefined.py --domain legal
"""

import argparse
import json
import os
from collections import defaultdict

_ROOT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOMAINS    = ["agriculture", "cs", "legal", "health"]
FIXED_MOTIF_METHODS = ["grasp_rag_fixed_ego", "grasp_rag_fixed_bridge",
                        "grasp_rag_fixed_steiner", "grasp_rag_fixed_broad"]
BASELINES  = ["naive_rag", "lightrag", "pathrag", *FIXED_MOTIF_METHODS]
DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]
QUERY_TYPES = ["single_entity", "bridge", "multi_entity", "broad"]
UNDECIDED_KEYS = ("CannotEvaluate", "PositionBias", "NoMajority")


def load_gt_labels(domain: str) -> dict[str, str]:
    path = os.path.join(_ROOT, "data", "labels", f"{domain}_gt_labels.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q: v for q, v in data.items() if isinstance(v, str)}


def load_eval_results(domain: str) -> list[dict]:
    path = os.path.join(_ROOT, "data", "results", f"{domain}_ultradomain_eval.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data if isinstance(r, dict) and r.get("question")
            and r.get("gt_comparison")]


def analyze_domain(domain: str, baselines: list[str]) -> None:
    gt     = load_gt_labels(domain)
    evals  = load_eval_results(domain)

    if not evals:
        print(f"  {domain}: no gt_comparison eval records found "
              f"(run evaluate.py --ultradomain first)")
        return

    # counts[baseline][query_type] = {total, undef, CannotEvaluate, PositionBias, NoMajority}
    counts: dict[str, dict[str, dict[str, int]]] = {
        bl: {qt: {"total": 0, "undef": 0, **{k: 0 for k in UNDECIDED_KEYS}}
             for qt in QUERY_TYPES + ["ALL"]}
        for bl in baselines
    }

    for r in evals:
        bl = r.get("baseline")
        if bl not in baselines:
            continue
        q  = r.get("question", "")
        qt = gt.get(q, "unknown")
        for dim, verdict in r.get("outcome", {}).items():
            if dim not in DIMENSIONS:
                continue
            is_undecided = verdict in UNDECIDED_KEYS
            for bucket in (counts[bl]["ALL"], counts[bl].get(qt, {})):
                if not bucket:
                    continue
                bucket["total"] += 1
                if is_undecided:
                    bucket["undef"] += 1
                    bucket[verdict] += 1

    print(f"\n{'═'*68}")
    print(f"  {domain.upper()}  —  undecided rate by oracle query type")
    print(f"{'═'*68}")

    has_data = False
    for bl in baselines:
        if counts[bl]["ALL"]["total"] == 0:
            continue
        has_data = True
        all_c = counts[bl]["ALL"]
        overall_u = 100 * all_c["undef"] / all_c["total"]
        print(f"\n  vs {bl}  (overall undecided% = {overall_u:.1f}%: "
              f"CantEval {all_c['CannotEvaluate']}, PosBias {all_c['PositionBias']}, "
              f"NoMaj {all_c['NoMajority']})")
        print(f"  {'Oracle type':<18} {'Undec%':>8} {'n dims':>8}  note")
        print(f"  {'-'*55}")
        for qt in QUERY_TYPES:
            c = counts[bl][qt]
            if c["total"] == 0:
                continue
            u = 100 * c["undef"] / c["total"]
            delta = u - overall_u
            flag = "  ▲ higher" if delta > 5 else ("  ▼ lower" if delta < -5 else "")
            print(f"  {qt:<18} {u:>7.1f}% {c['total']:>8}{flag}")

    if not has_data:
        print(f"  No data for baselines {baselines} in {domain}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default=None,
                        help="Single domain (default: all four)")
    parser.add_argument("--baseline", nargs="+", choices=BASELINES, default=BASELINES)
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    for domain in domains:
        analyze_domain(domain, args.baseline)
    print()


if __name__ == "__main__":
    main()
