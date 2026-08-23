"""
Experiment 2 — Classifier accuracy + per-type win rate analysis.

Compares:
  - GRASP-RAG auto-labels:  data/labels/{domain}_classified.json
  - GPT-4o oracle labels:   data/labels/{domain}_gt_labels.json

Reports:
  1. Classification accuracy per query type (how often GRASP-RAG agrees with GPT-4o)
  2. Win rate per query type (from eval results, grouped by oracle label)
  3. Cross-method win rate comparison per type

Requires:
  - classify_questions.py to have been run  (auto-labels)
  - label_questions.py to have been run      (oracle labels)
  - evaluate.py --ultradomain to have been run (eval results)

Usage:
    python experiments/eval_classifier.py --domain agriculture
    python experiments/eval_classifier.py --domain agriculture --baseline naive_rag
    python experiments/eval_classifier.py --all-domains
"""

import argparse
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

DOMAINS = ["agriculture", "cs", "legal", "health"]
QUERY_TYPES = ["single_entity", "bridge", "multi_entity", "broad"]
DIMENSIONS = ["Comprehensiveness", "Diversity", "logicality", "Coherence", "Relevance"]
FIXED_MOTIF_METHODS = ["grasp_rag_fixed_ego", "grasp_rag_fixed_bridge",
                        "grasp_rag_fixed_steiner", "grasp_rag_fixed_broad"]
BASELINES = ["naive_rag", "lightrag", "pathrag", *FIXED_MOTIF_METHODS]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_auto_labels(domain: str) -> dict[str, str]:
    """Load GRASP-RAG classifier output. Falls back to classified_type in answers."""
    path = os.path.join(_ROOT, "data", "labels", f"{domain}_classified.json")
    data = load_json(path)
    # data is {question: {classified_type, classified_anchors}}
    return {q: v["classified_type"] for q, v in data.items() if isinstance(v, dict)}


def load_gt_labels(domain: str) -> dict[str, str]:
    """Load GPT-4o oracle labels."""
    path = os.path.join(_ROOT, "data", "labels", f"{domain}_gt_labels.json")
    data = load_json(path)
    # data is {question: query_type_string}
    return {q: v for q, v in data.items() if isinstance(v, str)}


def load_eval_results(domain: str) -> list[dict]:
    path = os.path.join(_ROOT, "data", "results", f"{domain}_ultradomain_eval.json")
    data = load_json(path)
    return [r for r in data if isinstance(r, dict) and r.get("question")]


# ── Accuracy analysis ─────────────────────────────────────────────────────────

def classifier_accuracy(auto: dict[str, str], gt: dict[str, str]) -> None:
    shared = set(auto) & set(gt)
    if not shared:
        print("  No overlap between auto-labels and GT labels.")
        return

    per_type: dict[str, dict] = {
        qt: {"correct": 0, "total": 0} for qt in QUERY_TYPES
    }
    overall_correct = 0

    for q in shared:
        gt_type  = gt[q]
        auto_type = auto[q]
        if gt_type in per_type:
            per_type[gt_type]["total"] += 1
            if auto_type == gt_type:
                per_type[gt_type]["correct"] += 1
                overall_correct += 1

    total = sum(v["total"] for v in per_type.values())
    print(f"\n  Classifier Accuracy  (n={total} questions with both labels)")
    print(f"  {'Query Type':<18} {'Correct':>8} {'Total':>7} {'Accuracy%':>10}")
    print("  " + "-" * 48)
    for qt in QUERY_TYPES:
        c = per_type[qt]["correct"]
        n = per_type[qt]["total"]
        acc = 100 * c / n if n else 0.0
        flag = " ⚠" if n < 10 else ""
        print(f"  {qt:<18} {c:>8} {n:>7} {acc:>9.1f}%{flag}")
    print("  " + "-" * 48)
    overall_acc = 100 * overall_correct / total if total else 0.0
    print(f"  {'OVERALL':<18} {overall_correct:>8} {total:>7} {overall_acc:>9.1f}%")


# ── Per-type win rate (per dimension) ────────────────────────────────────────

_UNDECIDED_KEYS = ("CannotEvaluate", "PositionBias", "NoMajority")


def _empty_dim_counts() -> dict[str, int]:
    return {"Win": 0, "Loss": 0, "CannotEvaluate": 0, "PositionBias": 0, "NoMajority": 0}


def _print_type_table(counts_for_bl: dict[str, dict[str, dict[str, int]]]) -> None:
    col_w = 8
    for qt in QUERY_TYPES + ["ALL"]:
        first_dim = DIMENSIONS[0]
        c0 = counts_for_bl[qt][first_dim]
        n_total = c0["Win"] + c0["Loss"] + sum(c0.get(k, 0) for k in _UNDECIDED_KEYS)
        if n_total == 0:
            continue

        sep = "═" * 70 if qt == "ALL" else "─" * 70
        print(f"\n  {sep}")
        print(f"  [{qt.upper()}]  ({n_total} questions)")
        print(f"  {'Dimension':<18} {'Win%':>{col_w}} {'Loss%':>{col_w}} "
              f"{'CantEval%':>{col_w+2}} {'PosBias%':>{col_w+1}} {'NoMaj%':>{col_w}} {'n':>5}")
        print(f"  {'-'*70}")

        for dim in DIMENSIONS:
            c = counts_for_bl[qt][dim]
            n_decided = c["Win"] + c["Loss"]
            n_dim     = n_decided + sum(c.get(k, 0) for k in _UNDECIDED_KEYS)
            if n_dim == 0:
                continue
            w  = 100 * c["Win"]  / n_decided if n_decided else 0.0
            l  = 100 * c["Loss"] / n_decided if n_decided else 0.0
            ce = 100 * c.get("CannotEvaluate", 0) / n_dim
            pb = 100 * c.get("PositionBias", 0) / n_dim
            nm = 100 * c.get("NoMajority", 0) / n_dim
            print(f"  {dim:<18} {w:>{col_w}.1f}% {l:>{col_w}.1f}% {ce:>{col_w+1}.1f}% "
                  f"{pb:>{col_w}.1f}% {nm:>{col_w-1}.1f}% {n_decided:>4}")


def per_type_win_rate(
    eval_records: list[dict],
    gt_labels: dict[str, str],
    baselines: list[str],
) -> None:
    # counts[baseline][query_type][dimension] = {Win, Loss, CannotEvaluate, PositionBias, NoMajority}
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        bl: {qt: {d: _empty_dim_counts() for d in DIMENSIONS} for qt in QUERY_TYPES + ["ALL"]}
        for bl in baselines
    }

    for r in eval_records:
        bl = r.get("baseline")
        if bl not in baselines:
            continue
        q       = r.get("question", "")
        qt      = gt_labels.get(q, "unknown")
        outcome = r.get("outcome", {})

        for dim, verdict in outcome.items():
            if dim not in DIMENSIONS:
                continue
            if qt in counts[bl]:
                counts[bl][qt][dim][verdict] += 1
            counts[bl]["ALL"][dim][verdict] += 1

    for bl in baselines:
        print(f"\n  {'═'*70}")
        print(f"  GRASP-RAG vs {bl}  —  per dimension per oracle query type")
        print(f"  {'═'*70}")
        _print_type_table(counts[bl])


# ── Per-type win rate by SYSTEM labels (what the classifier predicted) ────────

def per_type_win_rate_system_labels(
    eval_records: list[dict],
    auto_labels: dict[str, str],
    baselines: list[str],
) -> None:
    """Win rates grouped by the system's predicted type (auto_labels), not oracle."""
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        bl: {qt: {d: _empty_dim_counts() for d in DIMENSIONS} for qt in QUERY_TYPES + ["ALL"]}
        for bl in baselines
    }

    for r in eval_records:
        bl = r.get("baseline")
        if bl not in baselines:
            continue
        q       = r.get("question", "")
        qt      = auto_labels.get(q, "unknown")
        outcome = r.get("outcome", {})

        for dim, verdict in outcome.items():
            if dim not in DIMENSIONS:
                continue
            if qt in counts[bl]:
                counts[bl][qt][dim][verdict] += 1
            counts[bl]["ALL"][dim][verdict] += 1

    for bl in baselines:
        print(f"\n  {'═'*70}")
        print(f"  GRASP-RAG vs {bl}  —  per dimension per SYSTEM query type")
        print(f"  {'═'*70}")
        _print_type_table(counts[bl])


# ── Legal failure mode analysis (#5) ─────────────────────────────────────────

def error_analysis(
    eval_records: list[dict],
    gt_labels: dict[str, str],
    auto_labels: dict[str, str],
    baseline: str = "naive_rag",
) -> None:
    """
    For questions where GRASP-RAG loses to `baseline`, show the distribution
    of system-predicted vs. oracle query types.

    A question counts as a Loss if the majority of its 5 dimension outcomes
    are Loss (more Loss than Win across dimensions).
    """
    loss_questions: list[dict] = []
    win_questions:  list[dict] = []

    seen: set[str] = set()
    per_q: dict[str, dict[str, int]] = {}

    for r in eval_records:
        if r.get("baseline") != baseline:
            continue
        if not r.get("gt_comparison"):
            continue
        q = r.get("question", "")
        if q not in per_q:
            per_q[q] = {**_empty_dim_counts(),
                        "classified_type": auto_labels.get(q),
                        "oracle_type":     gt_labels.get(q, "unknown")}
        for dim, verdict in r.get("outcome", {}).items():
            if dim in DIMENSIONS:
                per_q[q][verdict] += 1

    for q, c in per_q.items():
        entry = {"q": q, **c}
        if c["Loss"] > c["Win"]:
            loss_questions.append(entry)
        elif c["Win"] > c["Loss"]:
            win_questions.append(entry)

    total = len(per_q)
    n_loss = len(loss_questions)
    n_win  = len(win_questions)

    print(f"\n  Error analysis: GRASP-RAG vs {baseline}")
    print(f"  {total} questions  |  Win: {n_win}  Loss: {n_loss}  "
          f"Mixed/Tied: {total - n_win - n_loss}")

    # Distribution of oracle types among losses vs wins
    def type_dist(items: list[dict]) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for it in items:
            d[it.get("oracle_type", "unknown")] += 1
        return d

    def sys_dist(items: list[dict]) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for it in items:
            d[it.get("classified_type") or "unknown"] += 1
        return d

    def mismatch_rate(items: list[dict]) -> float:
        if not items:
            return 0.0
        bad = sum(1 for it in items
                  if it.get("classified_type") and
                     it["classified_type"] != it.get("oracle_type"))
        return 100 * bad / len(items)

    print(f"\n  Oracle-type distribution  (loss questions vs win questions)")
    print(f"  {'Type':<18} {'In losses':>10} {'In wins':>10}")
    print(f"  {'-'*40}")
    loss_od = type_dist(loss_questions)
    win_od  = type_dist(win_questions)
    for qt in QUERY_TYPES:
        print(f"  {qt:<18} {loss_od.get(qt, 0):>10} {win_od.get(qt, 0):>10}")

    print(f"\n  System-type distribution  (what classifier actually used)")
    print(f"  {'Type':<18} {'In losses':>10} {'In wins':>10}")
    print(f"  {'-'*40}")
    loss_sd = sys_dist(loss_questions)
    win_sd  = sys_dist(win_questions)
    for qt in QUERY_TYPES:
        print(f"  {qt:<18} {loss_sd.get(qt, 0):>10} {win_sd.get(qt, 0):>10}")

    ml = mismatch_rate(loss_questions)
    mw = mismatch_rate(win_questions)
    print(f"\n  Classifier mismatch rate:  losses {ml:.1f}%  wins {mw:.1f}%")
    print(f"  (mismatch = classified_type != oracle_type)")


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze(domain: str, baselines: list[str],
            system_labels: bool = False, error_analysis_flag: bool = False):
    print(f"\n{'═'*60}")
    print(f"  Domain: {domain.upper()}")
    print(f"{'═'*60}")

    auto   = load_auto_labels(domain)
    gt     = load_gt_labels(domain)
    evals  = load_eval_results(domain)

    if not gt:
        print("  No oracle labels found. Run label_questions.py first.")
        return

    if not auto:
        print("  No auto-labels found. Run classify_questions.py first.")
    else:
        print(f"\n  Loaded {len(auto)} auto-labels, {len(gt)} oracle labels")
        classifier_accuracy(auto, gt)

    if not evals:
        print("\n  No eval results found. Run evaluate.py --ultradomain first.")
        return

    print(f"\n  Loaded {len(evals)} eval records")

    if system_labels:
        if not auto:
            print("  --system-labels requires auto-labels. Run classify_questions.py first.")
        else:
            print("\n  [System-label grouping — by classifier-predicted type]")
            per_type_win_rate_system_labels(evals, auto, baselines)
    else:
        print("\n  [Oracle-label grouping — by GPT-4o ground-truth type]")
        per_type_win_rate(evals, gt, baselines)

    if error_analysis_flag:
        for bl in baselines:
            error_analysis(evals, gt, auto, baseline=bl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--all-domains", action="store_true",
                        help="Run all four domains")
    parser.add_argument("--baseline", nargs="+", choices=BASELINES, default=BASELINES,
                        help="Which baselines to show in per-type win rate table")
    parser.add_argument("--system-labels", action="store_true",
                        help="Group win rates by system-predicted type instead of oracle labels")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Show failure mode breakdown (which types lose, mismatch rates)")
    args = parser.parse_args()

    domains = DOMAINS if args.all_domains else [args.domain]
    for domain in domains:
        analyze(domain, args.baseline,
                system_labels=args.system_labels,
                error_analysis_flag=args.error_analysis)


if __name__ == "__main__":
    main()
