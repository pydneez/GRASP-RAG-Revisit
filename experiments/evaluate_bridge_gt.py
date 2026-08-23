"""
Ground-truth bridge evaluation.

For each bridge question, checks whether each method's answer mentions
the correct intermediate (bridging) entity.

Metric: Bridging Entity Recall (BER) — % of questions where the answer
contains the ground-truth intermediate entity (case-insensitive substring match).

This directly tests whether graph traversal finds the right structural path,
without LLM-as-judge length bias.

Usage:
    python experiments/evaluate_bridge_gt.py --domain agriculture
"""

import argparse
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

METHODS = ["grasp_rag", "naive_rag", "lightrag", "pathrag"]


def answers_path(domain: str) -> str:
    return os.path.join(_ROOT, "data", "answers", f"{domain}_bridge_gt.jsonl")


def questions_path(domain: str) -> str:
    return os.path.join(_ROOT, "data", "questions", f"{domain}_bridge_gt.json")


def entity_in_answer(entity: str, answer: str) -> bool:
    """Case-insensitive check — also tries each word of multi-word entities."""
    entity_l = entity.lower()
    answer_l = answer.lower()
    if entity_l in answer_l:
        return True
    # Also accept if all significant words of the entity appear
    words = [w for w in entity_l.split() if len(w) > 3]
    if words and all(w in answer_l for w in words):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="agriculture")
    args = parser.parse_args()

    q_path = questions_path(args.domain)
    a_path = answers_path(args.domain)

    if not os.path.exists(q_path):
        print(f"Questions not found: {q_path}")
        print("Run generate_bridge_gt.py first.")
        return
    if not os.path.exists(a_path):
        print(f"Answers not found: {a_path}")
        print("Run run_queries.py --bridge-gt first.")
        return

    with open(q_path, encoding="utf-8") as f:
        questions = {q["question"]: q for q in json.load(f)}

    answers = defaultdict(dict)
    with open(a_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[r["question"]][r["method"]] = r["answer"]

    results = defaultdict(lambda: {"hit": 0, "miss": 0, "total": 0})
    misses = defaultdict(list)

    for q_text, method_answers in answers.items():
        q_info = questions.get(q_text)
        if not q_info:
            continue
        gt = q_info["ground_truth"]

        for method, answer in method_answers.items():
            hit = entity_in_answer(gt, answer)
            results[method]["total"] += 1
            if hit:
                results[method]["hit"] += 1
            else:
                results[method]["miss"] += 1
                misses[method].append({
                    "question": q_text,
                    "ground_truth": gt,
                    "answer_snippet": answer[:200],
                })

    print(f"\nBridge Ground-Truth Evaluation — domain={args.domain}")
    print(f"Metric: Bridging Entity Recall (BER)\n")
    print(f"{'Method':<16} {'BER':>8}  {'Hits':>6} / {'Total':>5}")
    print("-" * 42)

    ber_scores = {}
    for method in METHODS:
        r = results[method]
        if r["total"] == 0:
            print(f"  {method:<14}  no data")
            continue
        ber = r["hit"] / r["total"] * 100
        ber_scores[method] = ber
        print(f"  {method:<14}  {ber:>6.1f}%  {r['hit']:>6} / {r['total']:>5}")

    # Winner
    if ber_scores:
        winner = max(ber_scores, key=ber_scores.get)
        print(f"\nBest method: {winner} ({ber_scores[winner]:.1f}%)")

    # Show what GRASP-RAG missed that others got right
    grasp_misses = {m["question"] for m in misses.get("grasp_rag", [])}
    naive_hits  = {q for q, ans in answers.items()
                   if entity_in_answer(questions.get(q, {}).get("ground_truth", ""),
                                       ans.get("naive_rag", ""))}
    unique_grasp_misses = grasp_misses & naive_hits

    if unique_grasp_misses:
        print(f"\nQuestions naive_rag got right but GRASP-RAG missed ({len(unique_grasp_misses)}):")
        for q in list(unique_grasp_misses)[:3]:
            gt = questions[q]["ground_truth"]
            print(f"  Q: {q}")
            print(f"     Ground truth: [{gt}]")
            grasp_ans = answers[q].get("grasp_rag", "")[:150]
            print(f"     GRASP answer: {grasp_ans}...")
            print()

    # Save detailed results
    out = {
        "domain": args.domain,
        "metric": "bridging_entity_recall",
        "results": {m: {"ber": results[m]["hit"] / max(results[m]["total"], 1) * 100,
                         **results[m]}
                    for m in METHODS},
        "misses": dict(misses),
    }
    out_path = os.path.join(_ROOT, "data", "results",
                            f"{args.domain}_bridge_gt_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Detailed results → {out_path}")


if __name__ == "__main__":
    main()
