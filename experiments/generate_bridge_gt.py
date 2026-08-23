"""
Generate bridge ground-truth questions directly from the knowledge graph.

No LLM needed — questions are built from graph structure.
For each 2-hop pair (src → bridge → tgt), the intermediate node IS the ground truth.

Output format per question:
  {
    "question":     "How does Varroa Mite relate to Honey Bee?",
    "query_type":   "bridge",
    "src":          "Varroa Mite",
    "bridge":       "Colony Collapse Disorder",
    "tgt":          "Honey Bee",
    "ground_truth": "Colony Collapse Disorder"   ← what the answer MUST mention
  }

Usage:
    python experiments/generate_bridge_gt.py --domain agriculture --n 40
"""

import argparse
import json
import os
import random
import sys

import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

DOMAINS = ["agriculture", "cs", "legal", "mix"]

_EXCLUDED_TYPES = {
    "person", "location", "content", "UNKNOWN", "data", "document",
    "book", "event", "date", "people", "occupation", "nation",
    "publication", "timeframe", "timeperiod", "time", "currency",
}

_TEMPLATES = [
    "How does {src} relate to {tgt}?",
    "What is the connection between {src} and {tgt}?",
    "How are {src} and {tgt} linked in the context of {domain}?",
    "What connects {src} to {tgt}?",
    "Through what mechanism does {src} connect to {tgt}?",
]


def _is_clean(G: nx.Graph, node: str) -> bool:
    if "_" in node:
        return False
    attrs = G.nodes[node]
    if attrs.get("entity_type", "UNKNOWN") in _EXCLUDED_TYPES:
        return False
    if len(attrs.get("description", "")) < 30:
        return False
    return True


def find_bridge_pairs(G: nx.Graph, n: int, seed: int = 42) -> list[dict]:
    """Find n clean 2-hop bridge pairs where ground truth = intermediate node."""
    top_nodes = sorted(
        [nd for nd in G.nodes() if _is_clean(G, nd)],
        key=lambda nd: G.degree(nd),
        reverse=True,
    )[:150]

    pairs = []
    seen = set()

    for src in top_nodes:
        for tgt in top_nodes:
            if src == tgt:
                continue
            if G.has_edge(src, tgt):
                continue  # direct edge — not a bridge question
            key = tuple(sorted([src, tgt]))
            if key in seen:
                continue
            try:
                path = nx.shortest_path(G, src, tgt)
            except nx.NetworkXNoPath:
                continue
            if len(path) != 3:
                continue  # want exactly 2-hop
            bridge = path[1]
            if not _is_clean(G, bridge):
                continue
            seen.add(key)
            src_ed = dict(G.edges[src, bridge]) if G.has_edge(src, bridge) else {}
            tgt_ed = dict(G.edges[bridge, tgt]) if G.has_edge(bridge, tgt) else {}
            pairs.append({
                "src": src,
                "bridge": bridge,
                "tgt": tgt,
                "src_keywords": src_ed.get("keywords", ""),
                "tgt_keywords": tgt_ed.get("keywords", ""),
            })

    random.seed(seed)
    random.shuffle(pairs)
    return pairs[:n]


def build_questions(pairs: list[dict], domain: str) -> list[dict]:
    random.seed(42)
    questions = []
    for p in pairs:
        template = random.choice(_TEMPLATES)
        q = template.format(src=p["src"], tgt=p["tgt"], domain=domain)
        questions.append({
            "question":     q,
            "query_type":   "bridge",
            "src":          p["src"],
            "bridge":       p["bridge"],
            "tgt":          p["tgt"],
            "ground_truth": p["bridge"],   # the entity the answer MUST mention
            "src_keywords": p["src_keywords"],
            "tgt_keywords": p["tgt_keywords"],
            "domain":       domain,
        })
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of bridge questions to generate")
    args = parser.parse_args()

    graphml = os.path.join(_ROOT, "data", "working_dir", args.domain,
                           "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml):
        print(f"Graph not found: {graphml}")
        sys.exit(1)

    print(f"Loading graph for domain={args.domain}...")
    G = nx.read_graphml(graphml)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pairs = find_bridge_pairs(G, n=args.n)
    print(f"  Found {len(pairs)} clean 2-hop bridge pairs")

    questions = build_questions(pairs, args.domain)

    out_dir = os.path.join(_ROOT, "data", "questions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.domain}_bridge_gt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(questions)} questions → {out_path}")
    print("\nSample questions:")
    for q in questions[:5]:
        print(f"  Q: {q['question']}")
        print(f"     Ground truth: [{q['ground_truth']}]")
        print(f"     Path: {q['src']} → {q['bridge']} → {q['tgt']}")
        print()


if __name__ == "__main__":
    main()
