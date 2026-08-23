#!/usr/bin/env python3
"""One-time data preprocessing for the GRASP-RAG dashboard.

Run from GRASP-RAG root:
    python3 dashboard/preprocess.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "dashboard", "public", "data")
os.makedirs(OUT, exist_ok=True)

DOMAINS = ["agriculture", "cs", "legal", "mix"]
METHODS = ["grasp_rag", "naive_rag", "lightrag", "pathrag"]
DIMS    = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. BER results  (all four domains)
# ─────────────────────────────────────────────────────────────────────────────
print("Compiling BER results...")
ber_out = {}
for domain in DOMAINS:
    path = os.path.join(ROOT, "data", "results", f"{domain}_bridge_gt_eval.json")
    if not os.path.exists(path):
        print(f"  WARN: {path} not found — skipping")
        continue
    with open(path) as f:
        d = json.load(f)

    # Also load bridge questions file for ground-truth src/tgt metadata
    q_meta = {}
    qpath = os.path.join(ROOT, "data", "questions", f"{domain}_bridge_gt.json")
    if os.path.exists(qpath):
        with open(qpath) as f:
            for q in json.load(f):
                q_meta[q["question"]] = {
                    "src": q.get("src", ""),
                    "bridge": q.get("bridge", q.get("ground_truth", "")),
                    "tgt": q.get("tgt", ""),
                    "ground_truth": q.get("ground_truth", ""),
                }

    # Attach ground_truth info to each miss entry
    misses_annotated = {}
    for method, miss_list in d.get("misses", {}).items():
        annotated = []
        for m in miss_list:
            meta = q_meta.get(m["question"], {})
            annotated.append({**m, **meta})
        misses_annotated[method] = annotated

    ber_out[domain] = {
        "results": d["results"],
        "misses": misses_annotated,
        "q_meta": q_meta,
    }

with open(os.path.join(OUT, "ber_results.json"), "w") as f:
    json.dump(ber_out, f, indent=2, ensure_ascii=False)
print(f"  → ber_results.json ({len(ber_out)} domains)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM-as-judge results  (agriculture_typed_eval.json)
# ─────────────────────────────────────────────────────────────────────────────
print("Compiling judge results...")
judge_path = os.path.join(ROOT, "data", "results", "agriculture_typed_eval.json")

# Build question → query_type map from JSONL
q2type = {}
ans_path = os.path.join(ROOT, "data", "answers", "agriculture_typed.jsonl")
if os.path.exists(ans_path):
    with open(ans_path) as f:
        for line in f:
            r = json.loads(line)
            q2type[r["question"]] = r.get("query_type", "unknown")

judge_records = []
if os.path.exists(judge_path):
    with open(judge_path) as f:
        raw = json.load(f)
    for rec in raw:
        q = rec["question"]
        judge_records.append({
            "question": q,
            "query_type": q2type.get(q, "unknown"),
            "outcome": rec["outcome"],
            "domain": rec.get("domain", "agriculture"),
        })

# Aggregate summary counts for quick display
def aggregate_judge(records):
    by_dim   = {d: {"Win": 0, "Loss": 0, "Tie": 0} for d in DIMS}
    by_type  = {}
    overall  = {"Win": 0, "Loss": 0, "Tie": 0}

    for rec in records:
        qt = rec.get("query_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {"Win": 0, "Loss": 0, "Tie": 0}
        for dim in DIMS:
            outcome = rec["outcome"].get(dim, "Tie")
            by_dim[dim][outcome]  += 1
            by_type[qt][outcome] += 1
            overall[outcome]     += 1

    return {"by_dim": by_dim, "by_type": by_type, "overall": overall}

summary = aggregate_judge(judge_records)

with open(os.path.join(OUT, "judge_results.json"), "w") as f:
    json.dump({"records": judge_records, "summary": summary}, f,
              indent=2, ensure_ascii=False)
print(f"  → judge_results.json ({len(judge_records)} judge records)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Answers per domain  (for answer explorer)
# ─────────────────────────────────────────────────────────────────────────────
print("Compiling answers...")
for domain in DOMAINS:
    answers = defaultdict(lambda: {"methods": {}, "query_type": "unknown",
                                   "ground_truth": "", "src": "", "tgt": ""})

    # Load bridge ground-truth metadata
    qpath = os.path.join(ROOT, "data", "questions", f"{domain}_bridge_gt.json")
    bridge_meta = {}
    if os.path.exists(qpath):
        with open(qpath) as f:
            for q in json.load(f):
                bridge_meta[q["question"]] = {
                    "ground_truth": q.get("ground_truth", ""),
                    "src": q.get("src", ""),
                    "tgt": q.get("tgt", ""),
                }

    def ingest_jsonl(path, query_type_override=None):
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                q = r["question"]
                if answers[q]["query_type"] == "unknown":
                    answers[q]["query_type"] = (
                        query_type_override or r.get("query_type", "unknown")
                    )
                answers[q]["methods"][r["method"]] = r["answer"]
                if q in bridge_meta:
                    answers[q].update(bridge_meta[q])

    ingest_jsonl(os.path.join(ROOT, "data", "answers", f"{domain}_typed.jsonl"))
    ingest_jsonl(os.path.join(ROOT, "data", "answers", f"{domain}_bridge_gt.jsonl"),
                 query_type_override="bridge")

    out_path = os.path.join(OUT, f"answers_{domain}.json")
    with open(out_path, "w") as f:
        json.dump(dict(answers), f, indent=2, ensure_ascii=False)
    print(f"  → answers_{domain}.json ({len(answers)} questions)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph subgraphs  (top-100 nodes by degree per domain)
# ─────────────────────────────────────────────────────────────────────────────
print("Extracting graph subgraphs (top 100 nodes by degree)...")
try:
    import networkx as nx
except ImportError:
    print("  networkx not found — install with: pip3 install networkx")
    print("  Skipping graph extraction.")
    sys.exit(0)

ENTITY_TYPE_ORDER = [
    "concept", "method", "organisation", "organization",
    "person", "technology", "event", "location", "unknown",
]

for domain in DOMAINS:
    gml = os.path.join(ROOT, "data", "working_dir", domain,
                       "graph_chunk_entity_relation.graphml")
    if not os.path.exists(gml):
        print(f"  WARN: {gml} not found — skipping")
        continue

    mb = os.path.getsize(gml) // 1024 // 1024
    print(f"  [{domain}] reading {mb}MB graphml...", end=" ", flush=True)
    G = nx.read_graphml(gml)
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    print(f"{total_nodes} nodes, {total_edges} edges")

    # Top 300 nodes by degree (UI controls how many are actually displayed)
    MAX_NODES = 300
    top_pairs = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:MAX_NODES]
    top_ids   = {n for n, _ in top_pairs}
    subG      = G.subgraph(top_ids)

    nodes = []
    for node_id, deg in top_pairs:
        attrs = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "label": str(node_id)[:60],
            "entity_type": str(attrs.get("entity_type", "unknown")).lower(),
            "degree": deg,
            "description": str(attrs.get("description", ""))[:300],
        })

    edges = []
    for u, v, attrs in subG.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "keywords": str(attrs.get("keywords", ""))[:120],
            "weight": float(attrs.get("weight", 1.0)),
        })

    out_path = os.path.join(OUT, f"graph_{domain}.json")
    with open(out_path, "w") as f:
        json.dump({
            "stats": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "max_sampled_nodes": len(nodes),
                "sampled_edges": len(edges),
            },
            "nodes": nodes,   # sorted by degree descending; UI slices top-N
            "edges": edges,   # all edges among sampled nodes
        }, f, indent=2, ensure_ascii=False)
    print(f"  → graph_{domain}.json ({len(nodes)} nodes, {len(edges)} edges)")

print("\nAll done — data files written to dashboard/public/data/")
