"""
Experiment 3 — ad-hoc KV-store / retrieval inspector.

Not part of the pipeline — a debugging tool for manually answering "what did
GRASP-RAG actually retrieve for this question, and why." Grew out of manually
diagnosing the swimming-pool naive_rag-win case; see EXPERIMENT_3.md /
conversation history for that investigation.

Usage:
    # What does a graph node look like, and which chunks (with source
    # article) does it link to?
    python experiments/e3_inspect.py --node "Swimming Pool"

    # Raw content of one chunk, plus which article it's from
    python experiments/e3_inspect.py --chunk-id <id>

    # Reproduce the exact retrieval context for a question, given its
    # already-known anchors (no classifier call needed, fully deterministic;
    # get anchors from an existing answer_quality.jsonl/classified.jsonl
    # record's "classified_anchors" field)
    python experiments/e3_inspect.py --question "How can you help your child avoid swimming pool and water parks germs?" \\
        --type multi_entity --anchors "Child,Swimming Pool,Water Park,Germs" --out context_dump.txt

    # What article is a given qid/doc_id actually about
    python experiments/e3_inspect.py --doc-id doc_1b59b69c748b
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _ROOT)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_kv(working_dir, name):
    path = os.path.join(working_dir, name)
    if not os.path.exists(path):
        print(f"Not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def show_node(working_dir, node_name):
    from grasp_rag import GRASPRAG
    grasp = GRASPRAG(working_dir=working_dir, cascade=True)
    await grasp.initialize()
    graph = grasp._rag.chunk_entity_relation_graph
    nd = await graph.get_node(node_name)
    if not nd:
        print(f"No node named {node_name!r} in the graph.")
        await grasp.finalize()
        return

    print(f"[Node: {node_name}]  type={nd.get('entity_type')}")
    print(f"Description:\n{nd.get('description', '')}\n")

    entity_chunks = load_kv(working_dir, "kv_store_entity_chunks.json")
    text_chunks = load_kv(working_dir, "kv_store_text_chunks.json")
    chunk_ids = entity_chunks.get(node_name, {}).get("chunk_ids", [])
    print(f"Linked chunks: {len(chunk_ids)} total (first 3 are what "
          f"_multi_entity_context() actually uses)\n")
    for i, cid in enumerate(chunk_ids):
        c = text_chunks.get(cid, {})
        marker = " <-- used" if i < 3 else ""
        print(f"  [{i}] chunk_id={cid}  doc_id={c.get('full_doc_id')}{marker}")
        print(f"      {c.get('content', '')[:200]}...")
        print()

    await grasp.finalize()


def show_chunk(working_dir, chunk_id):
    text_chunks = load_kv(working_dir, "kv_store_text_chunks.json")
    c = text_chunks.get(chunk_id)
    if not c:
        print(f"No chunk with id {chunk_id!r}")
        return
    print(f"doc_id: {c.get('full_doc_id')}")
    print(f"chunk_order_index: {c.get('chunk_order_index')}")
    print(f"tokens: {c.get('tokens')}")
    print(f"\ncontent:\n{c.get('content', '')}")


def show_doc(data_dir, doc_id):
    corpus = load_jsonl(os.path.join(data_dir, "corpus.jsonl"))
    doc = next((d for d in corpus if d.get("doc_id") == doc_id), None)
    if not doc:
        print(f"No article with doc_id {doc_id!r} in corpus.jsonl")
        return
    print(f"[Article {doc_id}]")
    print(doc.get("context", doc.get("text", ""))[:2000])


async def show_retrieval(working_dir, question, query_type, anchors, out_path):
    from grasp_rag import GRASPRAG
    grasp = GRASPRAG(working_dir=working_dir, cascade=True)
    await grasp.initialize()
    grasp._ensure_retriever()

    paths = grasp._retriever.retrieve_direct(query_type, anchors)
    print("Retrieved motif paths:")
    for p in paths:
        print(" ", p)

    context = await grasp._build_structured_context(paths, query_type, question)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(context)
        print(f"\nFull context ({len(context)} chars) written to {out_path}")
    else:
        print(f"\n=== Full context ({len(context)} chars) ===")
        print(context)

    await grasp.finalize()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--working-dir", default=os.path.join(_ROOT, "data", "working_dir", "mashqa"))
    p.add_argument("--data", default=os.path.join(_ROOT, "data", "e3_data"))
    p.add_argument("--node", help="Show a graph node's description + linked chunks (with source doc_id)")
    p.add_argument("--chunk-id", help="Show one raw chunk's content + source doc_id")
    p.add_argument("--doc-id", help="Show an article's raw text by doc_id")
    p.add_argument("--question", help="Reproduce retrieval context for a question (needs --type and --anchors)")
    p.add_argument("--type", choices=["single_entity", "bridge", "multi_entity", "broad"])
    p.add_argument("--anchors", help="Comma-separated anchor entity names, e.g. 'Child,Swimming Pool,Germs'")
    p.add_argument("--out", help="Write retrieved context to this file instead of stdout")
    a = p.parse_args()

    if a.node:
        asyncio.run(show_node(a.working_dir, a.node))
    elif a.chunk_id:
        show_chunk(a.working_dir, a.chunk_id)
    elif a.doc_id:
        show_doc(a.data, a.doc_id)
    elif a.question:
        if not a.type or not a.anchors:
            print("--question requires --type and --anchors")
            sys.exit(1)
        anchors = [x.strip() for x in a.anchors.split(",")]
        asyncio.run(show_retrieval(a.working_dir, a.question, a.type, anchors, a.out))
    else:
        p.print_help()
