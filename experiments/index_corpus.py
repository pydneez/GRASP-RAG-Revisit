"""
Step 1: Download a UltraDomain corpus slice and index it with LightRAG.

The same working_dir is shared by all methods (LightRAG baselines and GRASP-RAG),
so graph construction happens exactly once. Only the retrieval step differs per method.

Usage:
    python experiments/index_corpus.py --domain agriculture
    python experiments/index_corpus.py --domain cs
    python experiments/index_corpus.py --domain legal
    python experiments/index_corpus.py --domain health

Experiment 3 (MASHQA) uses the same LightRAG construction (identical chunker,
extraction prompt, gleaning, merge, models) via a separate entry point that
sources its 521 articles from e3_prep.py's corpus.jsonl instead of downloading
an UltraDomain JSONL. Kept in this file rather than a new script per the
briefing's "prefer extending existing scripts" constraint.

    python experiments/index_corpus.py --mashqa --data data/e3_data
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_LIGHTRAG_PATH = os.path.join(_ROOT, "references", "LightRAG", "LightRAG")
sys.path.insert(0, _LIGHTRAG_PATH)
sys.path.insert(0, _ROOT)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

DOMAINS = ["agriculture", "cs", "legal", "health"]


def working_dir(domain: str) -> str:
    path = os.path.join(_ROOT, "data", "working_dir", domain)
    os.makedirs(path, exist_ok=True)
    return path


def corpus_path(domain: str) -> str:
    return os.path.join(_ROOT, "data", "corpus", f"{domain}.txt")


def download_jsonl(domain: str) -> str:
    """Download the domain JSONL file from UltraDomain on HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        sys.exit(1)

    raw_dir = os.path.join(_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading UltraDomain/{domain}.jsonl from HuggingFace ...")
    local_path = hf_hub_download(
        repo_id="TommyChien/UltraDomain",
        filename=f"{domain}.jsonl",
        repo_type="dataset",
        local_dir=raw_dir,
    )
    print(f"Downloaded → {local_path}")
    return local_path


def extract_unique_contexts(jsonl_path: str) -> list[str]:
    """Read JSONL and extract unique 'context' strings (LightRAG Step_0 logic)."""
    import json as _json
    seen, contexts = {}, []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                ctx = obj.get("context", "")
                if ctx and ctx not in seen:
                    seen[ctx] = None
                    contexts.append(ctx)
            except Exception:
                pass
    print(f"Extracted {len(contexts)} unique contexts")
    return contexts


def load_mashqa_corpus(data_dir: str) -> tuple[list[str], list[str]]:
    """Read e3_prep.py's corpus.jsonl -> (doc_ids, texts), same order."""
    import json as _json
    path = os.path.join(data_dir, "corpus.jsonl")
    doc_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            doc_ids.append(d["doc_id"])
            texts.append(d["text"])
    return doc_ids, texts


async def index_mashqa(data_dir: str, embedding_timeout: int = 90):
    doc_ids, texts = load_mashqa_corpus(data_dir)
    if not texts:
        print(f"No corpus found at {data_dir}/corpus.jsonl — run e3_prep.py first.")
        sys.exit(1)

    # Own working_dir, distinct from UltraDomain health's — these are two
    # unrelated corpora and must not share a graph.
    wd = working_dir("mashqa")
    print(f"Indexing {len(texts)} MASHQA articles into {wd} ...")

    # Explicit ids so each LightRAG chunk's full_doc_id matches e3_prep.py's
    # doc_id directly, instead of LightRAG's own chunk-id hash.
    rag = LightRAG(
        working_dir=wd,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        # LightRAG's default_embedding_timeout is 30s and the worker's hard
        # execution ceiling is 2x that (see lightrag/utils.py's
        # priority_limit_async_func_call). 30s is tight for a real embeddings
        # round-trip under any latency — this account has shown plenty, and two
        # prior runs aborted on "Worker execution timeout after 60s".
        default_embedding_timeout=embedding_timeout,
    )
    await rag.initialize_storages()
    await rag.ainsert(texts, ids=doc_ids)
    await rag.finalize_storages()
    print("Indexing complete.")


async def index(domain: str):
    # 1. Get the JSONL file (download if missing)
    raw_path = os.path.join(_ROOT, "data", "raw", f"{domain}.jsonl")
    if not os.path.exists(raw_path):
        raw_path = download_jsonl(domain)

    # 2. Extract unique contexts
    contexts = extract_unique_contexts(raw_path)
    if not contexts:
        print("No contexts found — check JSONL format.")
        sys.exit(1)

    # 3. Index with LightRAG (inserts list of strings, same as LightRAG reproduce/Step_1.py)
    wd = working_dir(domain)
    print(f"Indexing {len(contexts)} contexts into {wd} ...")

    rag = LightRAG(
        working_dir=wd,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    await rag.ainsert(contexts)
    await rag.finalize_storages()
    print("Indexing complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="agriculture")
    parser.add_argument("--mashqa", action="store_true",
                         help="Index the Experiment 3 MASHQA corpus instead of an UltraDomain --domain.")
    parser.add_argument("--data", default=os.path.join(_ROOT, "data", "e3_data"),
                         help="e3_prep.py output dir (used with --mashqa).")
    parser.add_argument("--embed-timeout", type=int, default=90,
                         help="Per-call embedding timeout in seconds, doubled internally for the "
                              "worker execution ceiling (used with --mashqa). LightRAG's own "
                              "default is 30s (60s ceiling) — too tight for this account's observed "
                              "latency; caused two prior runs to abort on 'Worker execution timeout "
                              "after 60s'.")
    args = parser.parse_args()
    if args.mashqa:
        asyncio.run(index_mashqa(args.data, embedding_timeout=args.embed_timeout))
    else:
        asyncio.run(index(args.domain))


if __name__ == "__main__":
    main()
