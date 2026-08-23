"""
GRASP-RAG: Graph Retrieval with Adaptive Structured Paths.

Core idea — query-adaptive motif retrieval:
  The query is classified into a motif type, which determines HOW the knowledge
  graph is traversed to build the context. The LLM then generates a comprehensive
  prose answer from that context — same generation style as LightRAG/PathRAG.

    bridge        → shortest-path traversal  (A → B → C hop chain as context)
    single_entity → 1-hop ego-network        (entity profile as context)
    multi_entity  → Steiner-tree subgraph    (per-entity profiles + connections)
    broad         → Personalized PageRank / LightRAG hybrid fallback

  The motif shapes WHAT is retrieved; prose generation is standard across all types.
"""

import asyncio
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIGHTRAG_PATH = os.path.join(_HERE, "..", "references", "LightRAG", "LightRAG")
if _LIGHTRAG_PATH not in sys.path:
    sys.path.insert(0, _LIGHTRAG_PATH)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed

from .classifier import QueryClassifier
from .llm_config import generation_complete
from .retriever import MotifRetriever

# ── Motif-specific prompts ────────────────────────────────────────────────────

# Single generation prompt used for all motif types.
#
# Why one prompt: LightRAG and PathRAG both use the same "comprehensive prose"
# instruction regardless of their context format (CSV tables vs KG chunks). The
# motif-specific structure lives in the CONTEXT itself (hop chain, entity profile,
# Steiner map) — not in the generation instruction. Separating the two means the
# judge compares answer quality fairly, not output format style.
_RESPONSE_PROMPT = """\
---Role---

You are an expert AI assistant specializing in synthesizing information from a \
provided knowledge base. Your primary function is to answer user queries accurately \
by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the context.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Context---

{context}

---Question---

{query}

Add sections and commentary to the response as appropriate for the length and format. \
Style the response in markdown.
"""

_PROMPTS = {
    "bridge":        _RESPONSE_PROMPT,
    "single_entity": _RESPONSE_PROMPT,
    "multi_entity":  _RESPONSE_PROMPT,
}

# Lightweight verifier used during cascade to score context quality.
# Kept intentionally short — we only need a scalar, not an explanation.
_VERIFIER_PROMPT = """\
---Task---

Score how well the retrieved context supports answering the question below.

---Question---

{query}

---Retrieved Context (excerpt)---

{context}

---Scoring---

1 = Irrelevant or empty
2 = Tangentially related, mostly unhelpful
3 = Partially addresses the question
4 = Contains most of the needed information
5 = Fully supports a comprehensive answer

Respond with a single integer only (1-5):"""

_UNCERTAINTY_THRESHOLD = 0.04
_CASCADE_MIN_SCORE = 2.5   # fall back to hybrid if best motif scores below this
_FUSION_THRESHOLD = 3.0    # merge all motif contexts scoring at or above this
_VERIFIER_CONTEXT_CAP = 2000  # chars sent to verifier — enough to judge, not to waste tokens
_CHUNK_LIMIT = 50


class GRASPRAG:
    """
    Query-Adaptive Graph RAG.

    Retrieval motifs:
      single_entity  →  1-hop ego-network around anchor entity
      bridge         →  bidirectional shortest-path between two anchors
      multi_entity   →  Steiner-tree approximation across all anchors
      broad          →  Personalized PageRank / LightRAG hybrid fallback

    All motifs produce a structured context that is then passed to the same
    prose-generation prompt, comparable to LightRAG/PathRAG's approach.

    cascade=False disables the motif cascade on uncertain/broad queries and
    falls straight back to LightRAG hybrid — used for ablation experiments.

    force_motif pins every query to one motif regardless of classifier output —
    used by the Fixed-motif ablations that test whether adaptive routing beats
    applying a single fixed shape to every query type. Bypasses cascade/fallback
    entirely: a thin or empty context from a mismatched motif is the intended
    signal, not something to route around.
    """

    def __init__(self, working_dir: str, cascade: bool = True,
                 flat_context: bool = False,
                 oracle_labels: dict | None = None,
                 force_motif: str | None = None):
        self._working_dir = working_dir
        self._cascade = cascade
        self._flat_context = flat_context
        self._oracle_labels = oracle_labels  # {question: query_type} — skips classifier when set
        self._force_motif = force_motif      # one of single_entity/bridge/multi_entity/broad
        self._rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=generation_complete,
            embedding_func=openai_embed,
        )
        self._classifier: QueryClassifier | None = None
        self._retriever: MotifRetriever | None = None
        self._last_classification: dict | None = None

    async def initialize(self):
        await self._rag.initialize_storages()
        self._entity_chunks, self._relation_chunks, self._text_chunks = \
            self._load_chunk_stores()

    def _load_chunk_stores(self) -> tuple[dict, dict, dict]:
        wd = self._working_dir
        def _load(name):
            p = os.path.join(wd, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            return {}
        return (
            _load("kv_store_entity_chunks.json"),
            _load("kv_store_relation_chunks.json"),
            _load("kv_store_text_chunks.json"),
        )

    async def ainsert(self, text: str | list[str]):
        await self._rag.ainsert(text)
        self._retriever = None

    async def _fetch_candidates(self, query: str, top_k: int = 20) -> list[str]:
        # Query entity and relationship VDBs in parallel (Idea 6).
        entity_task = self._rag.entities_vdb.query(query, top_k=top_k)
        rel_task = self._rag.relationships_vdb.query(query, top_k=top_k)
        entity_results, rel_results = await asyncio.gather(entity_task, rel_task, return_exceptions=True)

        names: list[str] = []
        seen: set[str] = set()

        if not isinstance(entity_results, Exception):
            for r in entity_results:
                n = r.get("entity_name")
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)

        # Relationship endpoints surface entity names that entity VDB may miss.
        if not isinstance(rel_results, Exception):
            for r in rel_results:
                for field in ("src_id", "tgt_id"):
                    n = r.get(field)
                    if n and n not in seen:
                        seen.add(n)
                        names.append(n)

        return names

    async def _resolve_anchors(self, anchors: list[str]) -> list[str]:
        """Map classifier anchor names to real graph nodes, trying normalization if exact match fails (Idea 5)."""
        graph = self._rag.chunk_entity_relation_graph
        resolved: list[str] = []
        for anchor in anchors:
            if await graph.get_node(anchor):
                resolved.append(anchor)
                continue
            variants = [
                anchor.rstrip("s"),
                anchor.rstrip("es"),
                anchor + "s",
                anchor.lower(),
                anchor.title(),
                anchor.upper(),
            ]
            found = None
            for v in variants:
                if v and v != anchor and await graph.get_node(v):
                    found = v
                    break
            if found:
                resolved.append(found)
            else:
                try:
                    hits = await self._rag.entities_vdb.query(anchor, top_k=1)
                    if hits and hits[0].get("entity_name"):
                        resolved.append(hits[0]["entity_name"])
                except Exception:
                    pass
        return resolved

    async def aquery(self, query: str) -> str:
        self._ensure_retriever()

        candidates = await self._fetch_candidates(query)

        if self._force_motif is not None:
            # Fixed-motif ablation: pin the type, skip classifier/oracle, skip
            # cascade and hybrid fallback entirely — always answer from exactly
            # what this one motif retrieves, even if that's thin or empty.
            query_type = self._force_motif
            n_anchors = {"single_entity": 1, "bridge": 2, "multi_entity": 3, "broad": 5}.get(query_type, 0)
            anchors = await self._resolve_anchors(candidates[:n_anchors])
            self._last_classification = {"query_type": query_type, "anchors": anchors}
            paths = self._retriever.retrieve_forced(query_type, anchors)
            _build = self._build_flat_context if self._flat_context else self._build_structured_context
            context = await _build(paths, query_type, query)
            prompt = _RESPONSE_PROMPT.format(context=context, query=query)
            return await self._rag.llm_model_func(prompt)

        if self._oracle_labels is not None:
            # Oracle mode: use ground-truth type, assign VDB candidates as anchors.
            query_type = self._oracle_labels.get(query, "broad")
            n_anchors  = {"single_entity": 1, "bridge": 2, "multi_entity": 3}.get(query_type, 0)
            anchors    = await self._resolve_anchors(candidates[:n_anchors])
        else:
            query_type, anchors = await self._classifier.classify(query, candidates=candidates)
            anchors = await self._resolve_anchors(anchors)

        self._last_classification = {"query_type": query_type, "anchors": anchors}
        paths = self._retriever.retrieve_direct(query_type, anchors)
        confidence = self._retriever.uncertainty(paths)

        # Fast path: classifier is confident and motif is non-broad → use it directly.
        if query_type != "broad" and confidence >= _UNCERTAINTY_THRESHOLD and paths:
            _build = self._build_flat_context if self._flat_context else self._build_structured_context
            context = await _build(paths, query_type, query)
            if context.strip():
                prompt = _RESPONSE_PROMPT.format(context=context, query=query)
                return await self._rag.llm_model_func(prompt)

        # Uncertain or broad: cascade (if enabled) or straight hybrid fallback.
        if self._cascade:
            return await self._motif_cascade(query, candidates)
        return await self._rag.aquery(query, param=QueryParam(mode="hybrid"))

    async def _score_context(self, query: str, context: str) -> float:
        """Ask the LLM to score how well a context supports answering the query (1–5)."""
        if not context.strip():
            return 0.0
        excerpt = context[:_VERIFIER_CONTEXT_CAP]
        prompt = _VERIFIER_PROMPT.format(query=query, context=excerpt)
        try:
            result = await self._rag.llm_model_func(prompt)
            m = re.search(r"\b([1-5])\b", result)
            if m:
                return float(m.group(1))
        except Exception:
            pass
        # Fallback heuristic: longer context is weakly better than empty
        return min(2.0, len(context) / 500)

    async def _motif_cascade(self, query: str, candidates: list[str]) -> str:
        """
        Motif Cascade: run single_entity / bridge / multi_entity in parallel,
        score each context with the verifier, return an answer from the winner.

        Anchors are assigned from top VDB candidates — no extra LLM call needed.
        Falls back to LightRAG hybrid only if all contexts score below the threshold.
        """
        # Assign top VDB candidates to each motif by how many anchors they need.
        motif_anchors: dict[str, list[str]] = {}
        if len(candidates) >= 1:
            motif_anchors["single_entity"] = candidates[:1]
        if len(candidates) >= 2:
            motif_anchors["bridge"] = candidates[:2]
        if len(candidates) >= 3:
            motif_anchors["multi_entity"] = candidates[:3]

        # Retrieve paths for each motif (sync graph traversal — fast).
        motif_paths = {
            mtype: self._retriever.retrieve_direct(mtype, anchors)
            for mtype, anchors in motif_anchors.items()
        }
        # Keep only motifs that actually found paths.
        motif_paths = {m: p for m, p in motif_paths.items() if p}

        if not motif_paths:
            return await self._rag.aquery(query, param=QueryParam(mode="hybrid"))

        # Build all contexts in parallel.
        motif_list = list(motif_paths.keys())
        _build = self._build_flat_context if self._flat_context else self._build_structured_context
        contexts = await asyncio.gather(*[
            _build(motif_paths[m], m, query)
            for m in motif_list
        ])
        motif_contexts = {
            m: ctx for m, ctx in zip(motif_list, contexts) if ctx.strip()
        }

        if not motif_contexts:
            return await self._rag.aquery(query, param=QueryParam(mode="hybrid"))

        # Score all contexts in parallel.
        scored_motifs = list(motif_contexts.keys())
        scores = await asyncio.gather(*[
            self._score_context(query, motif_contexts[m])
            for m in scored_motifs
        ])
        motif_scores = dict(zip(scored_motifs, scores))

        best_motif = max(motif_scores, key=motif_scores.__getitem__)
        best_score = motif_scores[best_motif]

        print(f"  [cascade] {motif_scores} → {best_motif} ({best_score:.1f})")

        if best_score < _CASCADE_MIN_SCORE:
            return await self._rag.aquery(query, param=QueryParam(mode="hybrid"))

        # Late fusion: merge every motif context that scores at or above the
        # threshold (Idea 4). If only one qualifies, skip the header overhead.
        good = sorted(
            [(m, motif_contexts[m]) for m in scored_motifs if motif_scores[m] >= _FUSION_THRESHOLD],
            key=lambda x: -motif_scores[x[0]],
        )
        if len(good) > 1:
            merged = "\n\n".join(f"[{m.upper()} CONTEXT]\n{ctx}" for m, ctx in good)
            context = merged
        else:
            context = motif_contexts[best_motif]

        prompt = _RESPONSE_PROMPT.format(context=context, query=query)
        return await self._rag.llm_model_func(prompt)

    # ── internals ────────────────────────────────────────────────────────────

    def _ensure_retriever(self):
        if self._retriever is not None:
            return
        G = self._rag.chunk_entity_relation_graph._graph
        self._retriever = MotifRetriever(G, self._rag.entities_vdb)
        self._classifier = QueryClassifier(self._rag.llm_model_func)

    # ── chunk helpers ─────────────────────────────────────────────────────────

    def _node_chunks(self, node: str, seen: set) -> list[str]:
        texts = []
        for cid in self._entity_chunks.get(node, {}).get("chunk_ids", []):
            if cid not in seen and cid in self._text_chunks:
                seen.add(cid)
                texts.append(self._text_chunks[cid].get("content", ""))
        return texts

    def _edge_chunks(self, src: str, tgt: str, seen: set) -> list[str]:
        texts = []
        for key in (f"{src}<SEP>{tgt}", f"{tgt}<SEP>{src}"):
            for cid in self._relation_chunks.get(key, {}).get("chunk_ids", []):
                if cid not in seen and cid in self._text_chunks:
                    seen.add(cid)
                    texts.append(self._text_chunks[cid].get("content", ""))
        return texts

    async def _secondary_chunks(self, query: str, seen: set, limit: int) -> list[str]:
        """Fill remaining budget via vector search over the full chunk corpus."""
        texts = []
        try:
            results = await self._rag.chunks_vdb.query(query, top_k=limit * 2)
            for r in results:
                cid = r.get("id", "")
                if cid not in seen and cid in self._text_chunks:
                    seen.add(cid)
                    texts.append(self._text_chunks[cid].get("content", ""))
                    if len(texts) >= limit:
                        break
        except Exception:
            pass
        return texts

    def _fmt_chunks(self, chunks: list[str]) -> str:
        return "\n\n".join(f"[Passage]\n{c}" for c in chunks if c.strip())

    # ── bridge path helpers ───────────────────────────────────────────────────

    async def _path_edge_keywords(self, path: list[str]) -> str:
        """Return edge keywords along a path joined as a single string."""
        graph = self._rag.chunk_entity_relation_graph
        parts = []
        for i in range(len(path) - 1):
            ed = await graph.get_edge(path[i], path[i + 1]) or \
                 await graph.get_edge(path[i + 1], path[i])
            if ed:
                kw = ed.get("keywords", "").strip()
                if kw:
                    parts.append(kw)
        return ", ".join(parts)

    async def _rank_paths_by_edge_relevance(
        self, paths: list[list[str]], query: str
    ) -> list[list[str]]:
        """
        Idea 3 — rank paths by how relevant their edge keywords are to the query.

        Score = average normalised token overlap between query words and the
        keyword string on each edge, with a small bonus for shorter paths.
        No extra embedding or LLM calls — only fast graph dict lookups.
        """
        query_tokens = set(query.lower().split())

        async def _score(path: list[str]) -> float:
            n_edges = len(path) - 1
            if n_edges == 0:
                return 0.0
            total = 0.0
            graph = self._rag.chunk_entity_relation_graph
            for i in range(n_edges):
                ed = await graph.get_edge(path[i], path[i + 1]) or \
                     await graph.get_edge(path[i + 1], path[i])
                kw = ed.get("keywords", "") if ed else ""
                kw_tokens = set(kw.lower().replace(",", " ").split())
                if kw_tokens:
                    total += len(query_tokens & kw_tokens) / len(kw_tokens)
            avg = total / n_edges
            length_bonus = 0.05 / len(path)   # prefer shorter paths on ties
            return avg + length_bonus

        scored = [(await _score(p), p) for p in paths]
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    # ── structured context builders ───────────────────────────────────────────

    async def _build_structured_context(
        self, paths: list[list[str]], query_type: str, query: str
    ) -> str:
        if query_type == "bridge":
            return await self._bridge_context(paths, query)
        if query_type == "single_entity":
            return await self._single_entity_context(paths, query)
        if query_type == "multi_entity":
            return await self._multi_entity_context(paths, query)
        if query_type == "broad":
            return await self._broad_context(paths, query)
        return ""

    async def _build_flat_context(
        self, paths: list[list[str]], query_type: str, query: str
    ) -> str:
        """Ablation variant: same subgraph as _build_structured_context, serialized as
        flat triples with no topology-aware step labels or reasoning-chain structure."""
        graph = self._rag.chunk_entity_relation_graph
        seen: set[str] = set()
        seen_nodes: set[str] = set()
        seen_edges: set[tuple] = set()

        node_lines: list[str] = []
        edge_lines: list[str] = []

        for path in paths:
            for i, node in enumerate(path):
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    nd = await graph.get_node(node)
                    desc = nd.get("description", "") if nd else ""
                    etype = nd.get("entity_type", "entity") if nd else "entity"
                    node_lines.append(f"Entity: {node} ({etype})\nDescription: {desc}")
                if i < len(path) - 1:
                    nxt = path[i + 1]
                    key = tuple(sorted([node, nxt]))
                    if key not in seen_edges:
                        seen_edges.add(key)
                        ed = (await graph.get_edge(node, nxt) or
                              await graph.get_edge(nxt, node))
                        keywords = ed.get("keywords", "") if ed else ""
                        edge_lines.append(
                            f"Relation: {node} → {nxt} | Keywords: {keywords}"
                        )

        all_chunks: list[str] = []
        for node in seen_nodes:
            all_chunks += self._node_chunks(node, seen)
        for path in paths:
            for i in range(len(path) - 1):
                all_chunks += self._edge_chunks(path[i], path[i + 1], seen)

        remaining = _CHUNK_LIMIT - len(seen)
        if remaining > 0:
            all_chunks += await self._secondary_chunks(query, seen, remaining)

        parts: list[str] = []
        if node_lines:
            parts.append("Entities:\n" + "\n\n".join(node_lines))
        if edge_lines:
            parts.append("Relations:\n" + "\n".join(edge_lines))
        if all_chunks:
            parts.append(f"Source Passages:\n{self._fmt_chunks(all_chunks[:20])}")
        return "\n\n".join(parts)

    async def _bridge_context(self, paths: list[list[str]], query: str) -> str:
        """
        Format the best path as an explicit hop-by-hop reasoning chain.

        Idea 3 — path selected by edge-keyword relevance to the query, not
        just shortest length.  Idea 1 — remaining chunk budget filled via a
        graph-guided re-query (edge keyword chain + original query) so the
        vector search looks for evidence of the specific connection, not just
        the query in general.

          [Step 1 — Entity: A]  description + passages
          [Step 2 — A → B via: keywords]  passages
          [Step 3 — Entity: B]  description + passages
          ...
        """
        if not paths:
            # No connecting path at all (e.g. Fixed-Bridge forced on a question with
            # <2 resolvable anchors) — genuinely empty context, not an error case.
            return ""

        graph = self._rag.chunk_entity_relation_graph
        seen: set[str] = set()

        # Idea 3: pick the path whose edge keywords best match the query,
        # not simply the shortest path.
        ranked = await self._rank_paths_by_edge_relevance(paths, query)
        best = ranked[0]

        sections: list[str] = []
        step = 1

        for i, node in enumerate(best):
            nd = await graph.get_node(node)
            desc = nd.get("description", "") if nd else ""
            etype = nd.get("entity_type", "entity") if nd else "entity"

            node_chunks = self._node_chunks(node, seen)
            section = (
                f"[Step {step} — Entity: {node} ({etype})]\n"
                f"Description: {desc}\n"
            )
            if node_chunks:
                section += f"Source Passages:\n{self._fmt_chunks(node_chunks[:3])}"
            sections.append(section)
            step += 1

            if i < len(best) - 1:
                nxt = best[i + 1]
                ed = await graph.get_edge(node, nxt) or await graph.get_edge(nxt, node)
                keywords = ed.get("keywords", "") if ed else ""
                edge_chunks = self._edge_chunks(node, nxt, seen)
                section = (
                    f"[Step {step} — Connection: {node} → {nxt}]\n"
                    f"Relationship keywords: {keywords}\n"
                )
                if edge_chunks:
                    section += f"Source Passages:\n{self._fmt_chunks(edge_chunks[:2])}"
                sections.append(section)
                step += 1

        # Supporting paths — process in relevance order (ranked[1:])
        support_chunks: list[str] = []
        for path in ranked[1:]:
            for i, node in enumerate(path):
                support_chunks += self._node_chunks(node, seen)
                if i < len(path) - 1:
                    support_chunks += self._edge_chunks(node, path[i + 1], seen)

        # Idea 1: graph-guided re-query.
        # Use the best path's edge keyword chain as the vector search query so
        # we find corpus passages that discuss the specific connection, not just
        # passages that mention the query words.  Combine with the original
        # query so neither signal is lost.
        edge_kw_chain = await self._path_edge_keywords(best)
        requery = f"{edge_kw_chain}, {query}" if edge_kw_chain else query

        remaining = _CHUNK_LIMIT - len(seen)
        if remaining > 0:
            support_chunks += await self._secondary_chunks(requery, seen, remaining)

        chain = "\n\n".join(sections)
        if support_chunks:
            chain += f"\n\n[Supporting Context]\n{self._fmt_chunks(support_chunks[:10])}"
        return chain

    async def _single_entity_context(self, paths: list[list[str]], query: str) -> str:
        """
        Entity profile: description + direct connections + source passages.

          [Entity Profile: X]
          Type / Description
          Connected to: A via keywords, B via keywords, ...
          Source Passages: ...
        """
        graph = self._rag.chunk_entity_relation_graph
        seen: set[str] = set()

        # Anchor is the first node of the first path
        anchor = paths[0][0] if paths else None
        if not anchor:
            return ""

        nd = await graph.get_node(anchor)
        desc = nd.get("description", "") if nd else ""
        etype = nd.get("entity_type", "entity") if nd else "entity"

        # Collect direct connections
        connections: list[str] = []
        for path in paths:
            if len(path) >= 2:
                neighbor = path[1]
                ed = await graph.get_edge(anchor, neighbor) or \
                     await graph.get_edge(neighbor, anchor)
                keywords = ed.get("keywords", "") if ed else ""
                connections.append(f"  - {neighbor}: {keywords}")

        profile = (
            f"[Entity Profile: {anchor}]\n"
            f"Type: {etype}\n"
            f"Description: {desc}\n"
        )
        if connections:
            profile += "Connected to:\n" + "\n".join(connections[:10]) + "\n"

        chunks = self._node_chunks(anchor, seen)
        for path in paths:
            for i, node in enumerate(path[1:], 1):
                chunks += self._node_chunks(node, seen)
                chunks += self._edge_chunks(path[i - 1], node, seen)

        remaining = _CHUNK_LIMIT - len(seen)
        if remaining > 0:
            chunks += await self._secondary_chunks(query, seen, remaining)

        if chunks:
            profile += f"\nSource Passages:\n{self._fmt_chunks(chunks[:20])}"
        return profile

    async def _multi_entity_context(self, paths: list[list[str]], query: str) -> str:
        """
        Per-entity profiles + connection map between them.

          [Entity 1: X]  description + passages
          [Entity 2: Y]  description + passages
          [Connections]  X ↔ Y via keywords
          [Source Passages for connections]
        """
        graph = self._rag.chunk_entity_relation_graph
        seen: set[str] = set()

        # Collect unique anchor nodes (endpoints of Steiner edges)
        anchor_set: dict[str, dict] = {}
        for path in paths:
            for node in path:
                if node not in anchor_set:
                    nd = await graph.get_node(node)
                    anchor_set[node] = nd or {}

        sections: list[str] = []
        for idx, (node, nd) in enumerate(anchor_set.items(), 1):
            desc = nd.get("description", "")
            etype = nd.get("entity_type", "entity")
            chunks = self._node_chunks(node, seen)
            section = (
                f"[Entity {idx}: {node} ({etype})]\n"
                f"Description: {desc}\n"
            )
            if chunks:
                section += f"Source Passages:\n{self._fmt_chunks(chunks[:3])}"
            sections.append(section)

        # Connection map
        conn_lines: list[str] = []
        conn_chunks: list[str] = []
        seen_edges: set[tuple] = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = tuple(sorted([u, v]))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                ed = await graph.get_edge(u, v) or await graph.get_edge(v, u)
                keywords = ed.get("keywords", "") if ed else ""
                conn_lines.append(f"  - {u} ↔ {v}: {keywords}")
                conn_chunks += self._edge_chunks(u, v, seen)

        if conn_lines:
            sections.append(
                "[Connections between entities]\n" + "\n".join(conn_lines)
            )

        remaining = _CHUNK_LIMIT - len(seen)
        if remaining > 0:
            conn_chunks += await self._secondary_chunks(query, seen, remaining)

        context = "\n\n".join(sections)
        if conn_chunks:
            context += f"\n\n[Connection Source Passages]\n{self._fmt_chunks(conn_chunks[:15])}"
        return context

    async def _broad_context(self, paths: list[list[str]], query: str) -> str:
        """
        Broad/PPR motif: profile the PageRank-diffused entities around the anchor
        set plus their connections — a topic-level snapshot rather than a single
        reasoning chain, for queries too diffuse for one specific motif.

          [Related Entity 1: X]  description + passages
          [Related Entity 2: Y]  description + passages
          [Connections]  X ↔ Y via keywords
          [Source Passages for connections]
        """
        graph = self._rag.chunk_entity_relation_graph
        seen: set[str] = set()

        node_set: dict[str, dict] = {}
        for path in paths:
            for node in path:
                if node not in node_set:
                    nd = await graph.get_node(node)
                    node_set[node] = nd or {}

        sections: list[str] = []
        for idx, (node, nd) in enumerate(node_set.items(), 1):
            desc = nd.get("description", "")
            etype = nd.get("entity_type", "entity")
            chunks = self._node_chunks(node, seen)
            section = (
                f"[Related Entity {idx}: {node} ({etype})]\n"
                f"Description: {desc}\n"
            )
            if chunks:
                section += f"Source Passages:\n{self._fmt_chunks(chunks[:3])}"
            sections.append(section)

        conn_lines: list[str] = []
        conn_chunks: list[str] = []
        seen_edges: set[tuple] = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = tuple(sorted([u, v]))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                ed = await graph.get_edge(u, v) or await graph.get_edge(v, u)
                keywords = ed.get("keywords", "") if ed else ""
                conn_lines.append(f"  - {u} ↔ {v}: {keywords}")
                conn_chunks += self._edge_chunks(u, v, seen)

        if conn_lines:
            sections.append("[Connections]\n" + "\n".join(conn_lines))

        remaining = _CHUNK_LIMIT - len(seen)
        if remaining > 0:
            conn_chunks += await self._secondary_chunks(query, seen, remaining)

        context = "\n\n".join(sections)
        if conn_chunks:
            context += f"\n\n[Connection Source Passages]\n{self._fmt_chunks(conn_chunks[:15])}"
        return context

    async def finalize(self):
        await self._rag.finalize_storages()
