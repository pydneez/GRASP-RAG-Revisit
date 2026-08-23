"""
Motif-based subgraph retriever for GRASP-RAG.

Query shape → retrieval motif:
  single_entity  →  1-hop ego-network around anchor
  bridge         →  shortest / simple paths between two anchors
  multi_entity   →  Steiner-tree approximation connecting all anchors
  broad          →  Personalized PageRank from anchors
"""

from difflib import get_close_matches

import networkx as nx
from networkx.algorithms.approximation import steiner_tree


class MotifRetriever:
    def __init__(self, graph: nx.Graph, entities_vdb):
        self.G = graph            # LightRAG gives us nx.Graph (undirected)
        self.vdb = entities_vdb
        self._nodes = list(graph.nodes())

    async def link_anchors(self, anchors: list[str]) -> list[str]:
        """Map extracted anchor strings to graph node names via vector search + fuzzy fallback."""
        linked, seen = [], set()
        for anchor in anchors:
            # 1. Vector search — most accurate for domain-specific names
            results = await self.vdb.query(anchor, top_k=3)
            matched = False
            for r in results:
                name = r.get("entity_name", "")
                if name and name in self.G and name not in seen:
                    linked.append(name)
                    seen.add(name)
                    matched = True
                    break
            if matched:
                continue
            # 2. Fuzzy string match as fallback
            hits = get_close_matches(anchor, self._nodes, n=1, cutoff=0.55)
            if hits and hits[0] not in seen:
                linked.append(hits[0])
                seen.add(hits[0])
        return linked

    async def retrieve(self, query_type: str, anchors: list[str]) -> list[list[str]]:
        nodes = await self.link_anchors(anchors)
        if not nodes:
            query_type = "broad"
        return self._dispatch(query_type, nodes)

    def retrieve_direct(self, query_type: str, anchors: list[str]) -> list[list[str]]:
        """Like retrieve() but anchors are already exact graph node names — no linking needed."""
        nodes = [a for a in anchors if a in self.G]
        if not nodes:
            query_type = "broad"
        return self._dispatch(query_type, nodes)

    def retrieve_forced(self, query_type: str, anchors: list[str]) -> list[list[str]]:
        """
        Like retrieve_direct(), but never reroutes to a different motif when anchors
        are insufficient for the requested one — used by Fixed-motif ablations, where
        a genuinely thin/degenerate result from a mismatched motif is the signal we
        want to measure, not something to paper over by silently falling back to PPR.
        """
        nodes = [a for a in anchors if a in self.G]
        if query_type == "single_entity":
            return self._ego(nodes[0]) if nodes else []
        if query_type == "bridge":
            if len(nodes) >= 2:
                return self._bridge(nodes[0], nodes[1])
            return self._ego(nodes[0], limit=7) if nodes else []
        if query_type == "multi_entity":
            return self._steiner(nodes)  # already degrades gracefully for <2 nodes
        return self._ppr(nodes)          # broad — already handles zero anchors

    def _dispatch(self, query_type: str, nodes: list[str]) -> list[list[str]]:
        if query_type == "single_entity" and nodes:
            return self._ego(nodes[0])
        if query_type == "bridge" and len(nodes) >= 2:
            return self._bridge(nodes[0], nodes[1])
        if query_type == "multi_entity" and len(nodes) >= 2:
            return self._steiner(nodes)
        return self._ppr(nodes)

    # ── motif implementations ────────────────────────────────────────────────

    def _ego(self, anchor: str, limit: int = 15) -> list[list[str]]:
        if anchor not in self.G:
            return []
        return [[anchor, nb] for nb in self.G.neighbors(anchor)][:limit]

    def _bridge(self, src: str, tgt: str, limit: int = 15) -> list[list[str]]:
        paths = []
        try:
            paths.append(nx.shortest_path(self.G, src, tgt))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        try:
            for p in nx.all_simple_paths(self.G, src, tgt, cutoff=3):
                if p not in paths:
                    paths.append(p)
                if len(paths) >= limit:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        if not paths:
            # No connecting path — return ego-networks from both ends
            paths = self._ego(src, 7) + self._ego(tgt, 7)
        return paths[:limit]

    def _steiner(self, nodes: list[str], limit: int = 15) -> list[list[str]]:
        valid = [n for n in nodes if n in self.G]
        if len(valid) < 2:
            return self._ego(valid[0]) if valid else []
        try:
            st = steiner_tree(self.G, valid)
            return [[u, v] for u, v in st.edges()][:limit]
        except Exception:
            # Fall back to pairwise shortest paths
            paths = []
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    try:
                        paths.append(nx.shortest_path(self.G, valid[i], valid[j]))
                    except nx.NetworkXNoPath:
                        pass
            return paths[:limit]

    def _ppr(self, nodes: list[str], top_n: int = 20, limit: int = 15) -> list[list[str]]:
        valid = [n for n in nodes if n in self.G]
        if not valid:
            # No anchors at all — use top-degree nodes as a proxy
            top = sorted(self.G.nodes(), key=lambda n: self.G.degree(n), reverse=True)[:5]
            return [[n, nb] for n in top for nb in list(self.G.neighbors(n))[:3]][:limit]

        pers = {n: 1.0 / len(valid) for n in valid}
        try:
            ppr = nx.pagerank(self.G, personalization=pers, alpha=0.85)
        except Exception:
            ppr = {n: 1.0 for n in self.G.nodes()}

        hot = set(sorted(ppr, key=ppr.get, reverse=True)[:top_n])
        seen, paths = set(), []
        for anchor in valid:
            for nb in self.G.neighbors(anchor):
                if nb in hot:
                    key = tuple(sorted([anchor, nb]))
                    if key not in seen:
                        seen.add(key)
                        paths.append([anchor, nb])
        return paths[:limit]

    # ── confidence signal ────────────────────────────────────────────────────

    def uncertainty(self, paths: list[list[str]]) -> float:
        """Confidence in [0, 1]. Low value → fall back to PathRAG-style retrieval."""
        if not paths:
            return 0.0
        nodes = {n for path in paths for n in path}
        edges = sum(len(p) - 1 for p in paths)
        n = len(nodes)
        if n <= 1:
            return 1.0
        density = edges / (n * (n - 1) / 2)
        size_factor = min(1.0, 8.0 / max(n, 1))
        return density * size_factor
