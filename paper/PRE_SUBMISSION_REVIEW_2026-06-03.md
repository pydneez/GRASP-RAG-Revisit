# Pre-Submission Referee Report

**Paper**: GRASP-RAG: Graph Retrieval with Adaptive Structured Paths and Path-Guided Reasoning
**Authors**: Nonprawich Intakaew, Lapitsala Chimpad, Pitinun Wiphumitsitsakul, Somchart Fugkeaw
**Date**: 2026-06-03
**Review Standard**: APSIPA 2026

---

## Overall Assessment

GRASP-RAG proposes routing queries to one of four graph traversal motifs—ego-network, shortest-path, Steiner-tree approximation, and LightRAG hybrid—based on a query-type classifier, plus a topology-mirroring context serialization step (PGR). The contribution is rated **Incremental**: each of the four motifs is drawn from a named prior work, and the novelty lies in the routing layer and context formatting. The most critical issue is that the paper is 7 pages against APSIPA's 5–6 page limit, and the PGR contribution is explicitly acknowledged in the paper's own Limitations section as unsubstantiated (the ablation isolating PGR from the motif's retrieval guarantee is listed as future work—yet PGR is presented as a named contribution throughout).

**Preliminary Recommendation**: Revise before sending

---

## 1. Contribution & Referee Assessment

## Agent 6: Contribution Evaluation (APSIPA 2026)

### Part 1 — Central Contribution

**One-sentence claim:** GRASP-RAG proposes that classifying queries into four logical shapes (single-entity, bridge, multi-entity, broad) and routing each to a structurally matched graph traversal algorithm—plus a topology-mirroring context serialization called Path-Guided Reasoning—produces better RAG answers than fixed-strategy graph retrieval systems.

**Is query-adaptive motif selection genuinely new?**
It is incremental rather than genuinely new. Each of the four motifs is directly borrowed from a named prior work: ego-network from LightRAG local mode, shortest-path from PathRAG, Steiner tree from G-Retriever, and PPR diffusion from HippoRAG. The paper is transparent about this in Table 1. The contribution is the routing layer—an LLM classifier that selects among these four existing algorithms at inference time—plus the PGR serialization step. The idea that query type should govern retrieval strategy is also explicit in LQR (Huang et al., 2024), which the paper correctly cites. The novelty is in the four-way taxonomy applied to a graph RAG setting and the KG-grounded anchor-linking mechanism. This is a meaningful engineering combination but not a conceptual breakthrough.

**Closest prior work:** PathRAG is the closest in the graph RAG space. GRASP-RAG adds: (1) the routing classifier so that PathRAG-style path traversal is reserved for bridge queries, and (2) the PGR context structuring. LQR is the closest in spirit conceptually.

**Rating: Incremental**

The combination of known graph algorithms under a learned routing layer is a legitimate engineering contribution, but no individual component is new. The paper's main conceptual claim is shared with LQR in spirit and with individual prior graph RAG papers in implementation. The cascade mechanism failing and the broad motif simply delegating to LightRAG hybrid further narrows what is genuinely new.

### Part 2 — Experimental Validity and Credibility

**Evaluation methodology:** LLM-as-judge with three judges (GPT-4o-mini, Claude Haiku, Gemini Flash), forward/reverse ordering for position bias, majority vote across five dimensions, bias correction requiring both orderings to agree. Experiment 2 adds reference-anchored evaluation using UltraDomain expert answers. LLM-as-judge is now broadly accepted but three non-trivial concerns apply:

- GPT-4o-mini is both the answer-generation backbone and one of three judges, creating a potential self-preference confound that the paper does not address.
- 25 questions per type per domain: a 2-question swing changes win rate by 8 percentage points. No confidence intervals or significance tests are reported.
- Typed questions generated from the same corpus as the KG: questions are constructed to name entities that exist in the graph by design, making anchor linking and classifier accuracy artificially high.

**Cascade failure:** The cascade wins only 16–36% against no-cascade. The paper does not analyze why: is it the verifier's calibration, the confidence threshold, or the late fusion design? The paper says "warrants re-examination" but a failed component that appears in Figure 1 requires deeper analysis.

**Broad motif delegation:** For 25% of test questions, GRASP-RAG selects LightRAG hybrid—meaning it is not routing to a new motif but to a competitor baseline. The 81.1% overall win rate against LightRAG is inflated by wins on broad queries where both systems are identical. This should be explicitly disaggregated.

**Legal domain:** Reporting results from a domain where the KG is known to be incomplete (20 of 199 questions returned no result) is not a legitimate experiment. The paper calls it "a lower bound" but that framing is insufficient.

### Part 3 — Analyses: Required and Suggested

**Required:**

1. **[CRITICAL] Statistical significance testing on win rates.** With 25 questions per cell, no confidence intervals or significance tests appear anywhere. A binomial test or bootstrap CI is necessary before any win-rate difference can be interpreted.

2. **[CRITICAL] PGR ablation: bridge motif + flat serialization vs. bridge motif + PGR step-by-step chain.** The paper explicitly admits it cannot separate motif retrieval from PGR context structuring. Without this ablation, the second named contribution is unsubstantiated. Feasibility: straightforward—requires only a modified context builder and re-evaluation.

3. **[CRITICAL] Address GPT-4o-mini self-preference confound.** Either exclude GPT-4o-mini from the judge panel and replace with a third independent model, or explicitly test whether GPT-4o-mini's verdicts differ systematically from Claude Haiku's and Gemini Flash's.

4. **[CRITICAL] Evaluation on a standard multi-hop QA benchmark (HotpotQA or 2WikiMultiHopQA).** The paper cites both but evaluates on neither. BridgeRAG achieves SOTA on these benchmarks and is cited in related work—direct comparison is expected.

5. **[CRITICAL] Complete Legal KG indexing and re-run before submission.** An experiment on a broken knowledge base cannot be presented as a publishable domain result.

**Suggested:**

1. **[MAJOR] Cascade mechanism diagnosis.** Report: the distribution of confidence scores, the fraction of queries triggering the cascade, and the verifier score distributions. This would clarify whether the failure is a threshold calibration problem or a fundamental design problem.

2. **[MAJOR] Classifier accuracy conditioned on win rate.** The paper proposes this as future work but the logged per-question classification labels already exist. A win-rate heat map conditioned on correct vs. incorrect classification would quantify how much routing accuracy drives the result.

3. **[MAJOR] Latency/cost analysis.** GRASP-RAG adds an LLM classification call, VDB search, and (in cascade mode) three parallel motif executions plus an LLM verifier. No latency or cost figures are reported.

4. **[MAJOR] Win rates disaggregated to exclude broad queries.** Report what the aggregate win rates look like when broad queries are excluded, to isolate the genuine adaptive-routing contribution from the LightRAG-vs-LightRAG broad comparisons.

### Part 4 — Literature Positioning

Coverage is appropriate for a 5–7 page paper: LightRAG, PathRAG, HippoRAG, G-Retriever, GraphRAG, LQR, BridgeRAG, HotpotQA, 2WikiMultiHopQA, PageRank, Kou Steiner tree, UltraDomain, and the LLM-as-judge paper (Zheng et al., 2023). Notable missing papers: FLARE (Jiang et al., 2023, active retrieval), IRCoT (Trivedi et al., 2022, interleaved retrieval with CoT), ToG (Think-on-Graph, Sun et al., 2024, LLM-guided beam search over KGs). These are not fatal omissions for APSIPA but would strengthen credibility.

Positioning is largely accurate and fair. One inaccuracy: the conclusion states "Pending experiments include a reference-answer evaluation on the UltraDomain benchmark" while Section 4.5 already reports that experiment in full.

### Part 5 — APSIPA Fit and Recommendation

**Page count:** The paper is 7 pages; APSIPA 2026 regular papers are 5–6 pages. This is a hard logistical problem—the paper must be cut before submission regardless of content quality.

**Recommendation: Revise before sending.**

Required revisions to reach APSIPA acceptance: cut to 6 pages; add PGR ablation; complete Legal KG indexing and re-run; add statistical significance markers; address GPT-4o-mini self-preference; fix the Conclusion's internal inconsistency about Experiment 2 being "pending."

**Best alternative venues:** EACL 2027 findings track, SIGIR 2027 short paper, ACL ARR rolling. If PGR ablation and a standard benchmark evaluation are added, competitive for NAACL findings.

### Part 6 — Questions to the Authors

1. GPT-4o-mini is your answer-generation backbone and one of three LLM judges. Have you tested for self-preference bias? Does GPT-4o-mini's verdict distribution differ significantly from Claude Haiku's and Gemini Flash's when GRASP-RAG is compared against LightRAG—which also uses GPT-4o-mini? If both systems use the same backbone, is GPT-4o-mini's vote informative at all?

2. You claim Path-Guided Reasoning (PGR) is a core contribution, but the paper explicitly states you cannot separate PGR's contribution from the bridge motif's retrieval guarantee. Without the ablation comparing "bridge motif + flat triple serialization" against "bridge motif + PGR step-by-step chain," how do you justify PGR as a contribution rather than an implementation detail?

3. The broad motif delegates entirely to LightRAG hybrid mode. For 25% of your test questions, GRASP-RAG is LightRAG. What is the win rate against LightRAG if broad queries are excluded? And if excluded, what is the genuine scope of GRASP-RAG's adaptive-routing claim?

4. The cascade mechanism loses 16–36% of dimension-level decisions against no-cascade. What is the distribution of confidence scores across queries? What fraction of queries actually trigger the cascade? Is the LLM verifier's 1–5 scoring correlated with actual answer quality?

5. Your typed questions are generated from the same corpus as the KG. By construction, bridge questions name entities that exist in the graph. What is the anchor-linking success rate on typed questions vs. UltraDomain benchmark questions? If anchor linking is substantially easier on typed questions, Experiment 1 numbers are optimistic by design.

6. The Legal KG was built from an incomplete corpus, with 20 of 199 UltraDomain questions returning no result. You call Legal results "a lower bound," but an experiment where the knowledge base is incomplete is a broken experimental condition. Why was Legal included before full indexing was complete?

7. BridgeRAG achieves SOTA on HotpotQA, 2WikiMultiHopQA, and MuSiQue—exactly the benchmarks your bridge motif is designed for. You cite BridgeRAG in related work but do not evaluate on any of those benchmarks. What prevents a HotpotQA evaluation?

---

## 2. Unsupported Claims & Evaluation Integrity

## Agent 3: Unsupported Claims & Evaluation Integrity

### Causal Overclaiming

1. **[CRITICAL] Abstract** | "confirming that matching the retrieval algorithm to the query's logical shape produces structurally better answers than fixed-strategy retrieval" | "Confirming" implies a controlled experiment isolating this variable. The experiment uses LLM-as-judge pairwise preference on questions generated from the same corpus—this establishes a preference signal, not a confirmed causal improvement. PGR and motif selection are always confounded. | Fix: Replace "confirming" with "consistent with the hypothesis that matching…"

2. **[CRITICAL] Discussion §"Why adaptive motif selection is a structural guarantee"** | "This explains the 93.4% average bridge win rate against LightRAG: LightRAG's 1-hop expansion frequently misses the structural intermediate entity, while GRASP-RAG's bidirectional shortest-path algorithm is guaranteed to traverse it when the VDB correctly returns both anchors as candidates." | "Explains" asserts causation. The paper has not run the ablation that would support this mechanism. | Fix: "This is a plausible structural explanation…A direct ablation isolating motif traversal from PGR context formatting would be needed to confirm it."

3. **[MAJOR] Discussion §"Legal domain"** | "The bridge edge is pruned as 'unreliable' precisely because it is a specific, low-frequency connection." | "Precisely because" is a causal claim about PathRAG's internal pruning behavior without direct evidence (e.g., inspection of which edges PathRAG prunes). | Fix: "One plausible explanation is that…"

4. **[MAJOR] Discussion** | "PGR then presents the path as a labeled step-by-step chain, directing the LLM to name each intermediate node explicitly." | "Directing" implies PGR causes the LLM to name the bridge entity. The Limitations paragraph admits the PGR contribution has not been isolated. This contradicts itself. | Fix: "PGR provides a labeled chain that is intended to direct…"

5. **[MAJOR] Introduction** | "A system that ignores these linguistic cues and applies one fixed traversal to every query will inevitably retrieve mismatched context for most of them." | PathRAG achieves 40.8% GRASP-RAG win rate overall—competitive results that directly challenge "inevitably." | Fix: Replace "will inevitably" with "risks."

### Generalization & Scope Issues

1. **[MAJOR] Abstract** | "81.1% average win rate against LightRAG" presented without any hedge about sample size. At 25 questions per type per domain, no confidence intervals are reported. Point estimates to one decimal imply precision the sample cannot support. | Suggested caveat: Report Wilson-score confidence intervals or at minimum a footnote acknowledging high variance.

2. **[MAJOR] Experiments §4.3** | "GRASP-RAG consistently outperforms LightRAG and NaiveRAG across all four domains." Agriculture achieves only 49.4% vs. NaiveRAG—barely above chance. Win rates span 69.2%–94.5% (25-point range). "Consistently" is not justified. | Fix: Qualify with "on average" or acknowledge Agriculture results are substantially weaker.

3. **[MAJOR] §4.5 Legal domain** | Legal results included in aggregate averages without propagating the incomplete-KG caveat. Averaged results that include Legal numbers should be separately reported or excluded. | Fix: Report averages with and without Legal.

4. **[MINOR] Abstract** | "Evaluated on all four domains of UltraDomain" may mislead readers into thinking the original UltraDomain benchmark questions were used throughout. Experiment 1 uses LLM-generated typed questions; only Experiment 2 uses actual UltraDomain questions. | Fix: Clarify the distinction.

### Mechanism Claims Stated as Facts

1. **[MAJOR] Method §3.5 / Conclusion** | "GRASP-RAG's step-by-step chain context resolves this by making the intermediate entity structurally unavoidable." | "Structurally unavoidable" is an empirically untested behavioral claim. An LLM can read a labeled chain and still omit the intermediate node. The paper explicitly states the PGR ablation is future work. | Missing evidence: A PGR ablation showing bridge-motif + flat serialization produces lower bridge-entity naming rates.

2. **[MAJOR] Classifier Accuracy §4.4** | The section implies classifier accuracy causes retrieval improvement. The proposed conditional win-rate analysis is listed as future work, meaning the causal link between classification accuracy and retrieval performance is currently unsubstantiated.

3. **[MINOR] Discussion** | "In a small graph, VDB anchor recall is near-perfect and the bridge path is always found." The causal chain is asserted without measuring VDB anchor recall rates per domain. | Missing evidence: Per-domain anchor recall rates.

### Missing Caveats

1. **[MAJOR] Cascade Ablation §4.3** | The cascade fails but the paper does not discuss what this implies about the confidence score formula's calibration or the LLM verifier's reliability. The cascade failure also undermines the motivation for the complex verifier sub-component. | Suggested text: "The cascade failure suggests that either the confidence score triggers cascades at incorrect rates, or the LLM verifier does not reliably rank retrieved contexts by quality."

2. **[MAJOR] Broad Motif Delegation** | The paper acknowledges but does not fully draw the implication: GRASP-RAG contributes zero novel retrieval for broad queries (25% of all queries). The 81.1% aggregate includes a component where GRASP-RAG is literally running LightRAG against LightRAG. | Suggested text: "Excluding broad queries, the win rate against LightRAG would be [X]%."

3. **[MAJOR] Verbosity Bias** | The paper controls for position bias but not verbosity bias. GRASP-RAG's step-by-step PGR format may systematically produce longer answers that judges favor independent of accuracy. | Suggested text: "We do not control for verbosity bias; GRASP-RAG's structured PGR contexts may produce longer answers that judges favor independent of accuracy."

4. **[MAJOR] Question Generation Circularity** | The Limitations paragraph mentions cross-corpus evaluation as future work but does not acknowledge the deeper circularity: questions generated with graph-structural constraints are biased toward shapes where graph-traversal systems have structural advantages. | Suggested text: "Because typed questions are generated with explicit graph-structural constraints, the question set may be biased toward query shapes where graph RAG systems have structural advantages, potentially overstating GRASP-RAG's gains."

5. **[MINOR] GPT-4o-mini self-preference** | GPT-4o-mini generates all answers and serves as one of three judges. Potential self-preference bias from this overlap is not discussed.

### Priority Assertions (must verify)

1. **[MAJOR] Related Work** | The implicit novelty claim that no prior graph-RAG work uses a four-way logical-shape taxonomy tied to graph traversal algorithms needs author verification. Coverage of adaptive graph RAG appears selective.

2. **[MAJOR] Introduction** | "Every method applies one fixed retrieval strategy to all queries" — LightRAG has a dual-mode selector (local/global/hybrid), which is a form of adaptive retrieval. The paper's novelty claim is vulnerable to this challenge.

---

## 3. Internal Consistency & Cross-Reference Verification

## Agent 2: Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **[CRITICAL] Broad motif definition — Method §3.4 ↔ Experiments/Discussion/Conclusion** | The method section defines Motif 4 (broad) as "PPR diffusion" with LightRAG hybrid as a last-resort cascade fallback. The Experiments, Discussion, and Conclusion all state flatly that "the broad motif delegates to LightRAG hybrid mode." These descriptions are irreconcilable. | Fix: Update the method section to reflect that PPR is a cascade option, not the primary broad motif, and that LightRAG hybrid is the standard broad-query path.

2. **[CRITICAL] tab:winrate_bytype "Overall" row arithmetic mismatch** | Simple average of the four per-type LightRAG win rates: (93.4+93.4+63.7+69.6)/4 = 80.0, but the Overall row reports 81.1. Similarly, NaiveRAG average: 69.5 vs. reported 70.9. The Overall row appears copied from tab:winrate (domain-level averages) rather than computed from the by-type data. | Fix: Clarify whether the Overall row is a by-type average or a by-domain average, and ensure it is computed correctly.

3. **[CRITICAL] Conclusion says UltraDomain evaluation is "pending" but §4.5 reports it** | Conclusion lines: "Pending experiments include a reference-answer evaluation on the UltraDomain benchmark." Section 4.5 and Table 3 fully report this experiment. The conclusion was not updated. | Fix: Remove UltraDomain from the "pending" list.

### Major Inconsistencies

4. **[MAJOR] Discussion claims 97.6% (Legal, bridge vs LightRAG) and 100% (Mix, bridge vs LightRAG) — not in any table** | These domain-type specific figures appear in Discussion prose but no table reports per-domain-per-type breakdowns. The claims are unverifiable from the paper's own evidence. | Fix: Add a cross-domain-by-type table or remove specific claims.

5. **[MAJOR] Introduction omits PathRAG win rate cited in abstract** | Abstract reports 40.8% vs. PathRAG; Introduction preview omits it, creating a misleading impression given PathRAG is the "closest prior work." | Fix: Add the PathRAG number to the Introduction's results preview.

6. **[MAJOR] tab:winrate_bytype tabular column declaration mismatch** | `\begin{tabular}{lcccc}` declares 5 columns; table has 4 columns. This generates LaTeX warnings and potential formatting artifacts. | Fix: Change `{lcccc}` to `{lccc}`.

7. **[MAJOR] tab:ultradomain average rows inconsistently rounded** | vs LightRAG: (58.4+74.2+43.4+74.2)/4 = 62.55, reported as 62.5 (truncated). vs PathRAG: 59.55, reported as 59.5 (truncated). Two values are truncated rather than rounded. | Fix: Round to nearest tenth.

8. **[MAJOR] Notation: anchor set $\mathcal{A}$ vs $\mathcal{N}$ with no bridging definition** | §"Query Classification" outputs $(\tau, \mathcal{A})$; §"Adaptive Motif Selection" switches to $(\tau, \mathcal{N})$ without explanation. | Fix: Use one symbol throughout or explicitly define the transition.

9. **[MAJOR] fig:winrate_chart defined but never referenced in text** | `\label{fig:winrate_chart}` is defined; `\ref{fig:winrate_chart}` appears nowhere. The figure may float to an arbitrary position. | Fix: Add "Figure~\ref{fig:winrate_chart} visualizes the same breakdown as a bar chart."

### Terminology Drift

10. **[MINOR] "win rate" vs "win-rate"** | Both forms appear. Unhyphenated dominates in noun usage; hyphenated appears in attributive position (correct). Standardize: hyphenated as modifier, unhyphenated as noun.

11. **[MINOR] Gemini judge version: "Gemini Flash" vs "Gemini 2.0 Flash"** | Abstract/experiments say "Gemini Flash"; fig:system caption says "Gemini 2.0 Flash." These should match.

12. **[MINOR] "query shape" / "logical shape" / "query type" used interchangeably** | No single primary term adopted. Recommend: "query type" (most common) as the standard term; reserve "logical shape" for its first-definition context.

### Missing or Broken References

13. **[MAJOR] `zheng2023judging` defined in .bib but never cited** | The foundational LLM-as-judge paper (MT-Bench, Zheng et al. 2023) is in the bibliography but never cited in the Evaluation Protocol or Related Work, even though the paper adopts an LLM-as-judge protocol. | Fix: Cite in §4.2.

14. **[MINOR] `\label{eq:conf}` defined but never cross-referenced** | The confidence score equation has a label that is never `\ref`'d. | Fix: Add "(Equation~\ref{eq:conf})" in the cascade discussion, or remove the label.

---

## 4. Notation, Equations & Technical Accuracy

## Agent 4: Notation, Equations & Technical Accuracy

### Mathematical/Algorithmic Errors

1. **[CRITICAL] method.tex §"Adaptive Motif Selection" — $\mathcal{N}$ used simultaneously as anchor set and neighborhood function** | "$\mathcal{S} = \{(n_1, v) \mid v \in \mathcal{N}(n_1)\}$" introduces $\mathcal{N}(\cdot)$ as the neighborhood function in the same paragraph where $\mathcal{N}$ was introduced as the anchor-node variable. Same symbol, two meanings in adjacent sentences. | Fix: Use $\Gamma(v)$ or $\mathcal{N}_\mathcal{G}(v)$ for neighborhood; define at first use.

2. **[CRITICAL] method.tex — anchor variable switches from $\mathcal{A}$ to $\mathcal{N}$ without explanation** | §"Query Classification" outputs $(\tau, \mathcal{A})$; §"Adaptive Motif Selection" opens with "Based on $(\tau, \mathcal{N})$…" — the same concept, two symbols, no transition. | Fix: Use $\mathcal{A}$ throughout, or explicitly define $\mathcal{N}$ as the resolved version of $\mathcal{A}$.

3. **[CRITICAL] method.tex Eq. (1) — confidence formula numerator does not match code** | Paper: $|\mathcal{E}_\mathcal{S}|$ (distinct edge count). Code (`retriever.py uncertainty()`): `edges = sum(len(p)-1 for p in paths)` — total path-edge traversals, which overcounts shared edges. For paths that share edges, the code and the formula diverge. | Fix: Either change the code to count distinct edges, or update the formula's numerator to "total edge traversals across retrieved paths."

4. **[CRITICAL] Bridge motif mislabeled "Bidirectional shortest path"** | The motif table says "Bidirectional shortest path." The code calls `nx.shortest_path(self.G, src, tgt)` on an undirected graph—standard Dijkstra/BFS, not bidirectional search. The Discussion also says "bidirectional shortest-path algorithm." Three inconsistent phrasings exist across table, method prose, and discussion. | Fix: Standardize to "shortest-path traversal (undirected graph)" throughout; remove "bidirectional."

5. **[MAJOR] experiments.tex §"Classifier Accuracy" — "counts out of 100" needs clarification** | 25 questions/type × 4 domains = 100 per row. Raw counts equal percentages numerically by coincidence. | Fix: State "Cell values are counts (= 25 questions/type × 4 domains); values coincidentally equal percentages."

### Notation Inconsistencies

6. **[MAJOR] $\mathcal{A}$ vs. $\mathcal{N}$ for anchor nodes** (see items 2 and above) — appears in adjacent subsections.

7. **[MAJOR] $\mathcal{N}(n_1)$ (neighborhood) vs. $\mathcal{N}$ (anchor set)** — same symbol, different meanings in the same motif-description block.

8. **[MINOR] $n_1, n_2$ vs. anchor set $\mathcal{N}/\mathcal{A}$** — node variables introduced without explicit tie to the anchor set. Add "where $n_i \in \mathcal{A}$."

9. **[MINOR] $\mathcal{S} \subseteq \mathcal{G}$ vs. $\mathcal{S} = \{(n_1,v) \mid \ldots\}$** — Problem Formulation defines $\mathcal{S}$ as a subgraph; Motif 1 defines it as a set of edge tuples. A subgraph is formally $(\mathcal{V}', \mathcal{E}')$. Fix: Write $\mathcal{E}_\mathcal{S}$ for the edge set or define $\mathcal{S}$ consistently.

### Undefined Notation

10. **[MAJOR] $\hat{\tau}$** — introduced in experiments.tex §"Classifier Accuracy" without prior definition. $\tau$ is the classifier output in method.tex, but $\hat{\tau}$ (predicted type) vs. $\tau$ (ground-truth type) distinction is never defined. | Fix: Define in §"Query Classification" or at the start of §"Classifier Accuracy."

11. **[MAJOR] "Win rate" formula** — prose definition in §4.2 but no mathematical expression. Specifically: how are Ties treated in the denominator (excluded only for "Undefined," not standard ties)? | Fix: Provide a formal equation.

12. **[MAJOR] Confidence score cascade trigger condition** — paper says "when $\tau=\texttt{broad}$ or $c(\mathcal{S}) < 0.04$"; code also triggers on empty retrieval (`paths` empty), which is not mentioned. | Fix: Add the empty-retrieval condition and clarify precedence.

13. **[MINOR] $\ell$ in Steiner approximation ratio $2(1-1/\ell)$** — not defined. Fix: "where $\ell = |\mathcal{A}|$ is the number of anchor nodes."

14. **[MINOR] top-$k$ and top-$m$ in PPR motif** — values never assigned. Code uses `top_n=20`, `limit=15`. Fix: specify or name as hyperparameters.

### Algorithm Description Issues

15. **[MAJOR] Bridge motif table omits edge-keyword path ranking** — The ranking of paths by edge-keyword query overlap (which determines which path is used as the primary chain) is entirely absent from the motif table. This is a distinctive differentiator from plain PathRAG. | Fix: Add to the Algorithm column.

16. **[MAJOR] Cascade description incomplete** — paper omits the empty-retrieval trigger and does not distinguish the cascade-on from the cascade-off (ablation) branch. The description implies cascade fires only on two conditions; code has three. | Fix: Add the third condition.

17. **[MAJOR] Cascade anchor sourcing** — "top VDB candidates as anchors" implies semantic selection; the code uses positional slicing (`candidates[:1]`, `candidates[:2]`, `candidates[:3]`). | Fix: State "mechanically assigns the top 1, 2, or 3 VDB candidates by rank as anchors."

### LaTeX Formatting

18. **[MINOR] tab:winrate_bytype `{lcccc}` → `{lccc}`** — structural LaTeX error (also flagged by Agent 5).

19. **[MINOR] $\mathbf{1}_{\mathcal{N}}$ for indicator vector** — non-standard; prefer $\mathbb{1}_{\mathcal{A}}$ or write out explicitly.

20. **[MINOR] Motif 2 long inline math** — `$\mathcal{S} = \textsc{ShortestPath}(\ldots) \cup \textsc{SimplePaths}(\ldots)$` may overflow a column; consider `equation*`.

---

## 5. Tables, Figures & Documentation

## Agent 5: Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

**tab:winrate**
- [MAJOR] Missing five-dimension names — "averaged across all five evaluation dimensions" without naming them. Add: "(Comprehensiveness, Diversity, Logicality, Coherence, Relevance)."
- [MAJOR] Missing judge/protocol description — caption silent on three judges, position-bias correction, majority-vote mechanism. Add one-sentence note.
- [MAJOR] Bold rule applied inconsistently — caption says "Bold = best per row" but Agriculture (69.2) and CS (80.3) rows have no bolded value. Fix: bold the missing cells or revise the convention.
- [MINOR] NaiveRAG not defined in caption. Add "(LightRAG naive mode)."
- [MINOR] No "Undefined" outcome note — Undefined dimensions are excluded from denominator but not flagged in caption.

**tab:winrate_bytype**
- [CRITICAL] `{lcccc}` declares 5 columns; table has 4 columns — LaTeX structural error. Fix: change to `{lccc}`.
- [MAJOR] Bold rule not stated in caption — bold applied selectively to Single-entity and Bridge rows but not Multi-entity or Broad.
- [MAJOR] Inconsistent bold — Multi-entity best = 64.2 (vs NaiveRAG, not bolded); Broad best = 69.6 (vs LightRAG, not bolded).
- [MAJOR] Missing sample definition — "25 questions per type per domain (100 per type total)" not in caption.
- [MAJOR] Missing judge/dimension information — caption silent on evaluation protocol.

**tab:ultradomain**
- [MAJOR] Bold rule unexplained — all Average-row cells bolded with no stated reason.
- [MAJOR] Missing sample definition — no per-domain question counts stated.
- [MAJOR] Column order inconsistent with other tables — [vs NaiveRAG, vs LightRAG, vs PathRAG] vs. [vs LightRAG, vs NaiveRAG, vs PathRAG] in tab:winrate. Fix: standardize column order.
- [MINOR] Win definition incomplete — no mention of JUDGE_PROMPT_GT, number of judges, or dimensions in this evaluation.

### Figures with Missing or Incomplete Notes

**fig:system**
- [MINOR] Caption does not name the five evaluation dimensions; refers only to "per dimension."

**fig:motifs**
- [MAJOR] Caption's color legend covers only three node roles (bridge, Steiner intermediate, PPR rank) without describing the ego-network's visual representation. Clarify which visual element represents ego-network expansion.
- [MINOR] No note that the diagram is schematic rather than domain-specific.

**fig:confusion**
- [MAJOR] "Counts out of 100" is technically correct (25 × 4 = 100 per row) but ambiguous. Fix: "Cell values are counts out of 100 (= 25 questions/type × 4 domains); each row sums to 100."

**fig:winrate_chart**
- [MAJOR] Caption omits color-to-baseline mapping — color legend is inside the figure but not in the caption. A self-contained caption should state: "Blue = vs LightRAG, green = vs NaiveRAG, orange = vs PathRAG."
- [MINOR] Caption omits sample size ("25 questions per type per domain").

### Cross-Reference Issues

- [CRITICAL] fig:winrate_chart has zero in-text citations — `\label{fig:winrate_chart}` is defined but `\ref{fig:winrate_chart}` appears nowhere. Add: "Figure~\ref{fig:winrate_chart} visualizes the same breakdown."
- [CRITICAL] conclusion.tex "pending" reference to UltraDomain experiment — contradicts §4.5 which reports it fully. Remove or replace.
- [MAJOR] fig:motifs referenced only once (method.tex); not referenced in Experiments or Discussion sections where it would aid comprehension.

### Formatting Inconsistencies

- [CRITICAL] tab:winrate_bytype `{lcccc}` → `{lccc}` — LaTeX structural error.
- [MAJOR] tab:winrate — bold rule stated but not applied to Agriculture and CS rows.
- [MAJOR] tab:ultradomain — column order differs from other tables.
- [MINOR] tab:winrate_bytype — Multi-entity and Broad row maxima not bolded.
- [MINOR] tab:ultradomain — all Average-row cells bolded with no stated reason.

---

## 6. Spelling, Grammar & Style

## Agent 1: Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **[CRITICAL] experiments.tex §4.5** | `"62.5\% vs LightRAG, 59.5\% vs PathRAG, and 52.6\% vs NaiveRAG"` → `"62.5\% vs.\ LightRAG, 59.5\% vs.\ PathRAG, and 52.6\% vs.\ NaiveRAG"` | Missing `\.` after `vs` — abbreviation period — in three consecutive instances; causes potential line-break issues.

2. **[CRITICAL] experiments.tex §4.5** | `"consistent with their well-indexed graphs (10 and 130 documents respectively)"` → `"consistent with their well-indexed graphs (10 documents for CS and 130 for Mix, respectively)"` | Ambiguous antecedent; reader cannot tell which number belongs to which domain.

3. **[CRITICAL] method.tex §3.4** | Variable $\mathcal{N}$ introduced in "Based on $(\tau, \mathcal{N})$" without definition after being used as a different symbol in the neighborhood equation — undefined variable in a critical formal description.

4. **[CRITICAL] experiments.tex §4.5** | `"causing 20 of 199 questions to return no result"` → `"causing 20 of 199 questions to return no results"` | "No result" is non-standard; "return no results" is conventional.

5. **[CRITICAL] introduction.tex** | `"losing the relational structure across entities essential for multi-hop reasoning"` → `"losing the relational structure across entities that is essential for multi-hop reasoning"` | Missing restrictive relative clause marker; "essential" reads as a dangling modifier.

### Major Issues

6. **[MAJOR] method.tex §3.6** | `"hypothesise"` → `"hypothesize"` | UK spelling; IEEE requires US English. Isolated inconsistency — no other UK spellings near this section.

7. **[MAJOR] experiments.tex §4.3** | `"PathRAG is a closer competitor"` → `"PathRAG is the strongest baseline"` | "Closer competitor" is an unanchored comparative (closer than what?); creates a dangling comparative.

8. **[MAJOR] experiments.tex §4.4** | "wins only 16--36%" understates: a win rate below 50% means cascade *hurts*, not merely "does not improve." | Fix: "consistently underperforms the simpler no-cascade variant."

9. **[MAJOR] conclusion.tex** | "The qualitative example demonstrates that…" — no qualitative example is defined or `\ref`'d in the current paper. | Fix: Remove or replace with a concrete cross-reference.

10. **[MAJOR] experiments.tex §4.2** | `"a win is counted only when the forward pass selects Answer~1 \emph{or} the reversed pass selects Answer~2 (i.e., both orderings agree)"` | The logical operator is wrong: "or" means either ordering suffices; the parenthetical says "both orderings agree." Replace `\emph{or}` with `\emph{and}`.

11. **[MAJOR] discussion.tex §5** | "The overall 40.8\% win rate against PathRAG therefore understates GRASP-RAG's advantage" — a 40.8% rate is a *loss*, not a suppressed advantage. | Fix: "…therefore obscures GRASP-RAG's advantage on structured query types…"

12. **[MAJOR] abstract.tex** | Single-entity vs NaiveRAG figure missing from parallel structure with bridge query numbers. Add 75.0% or restructure.

13. **[MAJOR] related_work.tex** | `"labelled"` → `"labeled"` | UK spelling; IEEE standard.

14. **[MAJOR] method.tex §3.5** | $\mathcal{E}_\mathcal{S}$ and $\mathcal{V}_\mathcal{S}$ used in Eq. (1) without definition as the edge set and vertex set of the retrieved subgraph. Add: "where $\mathcal{V}_\mathcal{S}$ and $\mathcal{E}_\mathcal{S}$ are the node and edge sets of $\mathcal{S}$."

### Minor Issues

15. **[MINOR] introduction.tex** | `"neighbourhood"` → `"neighborhood"` (twice; one instance in same document uses US spelling).

16. **[MINOR] experiments.tex §4.1** | `"totalling"` → `"totaling"` | UK English.

17. **[MINOR] experiments.tex §4.5** | Delete `"Notably,"` at sentence start — filler adverb.

18. **[MINOR] method.tex §3.6** | One Unicode em-dash `—` instead of LaTeX `---` in the PGR section; all other em-dashes use `---`.

19. **[MINOR] abstract.tex** | `"Steiner-tree"` hyphenated in abstract; `"Steiner tree"` unhyphenated everywhere else. Standardize to `"Steiner tree"`.

20. **[MINOR] conclusion.tex** | "Pending experiments include a reference-answer evaluation on the UltraDomain benchmark" — UltraDomain results are already reported in §4.5. This is factually incorrect in addition to being an inconsistency.

21. **[MINOR] introduction.tex** | `"because it never read the linguistic signal"` → `"because it ignores the linguistic signal"` | Tense inconsistency; colloquial phrasing.

22. **[MINOR] abstract.tex** | "bias-corrected majority vote" → "a bias-corrected majority vote" | Missing article.

23. **[MINOR] related_work.tex** | `"NLI"` used without expansion at first use. Expand: "natural language inference (NLI)."

### Style Patterns to Fix Throughout

**A. [MAJOR] UK vs. US English** — Four confirmed UK spellings: "hypothesise" (method.tex), "neighbourhood" (introduction.tex ×2), "labelled" (related_work.tex), "totalling" (experiments.tex). IEEE requires US English. Global fix: search-replace all.

**B. [MAJOR] `vs.` without `\.`** — "vs LightRAG" (no period, no `\.`) appears in §4.5 three times. Global fix: search for `vs ` and replace with `vs.\ ` throughout.

**C. [MAJOR] Variable naming drift ($\mathcal{A}$ vs. $\mathcal{N}$)** — §3.3 defines $\mathcal{A}$; §3.4 switches to $\mathcal{N}$. Global fix: pick one symbol.

**D. [MINOR] Filler adverb "Notably"** — one confirmed instance in §4.5. Delete.

**E. [MINOR] Em-dash encoding** — one Unicode `—` in method.tex PGR section; all others use `---`. Replace with `---`.

**F. [MINOR] Cross-reference hygiene** — "the qualitative example" in conclusion.tex has no `\ref{}`; fig:winrate_chart has no in-text citation; eq:conf label defined but never referenced.

---

## Priority Action Items

### CRITICAL (must fix — these could cause desk rejection or major referee objections)

1. **[Agent 6, Part 2]** No statistical significance testing on any win-rate figures; with 25 questions per cell, results are not interpretable without confidence intervals or significance markers.

2. **[Agent 6, Part 3]** PGR ablation missing: the second named contribution (Path-Guided Reasoning) cannot be justified without "bridge motif + flat context" vs. "bridge motif + PGR chain." The paper's own Limitations section admits this is unsubstantiated.

3. **[Agent 6, Part 5]** Paper is 7 pages; APSIPA 2026 limit is 5–6 pages. Submission will be desk-rejected or require mandatory shortening.

4. **[Agent 6, Part 2 + Agent 3]** GPT-4o-mini serves as both the answer-generation backbone and one of three LLM judges — self-preference confound is unaddressed.

5. **[Agent 6, Part 3]** Legal KG is incomplete (438 docs, sparse graph, 20/199 questions return no result). Evaluating on a broken knowledge base is not a legitimate experimental condition.

6. **[Agent 3]** LLM-generated typed questions are structurally biased toward graph-traversable shapes, inflating all Experiment 1 numbers. This circularity is understated in Limitations.

7. **[Agent 2]** Broad motif described as "PPR diffusion" in Method §3.4 but as "LightRAG hybrid" in Experiments, Discussion, and Conclusion — the descriptions are irreconcilable and one is factually wrong.

8. **[Agent 1]** Bias-correction logic uses "or" instead of "and" in §4.2 — the stated protocol and the logical operator directly contradict each other.

9. **[Agent 4]** Confidence score formula (Eq. 1) numerator does not match the code: paper says distinct edge count; code counts total path-edge traversals including duplicates.

10. **[Agent 4]** "Bidirectional shortest path" mislabels the algorithm in the motif table and Discussion; the code runs standard `nx.shortest_path` on an undirected graph.

11. **[Agent 2 + Agent 5]** tab:winrate_bytype `{lcccc}` declares 5 columns; table has 4 — LaTeX structural error causing alignment artifacts.

12. **[Agent 2]** tab:winrate_bytype "Overall" row arithmetic does not match simple average of the per-type rows (reported 81.1, computed 80.0 for LightRAG column).

13. **[Agent 2 + Agent 5]** Conclusion says UltraDomain evaluation is "pending" while §4.5 and Table 3 report it in full — factual internal contradiction.

### MAJOR (should fix — will likely be raised by referees)

14. **[Agent 6]** Broad motif delegates entirely to LightRAG hybrid for 25% of test questions; the 81.1% aggregate win rate includes wins where GRASP-RAG is literally running LightRAG. Report disaggregated win rates excluding broad queries.

15. **[Agent 6]** Cascade mechanism failure (16–36% win rate vs. no-cascade) is unexplained; confidence score calibration and verifier reliability are unexamined.

16. **[Agent 6]** No standard multi-hop QA benchmark evaluation (HotpotQA, 2WikiMultiHopQA); BridgeRAG is cited but not compared against.

17. **[Agent 3]** Verbosity bias in LLM-as-judge evaluation not addressed; PGR's structured format may produce longer answers that judges favor independent of quality.

18. **[Agent 1]** "win rate against PathRAG understates GRASP-RAG's advantage" is contradictory — 40.8% is a loss, not a suppressed advantage.

19. **[Agent 4]** $\mathcal{A}$ vs. $\mathcal{N}$ used for the same concept (anchor nodes) in adjacent subsections with no bridging definition; $\mathcal{N}(\cdot)$ also introduced as a neighborhood function in the same block.

20. **[Agent 2]** Discussion claims 97.6% (Legal, bridge vs. LightRAG) and 100% (Mix, bridge vs. LightRAG) — figures not in any table and unverifiable from the paper.

21. **[Agent 5]** tab:winrate bold rule stated in caption but not applied to Agriculture and CS rows.

22. **[Agent 5]** tab:ultradomain column order [NaiveRAG, LightRAG, PathRAG] differs from [LightRAG, NaiveRAG, PathRAG] in the other two tables.

23. **[Agent 2]** `zheng2023judging` (foundational LLM-as-judge paper) is in the bibliography but never cited, despite the paper adopting an LLM-as-judge protocol.

24. **[Agent 1]** Four UK spellings throughout (hypothesise, neighbourhood ×2, labelled, totalling) — IEEE requires US English.

25. **[Agent 4]** Bridge motif table omits the edge-keyword path-ranking step that determines which path is used as the primary reasoning chain — a distinctive differentiator from PathRAG.

### MINOR (polish — improves paper quality)

26. **[Agent 5]** fig:winrate_chart has no in-text citation; figure may float arbitrarily.

27. **[Agent 1]** `vs LightRAG` (no period, no `\.`) appears three times in §4.5; should be `vs.\ LightRAG` throughout.

28. **[Agent 1]** Delete filler adverb "Notably" at start of §4.5 sentence.

29. **[Agent 1]** One Unicode em-dash `—` in method.tex PGR section; all others use LaTeX `---`.

30. **[Agent 1]** "Steiner-tree" hyphenated in abstract; "Steiner tree" everywhere else — standardize.

31. **[Agent 4]** $\hat{\tau}$ (predicted query type) used in experiments §4.4 without prior formal definition.

32. **[Agent 4]** $\ell$ in Steiner approximation ratio $2(1-1/\ell)$ never defined; add "where $\ell = |\mathcal{A}|$."

33. **[Agent 2]** "query shape" / "logical shape" / "query type" used interchangeably; standardize to "query type."

34. **[Agent 2]** Gemini judge version: "Gemini Flash" vs. "Gemini 2.0 Flash" — inconsistent across abstract, experiments, and fig:system caption.

35. **[Agent 1]** `\label{eq:conf}` defined but never `\ref`'d; add cross-reference in cascade discussion or remove label.

---

*Report generated: 2026-06-03*
*Agents: 6 (Spelling/Grammar, Consistency, Claims, Notation, Tables/Figures, Contribution)*
*Issues flagged: 15 CRITICAL · 18 MAJOR · ~12 MINOR (abbreviated)*
