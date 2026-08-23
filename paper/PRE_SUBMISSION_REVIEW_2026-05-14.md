# Pre-Submission Referee Report

**Paper**: GRASP-RAG: Graph Retrieval with Adaptive Structured Paths and Path-Guided Reasoning  
**Authors**: [Placeholder — to be filled before submission]  
**Date**: 2026-05-14  
**Review Standard**: APSIPA 2026 (Asia Pacific Signal and Information Processing Association Annual Summit and Conference, Hanoi)

---

## Overall Assessment

GRASP-RAG introduces query-adaptive graph motif selection — routing queries to one of four traversal algorithms based on their logical shape — combined with topology-mirroring context formatting (Path-Guided Reasoning), achieving 100% Bridging Entity Recall on bridge queries across three UltraDomain domains versus 57.5–97.5% for three baselines. The Legal domain finding (PathRAG = NaiveRAG = 57.5%) is the paper's strongest and most memorable result, providing credible evidence that query-type-unaware path retrieval can be actively harmful. However, the headline 100% BER figure is presented in the Abstract and Conclusion without the critical qualifier — acknowledged only in Discussion — that questions are generated from the same graph GRASP-RAG queries, making the result near-guaranteed under correct classification; the paper must report classifier accuracy, anchor recovery rates, and at least one ablation separating its two claimed innovations before the result is interpretable.

**Preliminary Recommendation**: Major Revision

---

## 1. Contribution & Referee Assessment

### Part 1 — Central Contribution

**Stated contribution:** GRASP-RAG proposes query-adaptive graph motif selection (classifying queries into four structural types and routing to matching traversal algorithms) combined with topology-mirroring context formatting (Path-Guided Reasoning), demonstrated to achieve 100% Bridging Entity Recall on bridge queries across three UltraDomain domains versus 57.5–97.5% for baselines.

**Is this genuinely new?** The core insight — that different query shapes warrant different graph traversal strategies — is intellectually sound and not trivially replicated in the existing graph-RAG literature. HippoRAG, LightRAG, G-Retriever, and PathRAG all use a single retrieval strategy uniformly. The combination of query classification driving motif dispatch, plus topology-aware context linearisation, has not appeared in this exact form in any cited paper. However, the idea of query routing (classifying queries to select different retrieval pipelines) is well-established in the broader RAG and multi-hop QA literature, and the paper does not engage with that literature at all.

**Closest prior paper:** PathRAG (Chen et al., 2025) is the closest baseline. The key differentiator is that PathRAG applies uniform path extraction to all query types, whereas GRASP-RAG selects the traversal algorithm conditioned on the query's logical shape — a genuine and meaningful distinction, cleanly illustrated by the Legal domain result.

**Contribution rating: Borderline**

The core idea is legitimate and well-motivated, but the scope of empirical support is severely limited to a single motif (bridge) where the evaluation metric is designed in a way that near-guarantees 100% for the proposed method under correct classification. The paper's own Discussion section acknowledges this partial circularity clearly, but the Abstract and Conclusion present the 100% figure without adequate qualification. The contribution is original enough for the venue but the evaluation as presented does not fully substantiate the system-level claims.

### Part 2 — Technical Soundness

The architecture is sound in principle. Using VDB-retrieved candidate entity names to constrain the classifier is a practical design choice that reduces entity-linking errors. The four motifs map cleanly to graph-theoretic formulations with appropriate citations. However, the confidence-based fallback (Equation 1) threshold of 0.04 is not derived or justified — it appears to be an engineering choice presented as principled.

BER is deterministic and judge-free, which are genuine strengths. However, BER is constructed from top-150 degree nodes, which introduces selection bias: the benchmark over-represents hub entities, exactly the entities GRASP-RAG's VDB search is most likely to return as anchor candidates.

The paper does not report: (a) what fraction of queries the classifier correctly identified as bridge type, (b) what fraction had both anchors in the VDB top-20, or (c) BER on the subset where classification or anchor retrieval failed. Without these figures, the 100% headline is not interpretable. Additionally, GRASP-RAG makes 2 LLM API calls per query (classify + generate) while baselines make 1 — this cost differential is unacknowledged.

**Skeptical reviewer's question:** "You wrote the test questions from your own graph's 2-hop paths, then ran your system on that graph, and got 100%. Show me the classifier accuracy on those 120 questions and the BER when VDB misses an anchor, and then I'll believe the 100%."

### Part 3 — Required and Suggested Analyses

**Required:**

1. [CRITICAL] **Classifier accuracy breakdown.** Report what fraction of the 120 bridge queries were correctly classified as `bridge` and had both anchors in the VDB top-20. The 100% BER headline is uninterpretable without this. A positive result (e.g., 97/120 correctly classified) would show that the guarantee holds in practice and give honest failure-mode accounting.

2. [CRITICAL] **Per-query cost and latency.** GRASP-RAG makes ≥2 LLM calls per query vs. 1 for baselines. Report average tokens consumed and latency per domain for all four methods. Without this, the "practical improvement" claim is unsubstantiated.

3. [CRITICAL] **Error analysis on baseline failures (especially PathRAG Legal).** The paper asserts flow-based pruning removes specific 2-hop bridges in legal text. No diagnostic evidence is provided — PathRAG's retrieved subgraphs are not inspected. A 10-case qualitative analysis of PathRAG Legal failures would validate or refute the mechanistic claim.

4. [CRITICAL] **Fallback frequency and behavior.** Is Equation 1's 0.04 threshold ever triggered? If never, the fallback is untested. If frequently, it complicates the BER attribution. Report fallback frequency per domain and BER of fallback vs. non-fallback queries.

5. [CRITICAL] **Quantitative evaluation of at least one additional motif.** The paper claims a "complete four-motif system" but evaluates only bridge. The single-entity motif comparison against LightRAG local mode is trivially achievable given existing infrastructure. Even a null result (marginal gains) strengthens credibility.

**Suggested:**

1. [MAJOR] **Cross-graph generalization test.** Construct BER questions from one graph build, answer using a second build over the same corpus. Directly addresses partial circularity. Feasibility: moderate.

2. [MAJOR] **Ablation: motif selection vs. PGR formatting.** Run bridge queries with (a) correct motif + flat context, and (b) PGR + fixed path retrieval regardless of type. Isolates each claimed innovation's contribution. Feasibility: high.

3. [MAJOR] **LLM-as-judge re-evaluation with bridge-aware judge prompt.** The paper reports GRASP-RAG scored lower on LLM-as-judge despite perfect BER. Run evaluation restricted to bridge queries with a judge prompt that rewards naming the bridge entity. Feasibility: high.

4. [MAJOR] **Sensitivity of BER benchmark to entity selection.** Report BER when questions are constructed from uniformly sampled nodes (not degree-biased top-150). Tests whether results hold for rare entities. Feasibility: moderate.

5. [MAJOR] **Include the CS domain.** "Consistent results are expected" is not a result. The infrastructure exists; this is a compute/time issue. Feasibility: straightforward.

### Part 4 — Literature Positioning

**Serious omissions:**

- **Query routing / Adaptive RAG:** The paper's central contribution is query classification driving retrieval routing. Adaptive-RAG (Jeong et al., 2024), Self-RAG (Asai et al.), and complexity-adaptive QA approaches are not cited. A reviewer from this community will ask immediately.
- **Multi-hop QA:** Bridge queries are 2-hop multi-hop QA. HotpotQA, MuSiQue, 2WikiMultihopQA and path-based evaluation methods from that literature are absent.
- **KGQA literature:** The four-motif taxonomy mirrors KGQA query decomposition strategies.
- **Orphaned bibliography entries:** `thirunavukarasu2023large`, `zakka2024almanac`, and `gao2023retrieval` appear in references.bib but are never cited in the paper. Remove them or cite them.

The paper does well distinguishing from PathRAG (the closest baseline) — the mechanistic explanation for PathRAG's Legal failure is the paper's strongest analytical contribution.

### Part 5 — Venue Fit and Recommendation

**Recommendation: Major Revision**

The paper has genuine novelty in its motif dispatch architecture, a well-motivated metric, and an empirically honest Discussion. However, the 100% BER headline is presented without its crucial qualification in the paper's most visible sections. The absence of classifier accuracy, cost comparison, motif ablation, and any evaluation beyond bridge collectively prevent confident acceptance.

**Path to acceptance:**
1. Report classifier accuracy and VDB anchor recovery rates for the 120 bridge queries.
2. Add motif-selection vs. PGR ablation.
3. Remove uncited bibliography entries.
4. Add adaptive RAG and multi-hop QA citations.
5. Report per-query cost comparison.
6. Add BER interpretation qualifier to Abstract and Conclusion (not only Discussion).

**Best alternative venue if rejected:** EMNLP Findings 2026 or ACL Findings 2026 with full revision; ECIR 2027 or EACL 2026 for shorter path.

### Part 6 — Questions to the Authors

1. Table 2 reports 100% BER across 120 bridge queries. What fraction did the LLM classifier correctly label as `bridge`, and what fraction had both anchors in the VDB top-20? How were misclassified or anchor-missing queries handled — fallback or excluded from the denominator?

2. How frequently was Equation 1's confidence threshold of 0.04 triggered in the experiments? What is the empirical basis for this threshold? Did fallback queries contribute to the 100% BER denominator?

3. The paper claims two complementary innovations — motif selection and PGR. Have you run an ablation with (a) bridge motif + flat context and (b) PGR formatting + fixed path retrieval? Without this, it is impossible to attribute the BER gain to motif selection versus PGR.

4. Section 4.4 reports GRASP-RAG scored lower on LLM-as-judge despite perfect BER, attributed to length bias. Were the LLM-as-judge evaluations restricted to bridge queries, or did they include all query types? Could the step-by-step format reduce comprehensiveness on non-bridge questions?

5. BER questions are generated from the top-150 highest-degree nodes. Does the 100% BER result hold when evaluated on questions constructed from randomly sampled low-degree nodes, where VDB anchor recovery is less certain?

6. The CS domain is omitted with the assertion "consistent results are expected." What is the basis for this expectation? If CS results deviate, how would that change the paper's domain robustness claim?

7. GRASP-RAG makes at least 2 LLM API calls per query versus 1 for all baselines. What is the average per-query token consumption and latency for GRASP-RAG versus PathRAG and LightRAG?

---

## 2. Unsupported Claims & Technical Integrity

### Critical Overclaiming (must address)

1. [CRITICAL] Abstract | "demonstrating that query-adaptive structured retrieval substantially and consistently improves multi-hop reasoning" — only the bridge motif on bridge questions from the same graph is tested. "Multi-hop reasoning" is a broad capability claim. Fix: "demonstrating that query-type-aware path selection structurally guarantees intermediate entity recall on bridge queries within the same knowledge graph."

2. [CRITICAL] Abstract (and throughout) | "a bias-free complement to LLM-as-judge evaluation" — False in two respects: (a) substring match can false-positive on partial entity name matches, and (b) top-150 degree-node sampling biases toward hub entities most likely to appear in any retrieved context. Fix: Replace "bias-free" with "judge-free and length-bias-free" and add a sentence acknowledging substring false-positive risk and degree-based sampling bias.

3. [CRITICAL] Introduction (contribution list, point 4) | "Bridging Entity Recall (BER): a ground-truth, bias-free metric for bridge queries" — same issue. "Ground-truth" overstates given same-graph circular construction. Fix: "a judge-free, length-unbiased metric; results reflect structural guarantee conditions rather than open-domain generalization."

4. [CRITICAL] Conclusion | "GRASP-RAG, a graph RAG framework that advances the state of the art along two dimensions" — unqualified. The paper tests only the bridge motif on 3 domains with no established NLP benchmark. Fix: scope to "demonstrates a structural advantage on bridge queries within UltraDomain" or use "we introduce."

5. [CRITICAL] Discussion (§Limitations) | "bidirectional shortest-path is *guaranteed* to recover the 2-hop path" but earlier says "near-inevitable." The word "near" softens a mathematical certainty. Fix: Quantify the non-trivial failure modes (misclassification rate, VDB anchor miss rate) with empirical estimates, or use "guaranteed" throughout and scope the guarantee conditions clearly.

### Scope and Generalization Issues

6. [MAJOR] Section 4.1 | "CS omitted due to time constraints; consistent results are expected" — unverified prediction stated alongside actual results. Appears twice verbatim. Fix: State omission only; remove the expectation clause, or footnote it as speculation.

7. [MAJOR] Section 4.3 | "The single-entity motif is behaviorally similar to LightRAG's local mode, so marginal gains there are expected to be small" — asserted without measurement. Fix: Add "we conjecture" and acknowledge this is untested.

8. [MAJOR] Section 4.3 | "The broad motif explicitly delegates to LightRAG hybrid, so its performance equals LightRAG by design" — PGR context formatting still applies and could alter LLM behavior. Fix: Clarify whether PGR is bypassed for broad motif, or acknowledge downstream formatting differences may affect results.

9. [MAJOR] Discussion | "demonstrating that GRASP-RAG's advantage is not domain-specific" — three domains, one task type, same graph. Fix: "suggesting domain robustness on the bridge task across the three tested UltraDomain domains."

10. [MINOR] Discussion | "confirming that its structural guarantee is independent of corpus size" — inferred from three data points. Fix: "consistent with."

### Missing Ablations / Experimental Gaps

11. [CRITICAL] Missing: "Always use bridge motif" oracle baseline — without this, the classification step's contribution to BER cannot be assessed. If always-bridge also gets 100%, the adaptive design provides no BER benefit.

12. [CRITICAL] Missing: Classifier accuracy measurement — the paper never reports what fraction of bridge questions were correctly classified. The 100% guarantee is conditional on correct classification; without this fraction, the result is uninterpretable.

13. [CRITICAL] Missing: PGR ablation — no comparison of "bridge motif + flat context" vs. "bridge motif + PGR context." The contribution of each claimed innovation is undisentangled.

14. [MAJOR] Missing: VDB top-20 anchor recall — the structural guarantee requires both anchor nodes in the top-20; this condition is never empirically verified.

15. [MAJOR] Missing: Evaluation beyond BER — no human evaluation, factual accuracy assessment, or open benchmark is provided to assess whether precise BER trades off against other answer quality dimensions.

### Mechanism Claims Requiring Hedging

16. [MAJOR] Section 4.3 / Section 5 | "PathRAG's flow-based pruning can actively remove the bridge edge in legal text" — presented as established fact; PathRAG's retrieved subgraphs are not inspected to verify. Fix: "We hypothesize that PathRAG's flow-based pruning removes low-frequency bridge edges in legal text…"

17. [MAJOR] Section 4.3 | "entities tend to co-occur frequently in narrative text" (Mix domain explanation) — co-occurrence frequency is not measured; the small-corpus explanation and co-occurrence explanation are conflated. Fix: State the size explanation (supported by Table 1) is observed; co-occurrence is speculative.

18. [MINOR] Section 5 | "makes the bridge entity structurally unavoidable" — the guarantee is conditional on correct classification and VDB anchor recovery. Fix: Add "provided both anchors are correctly identified and appear in the candidate list."

### Priority / First-to-Show Claims

19. [MAJOR] Throughout (implicit) | The paper frames GRASP-RAG as the first to perform query-adaptive motif selection over a KG for RAG, but never explicitly verifies this. Fix: Add "To our knowledge, no prior system conditions graph traversal motif selection on explicit query-type classification with candidate-informed anchor extraction."

20. [MINOR] Introduction (point 3) | "eliminating fuzzy-match errors" — "eliminating" is too strong; VDB can return incorrect nodes. Fix: "substantially reducing."

---

## 3. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. [CRITICAL] Discussion §Domain Robustness ↔ Table 2 | Gap "22.5–42.5 percentage points" — the stated upper bound of 42.5 pp is from Legal (100−57.5). But Agriculture vs. NaiveRAG = 100−55.0 = **45.0 pp**, which is larger and is in Table 2. The true observed maximum gap is 45.0 pp. Fix: Change to "22.5–45.0 percentage points" or specify "22.5–27.5 pp relative to the best-performing baseline."

### Numerical Verification

| Claim | Verdict |
|---|---|
| GRASP avg = 100.0% | VERIFIED |
| PathRAG avg = 75.8% | VERIFIED (75.83%) |
| LightRAG avg = 79.2% | VERIFIED (79.17%) |
| NaiveRAG avg = 65.8% | VERIFIED (65.83%) |
| 120/120 bridge entities (3×40) | VERIFIED |
| Table 1 Agriculture baseline avg = 66.7% | VERIFIED |
| Table 1 Legal baseline avg = 62.5% | VERIFIED |
| Table 1 Mix baseline avg = 91.7% | VERIFIED |
| PathRAG Legal = NaiveRAG Legal = 57.5% | VERIFIED |
| Gap upper bound "42.5 pp" | **ERROR — should be 45.0 pp** |

### Missing or Broken Cross-References

2. [MINOR] `\label{fig:ber_chart}` — Figure 4 has a label but is **never cited in the text** via `\ref{fig:ber_chart}`. Add: "Figure~\ref{fig:ber_chart} visualises the per-domain BER scores" in Section IV-C.

3. [MINOR] `\label{eq:conf}` — Equation 1 is labeled but never referenced. Add `(Eq.~\ref{eq:conf})` at line 281 where the threshold is invoked.

4. [MINOR] `\label{sec:classify}` — Section 3.2 is labeled but never cross-referenced. The Introduction contribution list references sec:motif, sec:pgr, sec:ber but skips sec:classify. Add a reference in contribution point 1 or remove the dangling label.

5. [MINOR] `\label{sec:focus}` — Section 4.2 labeled but never cited.

6. [MINOR] Uncited bibliography entries: `gao2023retrieval`, `thirunavukarasu2023large`, `zakka2024almanac` are defined in references.bib but never cited in main.tex. Remove or cite.

### Terminology Drift

7. [MINOR] "query type" vs. "query shape" — used interchangeably. Standardize on **"query type"** (the formal symbol is τ); use "logical shape" only as an informal gloss on first introduction.

8. [MINOR] "retrieval motif" vs. "graph motif" — "graph motif" in the Introduction (line 118) is slightly misleading (the term has a different meaning in graph theory). Standardize on **"retrieval motif"**.

9. [MINOR] "bridge entity" vs. "intermediate entity" vs. "bridging entity" — formal metric uses "bridging entity" (BER) but body text alternates with "intermediate entity." Standardize on **"bridge entity"**.

10. [MINOR] "VDB" — used before any expansion. Spell out "vector database (VDB)" on first use.

11. [MINOR] Table 2 column header "GRASP" — the system is consistently named "GRASP-RAG" everywhere else (title, abstract, Figure 4 legend). Fix: rename column header to "GRASP-RAG."

12. [MINOR] "CS omitted…" paragraph appears verbatim twice (Section 4.1 setup and Section 4.3 results). Remove one instance.

13. [MINOR] `\res{...}` placeholders in author/affiliation fields render as bold blue text in the compiled PDF. Fill in before submission.

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors or Concerns

1. [MAJOR] Confidence formula (Eq. 1) — Division by zero when |V_S| = 0 or |V_S| = 1 (denominator C(|V_S|,2) = 0). These are exactly the degenerate subgraphs the fallback is supposed to handle. Fix: Define c(S) = 0 whenever |V_S| ≤ 1 and state this explicitly.

2. [MAJOR] Steiner tree approximation ratio — The paper states "2(1−1/ℓ)-approximation" citing Kou et al. 1981 but never defines ℓ. This is likely |N| (number of anchor/terminal nodes). Fix: Add "where ℓ = |N| is the number of terminal nodes."

3. [MINOR] BER formula — m (method index) in BER(m) collides with m (integer count of top-m nodes) used in the PPR motif definition just above. Fix: Rename the PPR parameter to a different letter (e.g., top-r nodes).

4. [MINOR] BER formula — `v_bridge ∈ answer` uses set-membership notation ∈ for substring containment, which is technically incorrect. Fix: Use `\text{contains}(\text{answer}_{i,m},\ v_\text{bridge}^{(i)})` and keep the prose clarification.

### Notation Inconsistencies

5. [MAJOR] **A vs. N for anchor nodes** — Section 3.2 uses `\mathcal{A}` for anchor nodes (`(τ, A) = Classify(q, C)`). Section 3.3 switches without explanation to `\mathcal{N}`: "Based on (τ, N)…". Whether A and N are the same set is never stated. Fix: Unify to a single symbol (e.g., always `\mathcal{A}`) or explicitly define `N := A`.

6. [MAJOR] **N used as neighborhood function AND anchor set** — Motif 1 writes `S = {(n_1, v) | v ∈ N(n_1)}` where `N(n_1)` is the neighborhood of n_1; but `N` is also used as the anchor set in Motifs 3 and 4. Same calligraphic symbol, two different mathematical roles, within the same section. Fix: Use `\text{Nbr}(n_1)` or `\mathcal{N}_G(n_1)` for neighborhoods; reserve `\mathcal{A}` for anchors.

7. [MAJOR] Plain **N in BER formula** (number of questions) vs. calligraphic **N for anchor set** — visually distinct in LaTeX but never called out in prose. Fix: explicitly define "where N is the number of evaluation questions" in the BER paragraph.

### Undefined Notation

8. [MAJOR] `E_S` and `V_S` — Used in Eq. (1) without prior definition. Fix: Add "where `V_S` and `E_S` are the node and edge sets of `S`, respectively" immediately before Eq. (1).

9. [MAJOR] `cutoff=3` in Bridge Path motif — the parameter value is never justified. Fix: Add "limiting enumeration to paths of at most 3 edges (sufficient to capture one additional intermediate node beyond the bridge)."

10. [MAJOR] `k` and `m` in PPR motif — "top-k edges incident to top-m ranked nodes" but k and m are never defined. Fix: State their values in the experimental setup or motif definition.

### Algorithm Specification Issues

11. [MAJOR] Bridge Path motif — `ShortestPath(G, n_1, n_2) ∪ SimplePaths(G, n_1, n_2, cutoff=3)` — ShortestPath is always a member of SimplePaths, making the first term redundant in the union. Likely intent is to use bidirectional ShortestPath for efficiency and SimplePaths for near-shortest alternatives. Fix: Either write `S = SimplePaths(G, n_1, n_2, cutoff=3)` or clarify the algorithmic role of each term.

12. [MAJOR] "computed bidirectionally" — it is unclear whether (a) both terms use bidirectional algorithms, (b) only ShortestPath does, or (c) the union includes both directions for directed graphs. Fix: Clarify whether G is directed or undirected and specify what "bidirectionally" means algorithmically.

### LaTeX Math Formatting

13. [MINOR] Eq. (1) is numbered but never referenced in the text. Fix: Add "(Eq.~\ref{eq:conf})" at the fallback threshold sentence.

14. [MINOR] BER and Classify equations are unnumbered but Equation 1 is numbered. Consider numbering the BER definition (the paper's main metric) as Eq. (2) for easier reference.

15. [MINOR] `\frac{1}{|\mathcal{N}|}\mathbf{1}_{\mathcal{N}}` as inline math — `\frac` is cramped inline. Fix: Use `\tfrac` or display math.

16. [MINOR] `\text{answer}_i^m` — method m as superscript on a text word is typographically unusual. Fix: Change to `\text{answer}_{i,m}`.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Information

**Table 1 (Domain Statistics)**

1. [MINOR] Column header "~Tokens" does not specify tokenizer. Add note: "Counted with `cl100k_base` (GPT-4o-mini)."
2. [MINOR] Caption says "Baseline avg. BER" but table header is shorter "Baseline BER" — make consistent.
3. [MINOR] No note states BER is computed over 40 bridge questions per domain. Add to caption.
4. [MINOR] Legal domain (438 docs, 26.7M tokens → only 1,986 nodes) is counterintuitive; add symbol (†) pointing to Discussion paragraph.

**Table 2 (BER Results)**

5. [MAJOR] Column header **"GRASP"** should be **"GRASP-RAG"** — inconsistent with title, abstract, Figure 4 legend, and all body text.
6. [MAJOR] No confidence intervals or statistical significance reported. N=40 per cell; 95% Wilson CI spans ±7–16 pp for interior values. Add a table note: "No repeated runs; 95% CI at N=40 spans ±7–16 pp for interior values."
7. [MINOR] Caption does not state which backbone LLM generated answers. Add: "All methods use GPT-4o-mini for answer generation."
8. [MINOR] Caption does not define BER for a standalone reader. Add: "BER: fraction of answers containing the ground-truth bridge entity (case-insensitive substring)."

**Table 3 (Qualitative Example)**

9. [MAJOR] Content is rendered as a `tcolorbox` inside a `table` float — IEEEtran proceedings may not accept this. Confirm template compatibility or relabel as "Example 1."
10. [MINOR] Caption says "Ground-truth bridge entity in bold" but bold is applied to both the entity ("Cows") and structural labels ("Q:", "GRASP-RAG", etc.) — ambiguous. Add: "Bold entity names denote the ground-truth bridge; bold labels (Q:, GRASP-RAG, etc.) are structural."
11. [MINOR] LightRAG's failure in the qualitative example receives no interpretation in the text — only PathRAG's failure is discussed. Add 1–2 sentences or a note in the caption.
12. [MINOR] Caption does not name the domain. Change to: "Qualitative example: bridge question from the **Agriculture** domain."

### Figures with Missing or Incomplete Documentation

**Figure 1 (Pipeline Diagram)**

13. [MAJOR] Caption does not explain "LLM → a" (LLM generating answer a), "top-20 nodes" (VDB returns 20 candidates), or "Q" (incoming query). Fix: "Q = incoming query; VDB = vector database (top-20 candidate entity nodes returned); (τ, A) = inferred query type and anchor nodes; PGR Builder = Path-Guided Reasoning context builder; LLM→a = LLM generating answer a."
14. [MINOR] All four motif boxes route to PGR Builder in the diagram, but the broad motif delegates to LightRAG hybrid (per Section 4.2) and does not go through PGR. The diagram is architecturally inaccurate. Add a note in caption or diagram.

**Figure 2 (Four Retrieval Motifs)**

15. [MINOR] Caption should specify panel order: "(a) Ego-network, (b) Bridge path, (c) Steiner tree, (d) PPR diffusion."
16. [MINOR] "green is a Steiner intermediate" → "green nodes are Steiner tree intermediate nodes."

**Figure 3 (PGR vs. Flat Context)**

17. [MINOR] Caption does not identify the specific query shown. Add: "Example from the Agriculture domain: query 'What connects Rational Grazing to Polyface Farm?'"
18. [MINOR] "flat retrieval context" is ambiguous — clarify: "flat retrieval context used by PathRAG and LightRAG."

**Figure 4 (BER Bar Chart)**

19. [MINOR] y-axis starts at 40% — truncated axis visually exaggerates gaps. Add note: "y-axis starts at 40%; absolute values in Table 2."
20. [MINOR] Caption does not state N=40 questions per bar. Add.

### Cross-Reference Issues

21. [CRITICAL] Figure 4 (`fig:ber_chart`) — defined at line 483 but **never cited anywhere in the main text**. Add "Figure~\ref{fig:ber_chart} visualises per-domain BER scores" in Section IV-C.

22. [MAJOR] `\label{sec:classify}` (Section 3.2) — defined but never cross-referenced. The Introduction contribution list references sec:motif, sec:pgr, sec:ber but omits sec:classify. Add reference in contribution bullet or remove the label.

23. [MINOR] `\label{sec:focus}` (Section 4.2) — defined but never cited.

### Formatting Inconsistencies

24. [MINOR] Figure 4 could benefit from `figure*` (two-column) layout — 4 methods × 3 domains in a single-column bar chart is cramped.
25. [MINOR] Abstract lists baselines as "LightRAG (79.2%), PathRAG (75.8%)" — reverses the column order in Table 2 (PathRAG then LightRAG). Standardize ordering everywhere to match Table 2.

---

## 6. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. [CRITICAL] Throughout | British spelling "linearised/linearises/serialise/neighbourhood(s)/labelled" → American English "linearized/linearizes/serialize/neighborhood(s)/labeled" | IEEE format requires American English. Check globally: `-ise` → `-ize`, `neighbourhood` → `neighborhood`, `labelled` → `labeled`.

2. [CRITICAL] Abstract line 75–76 + Section 4.3 + Conclusion | Baseline ordering inconsistency: Abstract lists "LightRAG, PathRAG, NaiveRAG"; Table 2 columns are "GRASP / PathRAG / LightRAG / NaiveRAG"; Section 4.3 lists "GRASP-RAG, LightRAG, PathRAG, NaiveRAG." Pick one ordering and apply it everywhere.

3. [CRITICAL] Section 3.3 line 251 | `(\tau, \mathcal{N})` for anchor nodes, but Section 3.2 uses `\mathcal{A}` for the same concept. Two different symbols for the same object within three lines. (Also flagged as MAJOR in Agent 4 — fix once resolves both.)

4. [CRITICAL] Section 3.3 lines 251 and 255 | `\mathcal{N}` simultaneously means "anchor node set" (line 251) and "neighborhood function N(n_1)" (line 255). Same symbol, two different mathematical roles. (Also flagged in Agent 4.)

5. [CRITICAL] Section 4.2 line 384 | "demonstrably fail on a **significant** fraction of them" — "significant" here means "large," not statistically significant. Fix: "a substantial fraction."

### Major Issues

6. [MAJOR] Throughout | Evaluative filler: "interesting" (line 488), "notably" (line 350), "striking" (line 493), "notable" (line 648) — delete or replace with factual characterizations. The findings speak for themselves.

7. [MAJOR] Section 5, Discussion heading | "Why graph size and corpus density determine baseline BER" — "density" is mentioned in the heading but the body paragraph does not quantify corpus density. Align heading with what the paragraph proves.

8. [MAJOR] Discussion §Domain Robustness line 605 | Gap range "22.5–42.5 percentage points" — upper bound is understated (Agriculture vs NaiveRAG = 45.0 pp). Fix: "22.5–45.0 percentage points." (Also flagged in Agent 2.)

9. [MAJOR] Conclusion line 648 | "A notable finding:" — delete the label; restructure as "On the Legal domain, PathRAG's flow-based pruning without query-type awareness matches NaiveRAG (57.5% each)…"

10. [MAJOR] Section 4.1 line 350 | "The Mix domain is **notably** the smallest corpus" — delete "notably."

11. [MAJOR] Section 4.3 line 488 | "The baselines show **interesting** domain-dependent variation" — delete "interesting"; "The baselines vary substantially across domains."

12. [MAJOR] Section 2 line 148 | "Flow-based pruning extracts relational paths…" — subject is ambiguous (PathRAG not stated). Fix: "PathRAG uses flow-based pruning to extract relational paths…"

### Minor Issues

13. [MINOR] Introduction line 91 | "losing the relational structure **across** entities" → "among entities" | "across" is informal for relationships between items in a set.

14. [MINOR] Introduction line 110 | "it never classified the query" — pronoun "it" is ambiguous. Fix: "because PathRAG never classifies the query…"

15. [MINOR] Section 3.3 Motif 4 line 270–272 | "...returns top-k edges…" — sentence fragment (no subject after semicolon). Fix with a full sentence.

16. [MINOR] Section 4.1 lines 343–346 | "Backbone: GPT-4o-mini… embeddings: text-embedding-3-small. Baselines: NaiveRAG…" — sentence fragments. Rewrite as complete sentences.

17. [MINOR] Section 4.1 line 339 | "consistent results are expected" — weakens the paper. Remove the second clause.

18. [MINOR] Section 4.3 line 489 | "literary/biographical" — use "literary and biographical" in formal prose.

19. [MINOR] Section 5 line 580 | "labelled" → "labeled."

20. [MINOR] Section 5 line 615 | "bidirectional shortest-path is *guaranteed*" — passive. Fix: "bidirectional shortest-path *guarantees* recovery…"

21. [MINOR] References | `ding2024ultradomain` — `journal = {arXiv preprint}` lacks the arXiv ID number. Fix: `journal = {arXiv preprint arXiv:XXXX.XXXXX}` (supply the ID).

22. [MINOR] Table 1 caption | "Baseline avg." — informal abbreviation. Use "Baseline average BER."

### Style Patterns to Fix Throughout

[MAJOR] **British vs. American English**: Global fix: `linearise(d)` → `linearize(d)`, `serialise` → `serialize`, `neighbourhood(s)` → `neighborhood(s)`, `labelled` → `labeled`. Run a document-wide search-and-replace.

[MAJOR] **Evaluative filler**: Search for `interesting`, `notably`, `notable`, `striking`, `crucially`, `importantly` and delete or replace each with a factual statement.

[MINOR] **Sentence fragments in experimental setup**: Convert telegraphic "Backbone: …" and "Baselines: …" lines to full sentences or a proper itemized list.

[MINOR] **Passive voice in results**: Scan results and discussion for "is guaranteed," "are expected," "is not a trivial guarantee" and convert to active constructions where an agent (GRASP-RAG, the algorithm) is clearly identifiable.

---

## Priority Action Items

### CRITICAL (must fix — these will cause rejection or unacceptable misrepresentation)

1. **[A6/A3] Report classifier accuracy and VDB anchor recovery rates** for the 120 bridge queries. The 100% BER headline is uninterpretable without this breakdown. (Agent 6 Required #1; Agent 3 Missing Ablation #12)

2. **[A3/A6] Remove the unqualified 100% BER claim from Abstract and Conclusion** — or add the same circularity qualifier that currently appears only in Discussion. The abstract says "100% BER on every domain" with no qualification; Section 5 correctly says this is "near-inevitable under correct classification." The two sections contradict each other in framing.

3. **[A3] Add ablation: bridge motif + flat context vs. bridge motif + PGR** — without this, the two claimed innovations (motif selection and PGR) cannot be disentangled. (Agent 3 #13; Agent 6 Suggested #2)

4. **[A6] Add quantitative evaluation of at least one additional motif** — the paper claims a "complete four-motif system" but evaluates only bridge. Even a single-entity vs. LightRAG-local comparison would prevent reviewer suspicion. (Agent 6 Required #5)

5. **[A3/A6] Report fallback frequency (Equation 1)** — is the 0.04 threshold ever triggered? If so, what is the BER of fallback queries, and do they enter the 120-question denominator? (Agent 6 Required #4)

6. **[A2/A1] Fix gap range: "22.5–42.5 pp" → "22.5–45.0 pp"** — a factual numerical error (Agriculture vs. NaiveRAG = 45.0 pp). (Agent 2 Critical #1; Agent 1 Major #10)

7. **[A4] Resolve the dual use of N** — `\mathcal{N}` means "anchor node set" on line 251 and "neighborhood function N(n_1)" on line 255. The same symbol serves two distinct mathematical roles within three lines. (Agent 4 Notation #2; Agent 1 Critical #4)

8. **[A4] Unify A vs. N for anchor nodes** — Section 3.2 uses `\mathcal{A}`; Section 3.3 uses `\mathcal{N}` for the same set without explanation. (Agent 4 Notation #1; Agent 1 Critical #3)

9. **[A6] Add per-query cost/call-count comparison** — GRASP-RAG makes ≥2 LLM calls per query versus 1 for baselines. This computational cost differential is unacknowledged and must be disclosed. (Agent 6 Required #2)

10. **[A5/A2] Figure 4 is never cited in the main text** — add "Figure~\ref{fig:ber_chart}" in Section IV-C. (Agent 5 Critical; Agent 2 Cross-Reference)

11. **[A1] Fix all British spellings** — global replace: `linearised/linearises` → `linearized/linearizes`, `serialise` → `serialize`, `neighbourhood(s)` → `neighborhood(s)`, `labelled` → `labeled`. (Agent 1 Critical #1, Major #6/#14/#15)

12. **[A6] Remove uncited bibliography entries** — `thirunavukarasu2023large`, `zakka2024almanac`, and `gao2023retrieval` are in references.bib but never cited. Remove before submission to avoid referee confusion about scope. (Agent 6 Part 4; Agent 2 Cross-Reference)

### MAJOR (should fix — will likely be raised by referees)

13. **[A6] Add citations to Adaptive-RAG, Self-RAG, and multi-hop QA literature** (HotpotQA, MuSiQue, 2WikiMultihopQA) — the paper's core contribution is query routing; the absence of this literature will be immediately noticed.

14. **[A4] Define E_S, V_S before Equation 1** — currently used without prior definition.

15. **[A4] Define cutoff=3, k, and m** in bridge path and PPR motif definitions.

16. **[A4] Clarify bridge path motif** — ShortestPath ∪ SimplePaths is redundant (ShortestPath ⊆ SimplePaths); specify what "computed bidirectionally" means for directed vs. undirected G.

17. **[A5] Table 2 column header "GRASP" → "GRASP-RAG"** everywhere for consistency with title and figure legend.

18. **[A5] Add confidence interval note to Table 2** — N=40 per cell; 95% CI spans ±7–16 pp for interior values; a reviewer will ask.

19. **[A3] Hedge PathRAG Legal failure as hypothesis** — "we hypothesize that flow-based pruning removes low-frequency bridge edges" rather than stating it as established fact.

20. **[A3] Hedge "not domain-specific" → "suggesting domain robustness on the bridge task"** — three domains is insufficient to claim general domain independence.

21. **[A5/A2] Fix sec:classify never cross-referenced** — add `(Section~\ref{sec:classify})` in introduction contribution point, or remove the dangling label.

22. **[A1] Delete all evaluative filler** — "interesting," "notably," "striking," "notable" — all violate scientific style.

23. **[A1] Fix baseline ordering inconsistency** — Abstract, Table 2, and Section 4.3 list baselines in three different orders. Standardize to the Table 2 column order (PathRAG, LightRAG, NaiveRAG) everywhere.

24. **[A5] Confirm tcolorbox Table 3 is IEEEtran-compatible** — if the proceedings editor strips tcolorbox, the qualitative example will vanish. Consider a standard tabular layout or confirm with the template.

25. **[A3] Add "To our knowledge" scope qualifier** to the implicit novelty claim for query-adaptive motif selection.

### MINOR (polish — improves paper quality)

26. [A1] Introduction line 91: "across entities" → "among entities."
27. [A1] Introduction line 110: "it never classified" → "because PathRAG never classifies."
28. [A1] Section 4.1 lines 343–346: Convert sentence fragments to full sentences.
29. [A1] Section 4.1 line 339: Remove "consistent results are expected" from CS omission note.
30. [A2] Standardize terminology: "query type" (not "query shape"), "retrieval motif" (not "graph motif"), "bridge entity" (not "intermediate entity"), spell out "VDB" on first use.
31. [A4] Add `(Eq.~\ref{eq:conf})` at the fallback threshold line; demote or remove the label if the equation is not to be referenced.
32. [A4] Define ℓ in Steiner approximation ratio: "where ℓ = |N| is the number of terminal nodes."
33. [A4] Rename PPR parameter m to avoid collision with method-index m in BER formula.
34. [A5] Table 3 caption: name the domain ("Agriculture domain"); clarify bold-entity vs. bold-label distinction.
35. [A5] Figure 1 caption: expand to explain all diagram elements (Q, VDB, PGR Builder, LLM→a); note that broad motif bypasses full PGR.
36. [A5] Figure 2 caption: add panel order "(a) Ego-network, (b) Bridge path, (c) Steiner tree, (d) PPR diffusion."
37. [A5] Figure 3 caption: name the specific query shown and specify "PathRAG and LightRAG" for "flat retrieval."
38. [A5] Figure 4 caption: disclose truncated y-axis (starts at 40%); add N=40 per bar.
39. [A2] Fill in `\res{...}` author/institution placeholders before compiling final submission PDF.
40. [A1] References: supply arXiv ID in `ding2024ultradomain` journal field.

---

*Report generated 2026-05-14 by 6-agent pre-submission review. Preliminary recommendation: **Major Revision**.*
