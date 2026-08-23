"""
Generate compost_graph.jpg — Steiner-tree subgraph connecting
Compost, Biochar, and Soil Erosion via Organic Matter and Tobacco.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(_ROOT, "data", "report", "compost_graph.jpg")

# ── Node labels ───────────────────────────────────────────────────────────────
node_labels = {
    "Compost":       "Compost",
    "Biochar":       "Biochar",
    "SoilErosion":   "Soil\nErosion",
    "OrganicMatter": "Organic\nMatter",
    "Tobacco":       "Tobacco",
}

# ── Edges (from Steiner-tree paths) ───────────────────────────────────────────
edges_raw = [
    # Path 1: Compost → Organic Matter → Biochar
    ("Compost",       "OrganicMatter", "agriculture,\nnutrient enhancement"),
    ("OrganicMatter", "Biochar",       "ecological impact,\nsoil fertility"),
    # Path 2: Compost → Tobacco → Soil Erosion
    ("Compost",       "Tobacco",       "farming method,\nsoil improvement"),
    ("Tobacco",       "SoilErosion",   "crop impact,\nenvironmental degradation"),
    # Path 3 reuses same edges (Biochar → OrganicMatter → Compost → Tobacco → SoilErosion)
    # Already covered above — add reverse direction for Biochar side
    ("Biochar",       "OrganicMatter", "ecological impact,\nsoil fertility"),
]

G = nx.DiGraph()
for n in node_labels:
    G.add_node(n)
for src, dst, lbl in edges_raw:
    if not G.has_edge(src, dst):
        G.add_edge(src, dst, label=lbl)

# ── Layout ────────────────────────────────────────────────────────────────────
# Triangle of anchors, intermediates inside
pos = {
    "Compost":       (-3.0,  2.0),
    "Biochar":       (-3.0, -2.0),
    "SoilErosion":   ( 3.5,  0.0),
    "OrganicMatter": ( 0.0,  0.5),
    "Tobacco":       ( 0.8, -1.8),
}

# ── Colors ────────────────────────────────────────────────────────────────────
anchors = {"Compost", "Biochar", "SoilErosion"}
node_colors = ["#3a7ebf" if n in anchors else "#b0b8c1" for n in G.nodes()]
node_sizes  = [3200 if n in anchors else 2400 for n in G.nodes()]

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=node_colors, node_size=node_sizes, alpha=0.92)

nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
    font_size=9.0, font_color="white", font_weight="bold")

nx.draw_networkx_edges(G, pos, ax=ax,
    edge_color="#6c757d", arrows=True,
    arrowstyle="-|>", arrowsize=20,
    connectionstyle="arc3,rad=0.12",
    width=1.8, alpha=0.78,
    min_source_margin=24, min_target_margin=24)

edge_labels = {(s, d): lbl for s, d, lbl in edges_raw if G.has_edge(s, d)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
    font_size=7.5, font_color="#343a40",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    label_pos=0.45)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#3a7ebf", label="Anchor entity (query concept)"),
    mpatches.Patch(color="#b0b8c1", label="Intermediate entity"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
          framealpha=0.85, edgecolor="#ced4da")

# ── Path annotation ───────────────────────────────────────────────────────────
paths_txt = (
    "Steiner-tree paths:  "
    "P1: Compost → Organic Matter → Biochar  |  "
    "P2: Compost → Tobacco → Soil Erosion  |  "
    "P3: Biochar → Organic Matter → Compost → Tobacco → Soil Erosion"
)
ax.text(0.5, -0.02, paths_txt, transform=ax.transAxes,
        ha="center", va="top", fontsize=6.5, color="#555", style="italic")

ax.set_title(
    'Steiner-Tree Subgraph: "Compost", "Biochar", "Soil Erosion"\n'
    '(GRASP-RAG multi_entity motif — 3 Steiner-tree paths found)',
    fontsize=12, fontweight="bold", pad=12, color="#212529"
)
ax.set_xlim(-5.0, 5.5)
ax.set_ylim(-3.2, 3.2)
ax.axis("off")

plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.07)
plt.savefig(OUT, dpi=150, format="jpeg", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
