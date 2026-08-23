"""
Generate bridge_graph.jpg — knowledge graph showing 4 bridge paths
from Phosphorus to Beekeeping Industry.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(_ROOT, "data", "report", "bridge_graph.jpg")

# ── Graph definition ──────────────────────────────────────────────────────────
G = nx.DiGraph()

# Internal node IDs (short) → display labels (may wrap)
node_labels = {
    "Phosphorus":    "Phosphorus",
    "Crops":         "Crops",
    "Farmers":       "Farmers",
    "BeeIndustry":   "Beekeeping\nIndustry",
    "Soil":          "Soil",
    "Agriculture":   "Agriculture",
    "MycoFungi":     "Mycorrhizal\nFungi",
    "SoilHealth":    "Soil Health",
    "China":         "China",
    "Chloramp":      "Chloramph.",
    "Honey":         "Honey",
}

nodes = list(node_labels.keys())

edges_raw = [
    # Path 1: Phosphorus → Crops → Farmers → BeeIndustry
    ("Phosphorus", "Crops",       "promotes growth"),
    ("Crops",      "Farmers",     "sustains livelihood"),
    ("Farmers",    "BeeIndustry", "supports"),
    # Path 2: Phosphorus → Soil → Agriculture → BeeIndustry
    ("Phosphorus", "Soil",        "enriches"),
    ("Soil",       "Agriculture", "supports"),
    ("Agriculture","BeeIndustry", "sustains ecosystem"),
    # Path 3: Phosphorus → MycoFungi → SoilHealth → BeeIndustry
    ("Phosphorus", "MycoFungi",   "feeds"),
    ("MycoFungi",  "SoilHealth",  "improves"),
    ("SoilHealth", "BeeIndustry", "enables forage"),
    # Path 4: Phosphorus → China → Chloramp → Honey → BeeIndustry
    ("Phosphorus", "China",       "imports"),
    ("China",      "Chloramp",    "produces"),
    ("Chloramp",   "Honey",       "contaminates"),
    ("Honey",      "BeeIndustry", "produced by"),
]

edges = edges_raw   # keep alias for edge_labels below

for n in nodes:
    G.add_node(n)
for src, dst, lbl in edges_raw:
    G.add_edge(src, dst, label=lbl)

# ── Layout (manual positions) ─────────────────────────────────────────────────
pos = {
    "Phosphorus":  (-5.2,  0.0),
    # Path 1
    "Crops":       (-2.4,  2.3),
    "Farmers":     ( 0.6,  2.3),
    # Path 2
    "Soil":        (-2.4,  0.75),
    "Agriculture": ( 0.6,  0.75),
    # Path 3
    "MycoFungi":   (-2.4, -0.75),
    "SoilHealth":  ( 0.6, -0.75),
    # Path 4
    "China":       (-2.4, -2.3),
    "Chloramp":    ( 0.2, -2.3),
    "Honey":       ( 2.4, -2.3),
    # Anchor right — pulled in so label fits
    "BeeIndustry": ( 4.0,  0.0),
}

# ── Colors ────────────────────────────────────────────────────────────────────
anchor_ids = {"Phosphorus", "BeeIndustry"}
node_colors = ["#3a7ebf" if n in anchor_ids else "#b0b8c1" for n in G.nodes()]
node_sizes  = [3400 if n in anchor_ids else 2200 for n in G.nodes()]

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 9))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=node_colors, node_size=node_sizes, alpha=0.92)

# Use human-readable wrapped labels
nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
    font_size=8.0, font_color="white", font_weight="bold")

nx.draw_networkx_edges(G, pos, ax=ax,
    edge_color="#6c757d", arrows=True,
    arrowstyle="-|>", arrowsize=18,
    connectionstyle="arc3,rad=0.08",
    width=1.6, alpha=0.75, min_source_margin=22, min_target_margin=22)

edge_labels = {(s, d): lbl for s, d, lbl in edges_raw}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
    font_size=6.8, font_color="#343a40",
    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
    label_pos=0.42)

# Explicit axis limits with generous padding so no labels are clipped
ax.set_xlim(-6.8, 5.6)
ax.set_ylim(-3.2, 3.2)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#3a7ebf", label="Anchor entity"),
    mpatches.Patch(color="#b0b8c1", label="Intermediate entity"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
          framealpha=0.85, edgecolor="#ced4da")

ax.set_title(
    'Bridge Paths: "Phosphorus" → "Beekeeping Industry"',
    fontsize=13, fontweight="bold", pad=12, color="#212529"
)
ax.axis("off")

# Keep figure margins fixed — never use bbox_inches='tight', it re-clips to content
plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)

plt.savefig(OUT, dpi=150, format="jpeg", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {OUT}")
