#!/usr/bin/env python3
"""Generate confusion matrix figure for GRASP-RAG paper (SF Pro, matplotlib)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os, glob

# ── SF Pro font setup ─────────────────────────────────────────────────────────
def _find_sf_pro():
    """Return path to an SF Pro font file, or None if not found."""
    search = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SF-Pro.ttf",
    ]
    search += glob.glob("/System/Library/Fonts/SF*.ttf")
    search += glob.glob("/Library/Fonts/SF Pro*.otf")
    search += glob.glob("/Library/Fonts/SF Pro*.ttf")
    for p in search:
        if os.path.exists(p):
            return p
    return None

sf_path = _find_sf_pro()
if sf_path:
    fm.fontManager.addfont(sf_path)
    prop = fm.FontProperties(fname=sf_path)
    sf_name = prop.get_name()
    plt.rcParams["font.family"] = sf_name
    print(f"Using font: {sf_name}  ({sf_path})")
else:
    # Fallback to best available sans-serif
    plt.rcParams["font.family"] = ["SF Pro Text", "Helvetica Neue",
                                   "Arial", "sans-serif"]
    print("SF Pro not found; falling back to system sans-serif")

# ── Data: row-normalised % (GT rows × predicted cols) ────────────────────────
#  Order: Single Entity | Bridge | Multi Entity | Broad
#  Pooled across 4 domains (agriculture, cs, legal, health), 806 questions.
data = np.array([
    [54.7, 34.4, 10.2,  0.7],   # GT = Single Entity   (n=590)
    [ 0.9, 87.7, 11.4,  0.0],   # GT = Bridge          (n=114)
    [ 0.0,  0.0,100.0,  0.0],   # GT = Multi Entity    (n=9)
    [18.3, 18.3, 52.7, 10.8],   # GT = Broad           (n=93)
])

tick_labels = ["Single\nEntity", "Bridge", "Multi\nEntity", "Broad"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6.5))
fig.patch.set_facecolor("white")

cmap = plt.cm.Blues
im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="equal")

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Accuracy (%)", rotation=270, labelpad=20, fontsize=11)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_visible(False)

# ── Cell annotations ──────────────────────────────────────────────────────────
for i in range(4):
    for j in range(4):
        v = data[i, j]
        # colour + size based on cell intensity
        if v >= 70:
            fc, fs, fw = "white", 22, "bold"
        elif v >= 15:
            fc, fs, fw = "#2255aa", 14, "normal"
        elif v > 0:
            fc, fs, fw = "#6699cc", 13, "normal"
        else:
            fc, fs, fw = "#c0cdd8", 13, "normal"

        # format: no decimal for whole numbers, 1 dp otherwise
        label = f"{int(v)}" if v == int(v) else f"{v:.1f}"
        ax.text(j, i, label,
                ha="center", va="center",
                color=fc, fontsize=fs, fontweight=fw)

# ── Axes decoration ───────────────────────────────────────────────────────────
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(tick_labels, fontsize=11, ha="center")
ax.set_yticklabels(tick_labels, fontsize=11, va="center")
ax.set_xlabel("Predicted Type", fontsize=13, labelpad=12)
ax.set_ylabel("True Type", fontsize=13, labelpad=12)
ax.set_title(
    "Classifier Confusion Matrix\n"
    "(row-normalised %;  pooled across 4 domains,  806 questions)",
    fontsize=12, pad=14,
)

# white grid lines between cells
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", bottom=False, left=False)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
out  = os.path.join(here, "assets")
os.makedirs(out, exist_ok=True)

png_path = os.path.join(out, "confusion_matrix.png")
plt.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
print(f"Saved {png_path}")

# PDF: render text as paths so system fonts don't need embedding
plt.rcParams["pdf.fonttype"] = 3
pdf_path = os.path.join(out, "confusion_matrix.pdf")
try:
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {pdf_path}")
except Exception as e:
    print(f"PDF save failed ({e}); PNG is the primary output")
