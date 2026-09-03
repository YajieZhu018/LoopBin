#!/usr/bin/env python3
"""Silhouette plots of the CONSENSUS clustering (co-association distance), for given k values.
Each loop = a horizontal bar of its silhouette; grouped+sorted by cluster; red line = mean.
Wide positive bars = loops that reliably co-cluster across seeds; bars near 0/negative = ambiguous
continuum loops. Uses the same subsample as the k-selection so values match k_selection.pdf.

Usage: silhouette_plot.py <sweep_dir> <seed_list_csv> <k_csv> [out_subdir] [n_sub]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

SW = os.path.abspath(sys.argv[1])
SEEDS = [s for s in sys.argv[2].split(",") if s]
KS = [int(x) for x in sys.argv[3].split(",")]
OUT = os.path.join(SW, sys.argv[4] if len(sys.argv) > 4 else "consensus_numselect"); os.makedirs(OUT, exist_ok=True)
N_SUB = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
rng = np.random.RandomState(0)

def agglo(D, k):
    try:
        return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
    except TypeError:
        return AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average").fit_predict(D)

L = {s: np.load(os.path.join(SW, f"seed{s}", "labels.npy")) for s in SEEDS}
n = len(next(iter(L.values())))
sub = rng.choice(n, size=min(N_SUB, n), replace=False)
C = np.zeros((len(sub), len(sub)), dtype=np.float32)
for s in SEEDS:
    lab = L[s][sub]
    C += (lab[:, None] == lab[None, :]).astype(np.float32)
C /= len(SEEDS)
D = (1.0 - C).astype(np.float64)

fig, axes = plt.subplots(1, len(KS), figsize=(5.5 * len(KS), 6), squeeze=False)
for ax, k in zip(axes[0], KS):
    lab = agglo(D, k)
    sil = silhouette_samples(D, lab, metric="precomputed")
    avg = silhouette_score(D, lab, metric="precomputed")
    # order clusters by size (largest first) for a tidy plot
    order = sorted(np.unique(lab), key=lambda c: -(lab == c).sum())
    y = 10
    for i, c in enumerate(order):
        v = np.sort(sil[lab == c])
        col = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, facecolor=col, edgecolor=col, alpha=0.8)
        ax.text(-0.05, y + len(v) / 2, f"c{i} ({(lab==c).mean()*100:.0f}%)", va="center", ha="right", fontsize=8)
        y += len(v) + 10
    ax.axvline(avg, color="red", ls="--", lw=1)
    ax.text(avg, y, f" mean={avg:.3f}", color="red", fontsize=9, va="bottom")
    ax.set_title(f"consensus silhouette — k={k}", fontsize=11)
    ax.set_xlabel("silhouette value (per loop)"); ax.set_yticks([])
    ax.set_xlim(min(-0.2, sil.min() - 0.05), 1.0)
    ax.axvline(0, color="grey", lw=0.6)
fig.suptitle(f"Per-loop consensus silhouette (subsample n={len(sub)})  —  "
             f"wide positive = reliable; near 0/negative = ambiguous (continuum boundary)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "silhouette_plot.pdf")
fig.savefig(out, dpi=150); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out, "(+ .png)  k-list:", KS, " means:",
      {k: round(silhouette_score(D, agglo(D, k), metric="precomputed"), 3) for k in KS})
