#!/usr/bin/env python3
"""t-SNE of the loop feature matrix, coloured by CONSENSUS cluster.

Deliberately embeds the INPUT features, not a VaDE latent space: consensus labels come from
agreement across independent runs, so embedding the data keeps the picture from being a
rendering of one model own decision boundary. Point opacity = per-loop consensus confidence,
so ambiguous loops fade rather than masquerading as confident members.

Three perplexities are drawn because t-SNE apparent cluster separation is perplexity dependent
and a single setting is not interpretable on its own.

Usage: plot_consensus_tsne.py <run_dir> [consensus_subdir] [n_sub]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RUN = os.path.abspath(sys.argv[1])
SUB = sys.argv[2] if len(sys.argv) > 2 else "consensus_numselect"
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
PERP = [30, 100, 200]
OUT = os.path.join(RUN, SUB)

X = np.load(os.path.join(RUN, "02_process", "merged_log_data.npy"))
lab = np.load(os.path.join(OUT, "labels.npy"))
conf = np.load(os.path.join(OUT, "loop_confidence.npy"))
assert len(X) == len(lab) == len(conf), (X.shape, lab.shape, conf.shape)
print(f"[tsne] n={len(X)} d={X.shape[1]} k={len(np.unique(lab))} sub={min(NSUB,len(X))}")

rng = np.random.RandomState(0)
idx = rng.choice(len(X), size=min(NSUB, len(X)), replace=False)
Xs, ls, cs = X[idx], lab[idx], conf[idx]

# PCA first: 352 -> 50 dims. Standard practice, and it makes t-SNE tractable on a loaded box.
P = PCA(n_components=50, random_state=0).fit_transform(Xs)
print(f"[tsne] PCA 50 comps, explained variance = {PCA(n_components=50, random_state=0).fit(Xs).explained_variance_ratio_.sum():.3f}")

fig, axes = plt.subplots(1, len(PERP), figsize=(5.2 * len(PERP), 5), constrained_layout=True)
cmap = plt.get_cmap("tab10")
for ax, p in zip(np.atleast_1d(axes), PERP):
    print(f"[tsne] perplexity={p} ...", flush=True)
    E = TSNE(n_components=2, init="pca", random_state=0, learning_rate="auto",
             perplexity=p, n_jobs=2).fit_transform(P)
    for c in np.unique(ls):
        m = ls == c
        ax.scatter(E[m, 0], E[m, 1], s=4, linewidths=0, color=cmap(c % 10),
                   alpha=np.clip(cs[m], 0.15, 1.0), label=f"c{c} ({m.sum()/len(ls)*100:.0f}%)")
    ax.set_title(f"perplexity = {p}")
    ax.set_xticks([]); ax.set_yticks([])
    np.savetxt(os.path.join(OUT, f"tsne_consensus_perplex{p}.csv"),
               np.column_stack([E, ls, cs]), delimiter=",", comments="",
               header="component_1,component_2,consensus_label,confidence")
np.atleast_1d(axes)[0].legend(markerscale=3, fontsize=8, loc="best", frameon=False)
fig.suptitle(f"t-SNE of loop features, coloured by consensus cluster "
             f"(n={len(idx)} of {len(X)}; opacity = confidence)", fontsize=11)
fig.savefig(os.path.join(OUT, "tsne_consensus.pdf"), dpi=150)
fig.savefig(os.path.join(OUT, "tsne_consensus.png"), dpi=150)
print(f"[tsne] DONE -> {OUT}/tsne_consensus.pdf")
