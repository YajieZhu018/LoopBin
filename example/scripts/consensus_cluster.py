#!/usr/bin/env python3
"""Consensus clustering over a LoopBin seed sweep (post-hoc; no retraining).
Co-association across seeds -> one stable partition + per-loop confidence, then full LoopBin-style
output (all_clusters.pdf, pie.pdf, labels, recoloured t-SNE, stats). No size prior imposed.

Usage: consensus_cluster.py <seedsweep_dir> <marks_csv>
  builds <seedsweep_dir>/consensus/ from seed1..seed5/labels.npy
"""
import os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from loopbin.fn import function, processing
from loopbin.plot import plotting

SW = os.path.abspath(sys.argv[1])
MARKS = sys.argv[2].split(",")
SEEDS = [1, 2, 3, 4, 5]
OUT = os.path.join(SW, "consensus"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(SW, "02_process", "merged_log_data.npy")

# --- load aligned seed labels ----------------------------------------------
L = {s: np.load(os.path.join(SW, f"seed{s}", "labels.npy")) for s in SEEDS}
n = len(next(iter(L.values())))
assert all(len(v) == n for v in L.values()), "seed label vectors differ in length"
ks = {s: len(np.unique(L[s])) for s in SEEDS}
print(f"[consensus] {SW}\n  n_loops={n}  per-seed k={ks}")

# --- co-association matrix C[i,j] = fraction of seeds where i,j co-cluster --
C = np.zeros((n, n), dtype=np.float32)
for s in SEEDS:
    lab = L[s]
    C += (lab[:, None] == lab[None, :]).astype(np.float32)
C /= len(SEEDS)

# --- consensus partition: agglomerative (average linkage) on distance 1-C ---
kmode = collections.Counter(ks.values()).most_common(1)[0][0]   # modal per-seed k
D = (1.0 - C).astype(np.float64)
try:
    cons = AgglomerativeClustering(n_clusters=kmode, metric="precomputed", linkage="average").fit_predict(D)
except TypeError:
    cons = AgglomerativeClustering(n_clusters=kmode, affinity="precomputed", linkage="average").fit_predict(D)
# relabel consensus clusters by size (0 = largest) for readable output
order = [c for c, _ in collections.Counter(cons).most_common()]
remap = {c: i for i, c in enumerate(order)}
cons = np.array([remap[c] for c in cons])
np.save(os.path.join(OUT, "labels.npy"), cons)

# --- per-loop confidence = mean co-association with own consensus-cluster ---
conf = np.zeros(n)
for c in np.unique(cons):
    idx = np.where(cons == c)[0]
    sub = C[np.ix_(idx, idx)]
    conf[idx] = (sub.sum(1) - 1.0) / max(len(idx) - 1, 1)   # exclude self
np.save(os.path.join(OUT, "loop_confidence.npy"), conf)

# --- stats ------------------------------------------------------------------
sizes = collections.Counter(cons.tolist())
lines = []
lines.append(f"consensus k = {kmode} (modal per-seed k); per-seed k = {ks}")
lines.append("consensus cluster sizes %: " + "  ".join(
    f"c{c}:{sizes[c]/n*100:.1f}" for c in sorted(sizes)))
lines.append(f"per-loop confidence: mean={conf.mean():.3f}  "
             f">=0.8:{(conf>=0.8).mean()*100:.1f}%  >=0.6:{(conf>=0.6).mean()*100:.1f}%  "
             f"<0.5:{(conf<0.5).mean()*100:.1f}%")
import itertools
seed_aris = [ARI(L[a], L[b]) for a, b in itertools.combinations(SEEDS, 2)]
lines.append(f"baseline pairwise seed ARI: mean={np.mean(seed_aris):.3f} "
             f"range=[{min(seed_aris):.3f},{max(seed_aris):.3f}]")
cons_aris = {s: ARI(L[s], cons) for s in SEEDS}
lines.append("each seed vs CONSENSUS ARI: " + "  ".join(f"s{s}:{cons_aris[s]:.3f}" for s in SEEDS)
             + f"  (mean={np.mean(list(cons_aris.values())):.3f})")
summary = "\n".join(lines)
print(summary)
open(os.path.join(OUT, "consensus_summary.txt"), "w").write(summary + "\n")

# --- full LoopBin-style figures on the consensus labels --------------------
data = np.load(DATA)
micro, epi = data[:, :256], data[:, 256:]
x_data = processing.create_data(epi, micro)
dict_ori = function.sep_cluster(x_data, cons)
plotting.plot_all_clusters(dict_ori, cons, OUT, MARKS)
plotting.plot_pie(dict_ori, cons, OUT)
print("[consensus] wrote all_clusters.pdf + pie.pdf")

# --- recoloured t-SNE: reuse a reference seed's latent coords, colour by consensus
try:
    import pandas as pd
    ref = os.path.join(SW, "seed1", "tsne_100.csv")
    df = pd.read_csv(ref)
    xy = df.select_dtypes("number").values[:, :2]
    if len(xy) == n:
        plt.figure(figsize=(6, 5))
        for c in np.unique(cons):
            m = cons == c
            plt.scatter(xy[m, 0], xy[m, 1], s=2, label=f"c{c}")
        plt.legend(markerscale=4, fontsize=8); plt.title("consensus on seed1 t-SNE latent")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "tsne_consensus.pdf"), dpi=150)
        print("[consensus] wrote tsne_consensus.pdf")
except Exception as e:
    print("[consensus] tsne recolour skipped:", e)
print("[consensus] DONE ->", OUT)
