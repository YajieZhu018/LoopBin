#!/usr/bin/env python3
"""Consensus clustering WITH data-driven k selection (no retraining; scales to large n).
1) Co-association on a subsample -> scan k, pick k* by silhouette/PAC (answers 'is num=7 right?').
2) Pick a medoid per consensus cluster; assign ALL loops to the medoid they most co-cluster with.
3) Emit consensus labels + per-loop confidence + full LoopBin figures.

Usage: consensus_numselect.py <sweep_dir> <marks_csv> <seed_list_csv> [out_subdir] [n_sub] [kmin] [kmax]
"""
import os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score as ARI
from loopbin.fn import function, processing
from loopbin.plot import plotting

SW = os.path.abspath(sys.argv[1])
MARKS = sys.argv[2].split(",")
SEEDS = [s for s in sys.argv[3].split(",") if s]
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else "consensus_numselect"
N_SUB = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
KMIN = int(sys.argv[6]) if len(sys.argv) > 6 else 2
KMAX = int(sys.argv[7]) if len(sys.argv) > 7 else 12
OUT = os.path.join(SW, OUTNAME); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(SW, "02_process", "merged_log_data.npy")
rng = np.random.RandomState(0)

def agglo(D, k):
    try:
        return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
    except TypeError:
        return AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average").fit_predict(D)

# --- load aligned seed labels ----------------------------------------------
L = {s: np.load(os.path.join(SW, f"seed{s}", "labels.npy")) for s in SEEDS}
n = len(next(iter(L.values())))
assert all(len(v) == n for v in L.values())
ks = {s: int(len(np.unique(L[s]))) for s in SEEDS}
print(f"[numselect] {SW}\n  n_loops={n}  seeds={SEEDS}  per-seed k={ks}")

# --- co-association on a SUBSAMPLE for k-selection --------------------------
sub = rng.choice(n, size=min(N_SUB, n), replace=False)
Csub = np.zeros((len(sub), len(sub)), dtype=np.float32)
for s in SEEDS:
    lab = L[s][sub]
    Csub += (lab[:, None] == lab[None, :]).astype(np.float32)
Csub /= len(SEEDS)
Dsub = (1.0 - Csub).astype(np.float64)

print(f"  k-selection on {len(sub)} subsampled loops:")
print(f"  {'k':>3} {'silhouette':>11} {'PAC(0.1-0.9)':>13} {'sizes%':>8}")
scores = {}
for k in range(KMIN, KMAX + 1):
    lab = agglo(Dsub, k)
    try:
        sil = silhouette_score(Dsub, lab, metric="precomputed")
    except Exception:
        sil = float("nan")
    iu = np.triu_indices(len(sub), 1)
    cv = Csub[iu]
    pac = float(((cv > 0.1) & (cv < 0.9)).mean())     # lower = cleaner consensus
    sizes = sorted((np.bincount(lab) / len(sub) * 100).round(1), reverse=True)
    scores[k] = (sil, pac)
    print(f"  {k:>3} {sil:>11.3f} {pac:>13.3f}   {sizes}")

# pick k* = max silhouette (PAC as tiebreak: prefer lower)
kstar = max(scores, key=lambda k: (round(scores[k][0], 3), -scores[k][1]))
print(f"  >> selected k* = {kstar}  (silhouette={scores[kstar][0]:.3f}, PAC={scores[kstar][1]:.3f})  [num used in training was 7]")
_fk = os.environ.get("LOOPBIN_FORCE_K")
if _fk:
    kstar = int(_fk)
    print(f"  >> FORCED k = {kstar} (LOOPBIN_FORCE_K) overriding auto-select")

# plot the selection curves
ksrange = list(range(KMIN, KMAX + 1))
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(ksrange, [scores[k][0] for k in ksrange], "o-", color="C0", label="silhouette")
ax1.axvline(kstar, ls="--", color="grey"); ax1.axvline(7, ls=":", color="red", label="num=7 (training)")
ax1.set_xlabel("k (consensus cut)"); ax1.set_ylabel("silhouette (consensus)", color="C0")
ax2 = ax1.twinx(); ax2.plot(ksrange, [scores[k][1] for k in ksrange], "s-", color="C1")
ax2.set_ylabel("PAC (lower=cleaner)", color="C1")
ax1.set_title(f"Data-driven k: selected k*={kstar}"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "k_selection.pdf"), dpi=150)

# --- full-data consensus at k* via medoid assignment (scales to all n) ------
lab_sub = agglo(Dsub, kstar)
medoids = []
for c in np.unique(lab_sub):
    idx = np.where(lab_sub == c)[0]
    within = Csub[np.ix_(idx, idx)].mean(1)
    medoids.append(sub[idx[np.argmax(within)]])         # global index of the medoid loop
# co-association of every loop to each medoid = mean over seeds of co-membership
coa = np.zeros((n, len(medoids)), dtype=np.float32)
for j, m in enumerate(medoids):
    for s in SEEDS:
        coa[:, j] += (L[s] == L[s][m]).astype(np.float32)
coa /= len(SEEDS)
cons = np.argmax(coa, axis=1)
conf = coa[np.arange(n), cons]                          # per-loop confidence
# optional min-size merge: drop medoids of clusters < frac, reassign their loops to nearest kept medoid
mf = float(os.environ.get("LOOPBIN_MERGE_MIN_FRAC", "0"))
if mf > 0:
    keep = [j for j in range(len(medoids)) if (cons == j).mean() >= mf]
    if 0 < len(keep) < len(medoids):
        coa = coa[:, keep]
        cons = np.argmax(coa, axis=1)
        conf = coa[np.arange(n), cons]
        print(f"  >> min-size merge < {mf*100:.1f}%: kept {len(keep)}/{len(medoids)} substantial clusters")
# relabel by size (0=largest)
order = [c for c, _ in collections.Counter(cons.tolist()).most_common()]
remap = {c: i for i, c in enumerate(order)}
cons = np.array([remap[c] for c in cons])
np.save(os.path.join(OUT, "labels.npy"), cons)
np.save(os.path.join(OUT, "loop_confidence.npy"), conf)

sizes = collections.Counter(cons.tolist())
import itertools
seed_aris = [ARI(L[a], L[b]) for a, b in itertools.combinations(SEEDS, 2)]
cons_aris = {s: ARI(L[s], cons) for s in SEEDS}
summary = [
    f"selected k* = {kstar}  (training num was 7; per-seed realized k = {ks})",
    "consensus cluster sizes %: " + "  ".join(f"c{c}:{sizes[c]/n*100:.1f}" for c in sorted(sizes)),
    f"per-loop confidence: mean={conf.mean():.3f}  >=0.8:{(conf>=0.8).mean()*100:.1f}%  "
    f">=0.6:{(conf>=0.6).mean()*100:.1f}%  <0.5:{(conf<0.5).mean()*100:.1f}%",
    f"baseline pairwise seed ARI: mean={np.mean(seed_aris):.3f}",
    "each seed vs CONSENSUS ARI: " + "  ".join(f"s{s}:{cons_aris[s]:.3f}" for s in SEEDS)
    + f"  (mean={np.mean(list(cons_aris.values())):.3f})",
]
print("\n".join(summary)); open(os.path.join(OUT, "consensus_summary.txt"), "w").write("\n".join(summary) + "\n")

# --- full LoopBin figures on the consensus labels --------------------------
data = np.load(DATA)
x_data = processing.create_data(data[:, 256:], data[:, :256])
dico = function.sep_cluster(x_data, cons)
plotting.plot_all_clusters(dico, cons, OUT, MARKS)
plotting.plot_pie(dico, cons, OUT)
print("[numselect] DONE ->", OUT)
