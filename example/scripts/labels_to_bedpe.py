#!/usr/bin/env python3
"""Split loops into per-cluster .bedpe files from any LoopBin labels.npy.

Row i of <sweep>/02_process/loop_file_analysis.bedpe is row i of labels.npy (the same
pasting `loopbin cluster` does for labels_loops.bedpe), so this works for a seed run
directory as well as for consensus/ and consensus_numselect/.

Usage: labels_to_bedpe.py <labels_dir> [loops.bedpe] [min_conf]
  <labels_dir>  dir holding labels.npy (+ optional loop_confidence.npy)
  [loops.bedpe] default <labels_dir>/../02_process/loop_file_analysis.bedpe
  [min_conf]    drop loops whose consensus confidence is below this (default 0 = keep all)

Writes  <labels_dir>/bedpe/cluster_<k>.bedpe   (+ all_loops_labeled.bedpe, low_confidence.bedpe)
"""
import os, sys, collections
import numpy as np

LD = os.path.abspath(sys.argv[1])
LOOPS = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("", "-") else os.path.join(os.path.dirname(LD), "02_process", "loop_file_analysis.bedpe")
MINC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

lab = np.load(os.path.join(LD, "labels.npy"))
cpath = os.path.join(LD, "loop_confidence.npy")
conf = np.load(cpath) if os.path.exists(cpath) else None

with open(LOOPS) as f:
    rows = [l.rstrip("\n") for l in f if l.strip()]
if len(rows) != len(lab):
    sys.exit(f"ERROR: {len(rows)} loops in {LOOPS} but {len(lab)} labels in {LD}/labels.npy")

OUT = os.path.join(LD, "bedpe"); os.makedirs(OUT, exist_ok=True)
hdr = "#chr1\tstart1\tend1\tchr2\tstart2\tend2\tcluster" + ("\tconfidence" if conf is not None else "")

def line(i):
    s = f"{rows[i]}\t{int(lab[i])}"
    return s + (f"\t{conf[i]:.3f}" if conf is not None else "")

keep = np.ones(len(lab), bool) if conf is None or MINC <= 0 else (conf >= MINC)
with open(os.path.join(OUT, "all_loops_labeled.bedpe"), "w") as f:
    f.write(hdr + "\n")
    for i in range(len(lab)):
        if keep[i]: f.write(line(i) + "\n")
if conf is not None and MINC > 0:
    with open(os.path.join(OUT, "low_confidence.bedpe"), "w") as f:
        f.write(hdr + "\n")
        for i in np.where(~keep)[0]: f.write(line(int(i)) + "\n")

n = len(lab)
print(f"[labels_to_bedpe] {LD}\n  loops={n}  clusters={sorted(set(lab.tolist()))}"
      + (f"  min_conf={MINC} -> kept {int(keep.sum())} ({keep.mean()*100:.1f}%)" if MINC > 0 else ""))
for c in sorted(set(lab.tolist())):
    idx = [i for i in np.where(lab == c)[0] if keep[i]]
    p = os.path.join(OUT, f"cluster_{c}.bedpe")
    with open(p, "w") as f:
        f.write(hdr + "\n")
        for i in idx: f.write(line(int(i)) + "\n")
    msg = f"  cluster_{c}.bedpe : {len(idx):>6} loops ({len(idx)/n*100:5.1f}%)"
    if conf is not None and len(idx):
        msg += f"   mean conf={conf[idx].mean():.3f}"
    print(msg)
print("  ->", OUT)
