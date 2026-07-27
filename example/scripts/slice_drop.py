#!/usr/bin/env python3
"""Slice out one mark's 32-col block from merged_log_data.npy. Prints remaining marks (orig order).
Usage: slice_drop.py <src_npy> <dst_npy> <drop_mark>   (marks order: CTCF,H3K27ac,H3K27me3,SMC1A)"""
import numpy as np, sys
src, dst, drop = sys.argv[1], sys.argv[2], sys.argv[3]
marks = ["CTCF", "H3K27ac", "H3K27me3", "SMC1A"]
d = np.load(src)
keep = [i for i, m in enumerate(marks) if m != drop]
cols = list(range(256)) + [256 + 32 * i + j for i in keep for j in range(32)]
np.save(dst, d[:, cols])
sys.stderr.write(f"[slice] drop {drop}: {d.shape} -> {d[:, cols].shape}\n")
print(",".join(m for m in marks if m != drop))
