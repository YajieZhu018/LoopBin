"""
preprocess.py — bigWig -> per-chromosome bedgraph (pure-Python orchestration).

Replaces the three shell scripts (src/preprocess_local.sh, src/fn/preprocessing.sh,
src/fn/empty_preprocess.sh). The genome is NOT hardcoded: chromosome names + sizes come
from a UCSC-style chrom.sizes file (--chrom-sizes), so the tool runs on any assembly.

Per chromosome it tiles [0, n*res) into <resolution>-bp bins, runs `bigWigAverageOverBed`
(kept — same numeric output as before; the 5th column 'mean0' = mean with no-data-as-0 is
what the bedgraph stores), and writes <chrom>_<NAME><res/1000>K.bedgraph. Chromosomes run in
parallel (multiprocessing). target == "empty" writes a zero track (old empty_preprocess.sh).
"""
import os
import subprocess
import multiprocessing as mp


def read_chrom_sizes(path):
    """Parse a UCSC chrom.sizes file (2 columns: name<TAB>size) -> list of (name, size)."""
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"--chrom-sizes file not found: {path}")
    sizes = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                sizes.append((parts[0], int(parts[1])))
    if not sizes:
        raise ValueError(f"no chromosomes parsed from {path}")
    return sizes


def _suffix(resolution):
    return f"{resolution // 1000}K"


def _one_chrom(task):
    """Tile one chromosome and extract its bedgraph (or a zero track if target=='empty')."""
    bigwig, name, out_dir, resolution, chrom, size = task
    suf = _suffix(resolution)
    bedgraph = os.path.join(out_dir, f"{chrom}_{name}{suf}.bedgraph")
    n = size // resolution
    if n < 1:
        return None

    # empty mode: write a zero track for the same tiling (old empty_preprocess.sh)
    if bigwig == "empty":
        start = 0
        with open(bedgraph, "w") as g:
            for _ in range(n):
                g.write(f"{chrom}\t{start}\t{start + resolution}\t0\n")
                start += resolution
        return bedgraph

    bed = os.path.join(out_dir, f"{chrom}_{name}{suf}.bed")
    tmp = os.path.join(out_dir, f"{chrom}{name}temp.txt")
    # bed: bins [0,res),[res,2res),...,[(n-1)res,n*res)  named <chrom>_<c>
    start = 0
    with open(bed, "w") as b:
        for c in range(1, n + 1):
            b.write(f"{chrom}\t{start}\t{start + resolution}\t{chrom}_{c}\n")
            start += resolution

    try:
        subprocess.run(["bigWigAverageOverBed", bigwig, bed, tmp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.g. chromosome absent from the bigWig — warn and skip rather than abort the whole run
        print(f"[preprocess] WARNING: bigWigAverageOverBed failed for {chrom} ({name}); skipping. "
              f"{e.stderr.decode(errors='ignore').strip()[:120]}")
        for p in (bed, tmp):
            if os.path.exists(p):
                os.remove(p)
        return None

    # join: bedgraph row = chrom start end mean0   (mean0 = column 5, index 4)
    mean0 = {}
    with open(tmp) as t:
        for line in t:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                mean0[p[0]] = p[4]
    with open(bed) as b, open(bedgraph, "w") as g:
        for line in b:
            c1, s, e, nm = line.rstrip("\n").split("\t")
            if nm in mean0:
                g.write(f"{c1}\t{s}\t{e}\t{mean0[nm]}\n")
    os.remove(tmp)
    os.remove(bed)
    return bedgraph


def run(bigwig, name, out_dir, resolution, chrom_sizes, nproc=8):
    """Preprocess one bigWig into per-chromosome bedgraphs for every chrom in chrom_sizes."""
    resolution = int(resolution)
    nproc = int(nproc) if nproc else 8
    os.makedirs(out_dir, exist_ok=True)
    sizes = read_chrom_sizes(chrom_sizes)
    tasks = [(bigwig, name, out_dir, resolution, chrom, size) for chrom, size in sizes]
    with mp.Pool(min(nproc, len(tasks))) as pool:
        outs = [o for o in pool.map(_one_chrom, tasks) if o]
    print(f"[preprocess] {name}: wrote {len(outs)}/{len(tasks)} bedgraphs to {out_dir} "
          f"(resolution={resolution}, suffix={_suffix(resolution)})")
    return outs
