#!/usr/bin/env python3
"""Assign one ChromHMM full-stack state to every loop ANCHOR by largest overlap.

A loop anchor is one bin wide (5 kb here); a full-stack segment is a run of 200 bp bins, so an
anchor typically spans several segments. The anchor takes the state covering the most base pairs
of it; ties break on the lowest state number so the result is deterministic.

The assignment is done ONCE over the distinct anchor loci of a run (an anchor shared by several
loops is one locus), and is independent of any clustering — plot_cluster_states.py joins it to
whichever labels.npy you point it at.

Also builds, once per annotation, a MATCHED BACKGROUND: every 5 kb window of chr1-19,chrX put
through the same max-overlap rule, so enrichment compares like with like.

Usage: anchor_chrom_states.py <sweep_dir> <chromstates_dir> [chrom_sizes] [out_subdir]
  <sweep_dir>        run dir holding 02_process/loop_file_analysis.bedpe
  <chromstates_dir>  an annotation dir holding a sorted segmentation BED + a state TSV:
                       references/mm10_chromstates/     (100-state full-stack)
                       references/mESC_chromHMM_wei12/  (12-state mESC, Hsieh Fig 3E)
  [chrom_sizes]      default <chromstates_dir>/../mm10.main.chrom.sizes
  [out_subdir]       default "chrom_states"

Writes  <sweep_dir>/<out_subdir>/anchor_states.tsv     one row per distinct anchor locus
        <sweep_dir>/<out_subdir>/anchor_loop_map.tsv   loop_idx, anchor(1|2) -> locus_id
        <chromstates_dir>/genome_5kb_window_{states,groups}.tsv  background (built once)
"""
import os, sys, subprocess, collections

SW = os.path.abspath(sys.argv[1])
CS = os.path.abspath(sys.argv[2])
SIZES = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(CS), "mm10.main.chrom.sizes")
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else "chrom_states"


def pick(*names):
    """First existing file among <chromstates_dir>/<name> — lets one script serve several
    annotations without renaming their files."""
    for n in names:
        p = os.path.join(CS, n)
        if os.path.exists(p):
            return p
    return os.path.join(CS, names[0])


SEG = pick("segments.sorted.bed", "mm10_100_segments.main.sorted.bed")
ANN = pick("annotation.tsv", "state_annotation_processed.tsv")
LOOPS = os.path.join(SW, "02_process", "loop_file_analysis.bedpe")
OUT = os.path.join(SW, OUTNAME); os.makedirs(OUT, exist_ok=True)
BG = os.path.join(CS, "genome_5kb_window_states.tsv")
BGG = os.path.join(CS, "genome_5kb_window_groups.tsv")
for p in (SEG, ANN, LOOPS, SIZES):
    if not os.path.exists(p):
        sys.exit("ERROR: missing " + p)


# ------------------------------------------------------- state annotation --
def load_annotation(path):
    """MNEMONIC -> (group, colour, order).

    Keyed by mnemonic on purpose. The BED segment names are '<chromhmm_id>_<mnemonic>'
    (e.g. '98_mTSS1'), and the TSV's own `state` column is a DIFFERENT numbering — TSV state 98
    is mEnhA17, not mTSS1 — so joining on the number silently mislabels every anchor. The
    mnemonic is the only key both files agree on.
    """
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(hdr)}
        need = ["group", "color", "mneumonics"]
        miss = [n for n in need if n not in col]
        if miss:
            sys.exit("ERROR: %s lacks column(s) %s; has %s" % (path, miss, hdr))
        ordc = col.get("state_order_by_group")
        ann = {}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr) or not p[col["mneumonics"]].strip():
                continue
            order = int(p[ordc]) if ordc is not None and p[ordc].strip() else 0
            ann[p[col["mneumonics"]].strip()] = (p[col["group"]].strip(),
                                                 p[col["color"]].strip(), order)
    return ann


ANNOT = load_annotation(ANN)
print("[anchor_states] %d states, %d groups" % (len(ANNOT), len(set(v[0] for v in ANNOT.values()))))


def split_name(label):
    """'98_mTSS1' -> (98, 'mTSS1'). The number is the ChromHMM state id used in the BED."""
    if "_" not in label:
        sys.exit("ERROR: unexpected segment name %r (want '<id>_<mnemonic>')" % label)
    num, mnem = label.split("_", 1)
    return int(num), mnem


# every mnemonic in the segmentation must exist in the annotation, or the join is wrong
seg_names = subprocess.run("cut -f4 " + SEG + " | LC_ALL=C sort -u", shell=True,
                           capture_output=True, text=True).stdout.split()
SEG_ID, unknown = {}, []
for nm in seg_names:
    i, mn = split_name(nm)
    SEG_ID[nm] = i
    if mn not in ANNOT:
        unknown.append(nm)
if unknown:
    sys.exit("ERROR: %d segment states are not in %s: %s" % (len(unknown), ANN, unknown[:5]))
print("  %d distinct segment states in the BED, all matched to the annotation by mnemonic"
      % len(seg_names))


# ------------------------------------------------------------- bedtools ----
def genome_file(path):
    """chrom-sizes reordered to the LC_ALL=C order of the segmentation, for `intersect -sorted`."""
    size = {}
    for line in open(SIZES):
        p = line.split()
        if len(p) >= 2:
            size[p[0]] = p[1]
    order = subprocess.run("cut -f1 " + SEG + " | uniq", shell=True,
                           capture_output=True, text=True).stdout.split()
    with open(path, "w") as f:
        for c in order:
            if c in size:
                f.write("%s\t%s\n" % (c, size[c]))
    return order


def max_overlap(bed_path, genome, expect):
    """Stream `bedtools intersect -wao` and keep, per -a record, the winner by overlapped bp.

    TWO winners are computed from the same overlaps:
      * state-level  — the single 100-state segment class covering the most bp (what was asked);
      * group-level  — the state GROUP (Prom, Enh, Quies, ...) covering the most bp, summing over
        all its states. A 5 kb anchor spans ~25 segments, so the top state usually holds only a
        third of it; aggregating first is far more stable, and the group is what gets plotted.
    Ties break on the lowest state number / alphabetically first group, so runs are deterministic.

    bedtools emits all hits of one -a record consecutively, so a running group-by is safe.
    Returns {name: (segname, state_bp, n_states, state_tie, group, group_bp, n_groups, group_tie)}.
    """
    cmd = ["bedtools", "intersect", "-a", bed_path, "-b", SEG, "-wao", "-sorted", "-g", genome]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    res = {}
    state = {"cur": None, "per": collections.Counter()}

    def flush():
        cur, per = state["cur"], state["per"]
        if cur is None:
            return
        if not per:
            res[cur] = (None, 0, 0, False, None, 0, 0, False)
            return
        top = max(per.values())
        winners = sorted((s for s, v in per.items() if v == top), key=lambda s: SEG_ID[s])
        gper = collections.Counter()
        for s, v in per.items():
            gper[ANNOT[s.split("_", 1)[1]][0]] += v
        gtop = max(gper.values())
        gwin = sorted(g for g, v in gper.items() if v == gtop)
        res[cur] = (winners[0], top, len(per), len(winners) > 1,
                    gwin[0], gtop, len(gper), len(gwin) > 1)

    for line in pr.stdout:
        p = line.rstrip("\n").split("\t")
        name, sname, ov = p[3], p[7], int(p[8])
        if name != state["cur"]:
            flush()
            state["cur"], state["per"] = name, collections.Counter()
        if ov > 0 and sname != ".":
            state["per"][sname] += ov
    flush()
    pr.stdout.close()
    if pr.wait() != 0:
        sys.exit("ERROR: bedtools intersect failed")
    if len(res) != expect:
        sys.exit("ERROR: bedtools returned %d records, expected %d" % (len(res), expect))
    return res


GEN = os.path.join(OUT, "_genome_order.txt")
genome_file(GEN)

# ------------------------------------------------------ anchors -> loci -----
loops = [l.rstrip("\n").split("\t") for l in open(LOOPS) if l.strip()]
loci, pairs = {}, []          # (chrom,start,end) -> locus_id ; (loop_idx, which, locus_id)
for i, p in enumerate(loops):
    for which, (c, s, e) in enumerate(((p[0], p[1], p[2]), (p[3], p[4], p[5])), start=1):
        key = (c, int(s), int(e))
        if key not in loci:
            loci[key] = len(loci)
        pairs.append((i, which, loci[key]))
keys = sorted(loci, key=lambda k: (k[0], k[1]))          # LC_ALL=C order == python str order
widths = collections.Counter(e - s for _, s, e in keys)
print("  loops=%d  anchor slots=%d  distinct anchor loci=%d  widths=%s"
      % (len(loops), len(pairs), len(loci), dict(widths)))

ANCH = os.path.join(OUT, "_anchors.bed")
with open(ANCH, "w") as f:
    for k in keys:
        f.write("%s\t%d\t%d\t%d\n" % (k[0], k[1], k[2], loci[k]))

hits = max_overlap(ANCH, GEN, len(loci))

GROUP_COLOR = {}
for _mn, (_gr, _col, _o) in ANNOT.items():
    GROUP_COLOR.setdefault(_gr, collections.Counter())[_col] += 1
GROUP_COLOR = {g: c.most_common(1)[0][0] for g, c in GROUP_COLOR.items()}

unassigned = ties = gties = agree = 0
fracs, gfracs = [], []
with open(os.path.join(OUT, "anchor_states.tsv"), "w") as f:
    f.write("chrom\tstart\tend\tlocus_id\tstate\tmnemonic\tgroup\tcolor\t"
            "n_states_overlapped\ttop_overlap_bp\ttop_overlap_frac\ttie\t"
            "group_maxov\tgroup_maxov_color\tn_groups_overlapped\tgroup_maxov_bp\t"
            "group_maxov_frac\tgroup_tie\tgroup_agrees\n")
    for k in keys:
        lid = loci[k]
        sn, bp, ns, tie, gr2, gbp, ng, gtie = hits[str(lid)]
        width = k[2] - k[1]
        if sn is None:
            unassigned += 1
            f.write("%s\t%d\t%d\t%d\tNA\tNA\tNoState\t#d3d3d3\t0\t0\t0.000\t0\t"
                    "NoState\t#d3d3d3\t0\t0\t0.000\t0\t1\n" % (k[0], k[1], k[2], lid))
            continue
        st, mn = split_name(sn)
        gr, col, _ = ANNOT[mn]
        ties += int(tie)
        gties += int(gtie)
        same = int(gr == gr2)
        agree += same
        fracs.append(bp / width)
        gfracs.append(gbp / width)
        f.write("%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\t%d\t%.3f\t%d\t%s\t%s\t%d\t%d\t%.3f\t%d\t%d\n"
                % (k[0], k[1], k[2], lid, st, mn, gr, col, ns, bp, bp / width, int(tie),
                   gr2, GROUP_COLOR.get(gr2, "#d3d3d3"), ng, gbp, gbp / width, int(gtie), same))

with open(os.path.join(OUT, "anchor_loop_map.tsv"), "w") as f:
    f.write("loop_idx\tanchor\tlocus_id\n")
    for i, w, lid in pairs:
        f.write("%d\t%d\t%d\n" % (i, w, lid))

def quart(v, label):
    v = sorted(v)
    g = lambda x: v[min(int(x * len(v)), len(v) - 1)]
    print("  %s fraction of the anchor: min=%.2f q25=%.2f median=%.2f q75=%.2f max=%.2f"
          % (label, v[0], g(.25), g(.5), g(.75), v[-1]))

n_ok = len(fracs)
print("  unassigned anchors=%d" % unassigned)
print("  STATE rule : ties broken by lowest state=%d (%.1f%%)" % (ties, ties / n_ok * 100))
quart(fracs, "winning-state ")
print("  GROUP rule : ties=%d (%.1f%%)" % (gties, gties / n_ok * 100))
quart(gfracs, "winning-group ")
print("  the two rules give the same group for %d/%d anchors (%.1f%%)"
      % (agree, n_ok, agree / n_ok * 100))
print("  -> %s/anchor_states.tsv  (+ anchor_loop_map.tsv)" % OUT)

# --------------------------------------------------- matched background -----
if os.path.exists(BG):
    print("  background already built: " + BG)
else:
    print("  building matched 5 kb-window background (once per annotation) ...")
    win_w = widths.most_common(1)[0][0]
    WIN = os.path.join(OUT, "_windows.bed")
    subprocess.run("bedtools makewindows -g %s -w %d | LC_ALL=C sort -k1,1 -k2,2n "
                   "| awk -v OFS='\\t' '{print $1,$2,$3,NR-1}' > %s" % (GEN, win_w, WIN),
                   shell=True, check=True)
    nwin = sum(1 for _ in open(WIN))
    bg = max_overlap(WIN, GEN, nwin)
    cnt = collections.Counter(v[0] for v in bg.values())
    with open(BG, "w") as f:
        f.write("# matched background: every %d bp window of the main chromosomes, "
                "same max-overlap rule\n" % win_w)
        f.write("state\tmnemonic\tgroup\tn_windows\n")
        for sn, n in sorted(cnt.items(), key=lambda x: (x[0] is None, SEG_ID.get(x[0], 0))):
            if sn is None:
                f.write("NA\tNA\tNoState\t%d\n" % n)
            else:
                i, mn = split_name(sn)
                f.write("%d\t%s\t%s\t%d\n" % (i, mn, ANNOT[mn][0], n))
    # group-level background, both rules, so the enrichment panel matches whichever
    # assignment the plots use
    by_state_rule = collections.Counter(ANNOT[v[0].split("_", 1)[1]][0] if v[0] is not None
                                        else "NoState" for v in bg.values())
    by_group_rule = collections.Counter(v[4] if v[4] is not None else "NoState"
                                        for v in bg.values())
    with open(BGG, "w") as f:
        f.write("# matched background over %d windows of %d bp, main chromosomes\n" % (nwin, win_w))
        f.write("group\tn_windows_state_rule\tn_windows_group_rule\n")
        for g in sorted(set(by_state_rule) | set(by_group_rule)):
            f.write("%s\t%d\t%d\n" % (g, by_state_rule[g], by_group_rule[g]))
    os.remove(WIN)
    print("  background: %d windows of %d bp -> %s\n              %s" % (nwin, win_w, BG, BGG))

for p in (ANCH, GEN):
    if os.path.exists(p):
        os.remove(p)
