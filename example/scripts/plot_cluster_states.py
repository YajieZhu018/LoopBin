#!/usr/bin/env python3
"""Chromatin-state composition of each loop cluster's anchor set.

Joins a labels.npy to the per-anchor state assignment from anchor_chrom_states.py and, for every
cluster, counts the state GROUPS (Prom, Enh, Quies, ...) over the cluster's ANCHOR SET — an anchor
locus used by several loops of the same cluster counts once.

Two assignment rules are available (both are columns of anchor_states.tsv):
  state (default) — the anchor takes the single 100-state class with the largest overlap, and that
                    state's group is plotted. This is the rule as specified.
  group           — the group with the largest total overlap, summing over its states. More stable
                    (a 5 kb anchor spans ~25 segments); use it as a robustness check.

Usage: plot_cluster_states.py <labels_dir> <chromstates_ref_dir> [rule] [anchor_dir] [out_subdir]
  <labels_dir>           dir holding labels.npy (consensus_numselect/, consensus/, seedN/)
  <chromstates_ref_dir>  annotation dir: references/mm10_chromstates/ (100-state full-stack) or
                         references/mESC_chromHMM_wei12/ (12-state mESC, Hsieh Fig 3E)
  [rule]                 state (default) | group  — identical when each state is its own group
  [anchor_dir]           default <labels_dir>/../chrom_states  (anchor_states.tsv, anchor_loop_map.tsv)
  [out_subdir]           default chrom_states[_grouprule]

Writes <labels_dir>/<out_subdir>/ :
  cluster_state_counts.tsv  states_histogram.pdf/.png  states_pie.pdf/.png
  states_enrichment_log2.pdf/.png  chrom_states_summary.txt
"""
import os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LD = os.path.abspath(sys.argv[1])
REF = os.path.abspath(sys.argv[2])
RULE = sys.argv[3] if len(sys.argv) > 3 else "state"
AD = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else os.path.join(os.path.dirname(LD), "chrom_states")
if RULE not in ("state", "group"):
    sys.exit("ERROR: rule must be 'state' or 'group', got %r" % RULE)
DEFOUT = "chrom_states" if RULE == "state" else "chrom_states_grouprule"
OUT = os.path.join(LD, sys.argv[5] if len(sys.argv) > 5 else DEFOUT)
os.makedirs(OUT, exist_ok=True)


def pick(*names):
    for n in names:
        p = os.path.join(REF, n)
        if os.path.exists(p):
            return p
    return os.path.join(REF, names[0])


ANN = pick("annotation.tsv", "state_annotation_processed.tsv")
BGG = os.path.join(REF, "genome_5kb_window_groups.tsv")
AST = os.path.join(AD, "anchor_states.tsv")
MAP = os.path.join(AD, "anchor_loop_map.tsv")
for p in (ANN, BGG, AST, MAP):
    if not os.path.exists(p):
        sys.exit("ERROR: missing %s (run 0_download_chrom_states.sh / anchor_chrom_states.py first)" % p)

GROUP_COL = "group" if RULE == "state" else "group_maxov"
BG_COL = "n_windows_state_rule" if RULE == "state" else "n_windows_group_rule"


def read_tsv(path):
    """Minimal TSV reader -> (header list, list of row dicts). Skips '#' comment lines."""
    rows, hdr = [], None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = p
            continue
        rows.append(dict(zip(hdr, p)))
    return hdr, rows


# ---------------------------------------------- group order + colours -------
_, ann = read_tsv(ANN)
order_of, color_votes = {}, collections.defaultdict(collections.Counter)
for r in ann:
    g = r["group"].strip()
    o = int(r["state_order_by_group"]) if r.get("state_order_by_group", "").strip() else 0
    order_of[g] = min(order_of.get(g, 10 ** 9), o)
    color_votes[g][r["color"].strip()] += 1
GCOLOR = {g: c.most_common(1)[0][0] for g, c in color_votes.items()}
mixed = [g for g, c in color_votes.items() if len(c) > 1]
GCOLOR["NoState"] = "#d3d3d3"
order_of["NoState"] = 10 ** 9

# ---------------------------------------------------- anchors and labels ----
_, anchors = read_tsv(AST)
STATE_OF = {int(r["locus_id"]): r[GROUP_COL] for r in anchors}

_, mrows = read_tsv(MAP)
loop_loci = collections.defaultdict(list)
for r in mrows:
    loop_loci[int(r["loop_idx"])].append(int(r["locus_id"]))

lab = np.load(os.path.join(LD, "labels.npy"))
if len(lab) != len(loop_loci):
    sys.exit("ERROR: %d labels but %d loops in %s" % (len(lab), len(loop_loci), MAP))
clusters = sorted(set(lab.tolist()))

# per-cluster ANCHOR SET (deduplicated within the cluster)
sets = {c: set() for c in clusters}
for i, c in enumerate(lab.tolist()):
    sets[c].update(loop_loci[i])
counts = {c: collections.Counter(STATE_OF[l] for l in sets[c]) for c in clusters}

# ----------------------------------------------------------- background -----
_, bgrows = read_tsv(BGG)
bg = {r["group"]: int(r[BG_COL]) for r in bgrows}
bg_tot = sum(bg.values())

groups = sorted({g for c in clusters for g in counts[c]} | set(bg),
                key=lambda g: (order_of.get(g, 10 ** 9), g))
# Keep every state the annotation defines, even at 0 % — the axis is the model's vocabulary and
# must read the same way as the source figure. Only synthetic categories (NoState, added when an
# anchor overlaps nothing) are dropped when empty.
ANN_GROUPS = {r["group"].strip() for r in ann}
groups = [g for g in groups
          if g in ANN_GROUPS or any(counts[c].get(g, 0) for c in clusters)]

# Optional: drop null/background states and renormalise, e.g. LOOPBIN_STATES_EXCLUDE=Intergenic
# reproduces the eleven states Hsieh et al. 2020 Fig 3E displays (the 12-state model minus the
# Intergenic null state, which is 76 % of the genome and would swamp every pie).
EXCL = [g.strip() for g in os.environ.get("LOOPBIN_STATES_EXCLUDE", "").split(",") if g.strip()]
dropped = {}
n_before = {c: len(sets[c]) for c in clusters}
if EXCL:
    dropped = {g: sum(counts[c].get(g, 0) for c in clusters) for g in EXCL if g in groups}
    groups = [g for g in groups if g not in EXCL]
    for c in clusters:
        sets[c] = {l for l in sets[c] if STATE_OF[l] not in EXCL}
        counts[c] = collections.Counter({g: v for g, v in counts[c].items() if g not in EXCL})
    bg = {g: v for g, v in bg.items() if g not in EXCL}
    bg_tot = sum(bg.values())
colors = [GCOLOR.get(g, "#d3d3d3") for g in groups]

pct = {c: {g: counts[c].get(g, 0) / max(len(sets[c]), 1) * 100 for g in groups} for c in clusters}
bgpct = {g: bg.get(g, 0) / bg_tot * 100 for g in groups}
EPS = 0.5 / bg_tot * 100          # half-window pseudo-count, keeps log2 finite
log2e = {c: {g: float(np.log2((pct[c][g] + EPS) / (bgpct[g] + EPS))) for g in groups} for c in clusters}

# ------------------------------------------------------------- tables -------
with open(os.path.join(OUT, "cluster_state_counts.tsv"), "w") as f:
    f.write("# assignment rule: %s (column %s of anchor_states.tsv); background: %s of %s\n"
            % (RULE, GROUP_COL, BG_COL, os.path.basename(BGG)))
    f.write("cluster\tgroup\tn_anchors\tpct_anchors\tbackground_pct\tlog2_obs_exp\n")
    for c in clusters:
        for g in groups:
            f.write("%d\t%s\t%d\t%.3f\t%.3f\t%.3f\n"
                    % (c, g, counts[c].get(g, 0), pct[c][g], bgpct[g], log2e[c][g]))

shared = collections.Counter()
for c in clusters:
    for l in sets[c]:
        shared[l] += 1
n_multi = sum(1 for v in shared.values() if v > 1)

lines = ["chromatin-state composition of cluster anchor sets  (rule=%s)" % RULE,
         "labels: %s" % os.path.join(LD, "labels.npy"),
         "anchors: %s" % AST,
         "background: %s [%s], %d windows" % (BGG, BG_COL, bg_tot)]
if EXCL:
    lines.append("EXCLUDED and renormalised: %s  (anchors removed per cluster: %s)"
                 % (", ".join(EXCL), ", ".join("%s=%d" % (g, n) for g, n in dropped.items())))
lines += ["",
          "cluster  n_loops  n_anchor_loci  (anchor slots = 2 x n_loops before de-duplication)"]
for c in clusters:
    lines.append("  c%-6d %-8d %d" % (c, int((lab == c).sum()), len(sets[c])))
lines += ["",
          "distinct loci overall: %d;  used by more than one cluster: %d (%.1f%%)"
          % (len(shared), n_multi, n_multi / len(shared) * 100),
          ""]
for c in clusters:
    top = sorted(groups, key=lambda g: -pct[c][g])[:3]
    enr = [g for g in sorted(groups, key=lambda g: -log2e[c][g]) if counts[c].get(g, 0) >= 10][:3]
    dep = [g for g in sorted(groups, key=lambda g: log2e[c][g]) if bgpct[g] >= 1.0][:3]
    lines.append("c%d  most abundant: %s" % (c, ", ".join("%s %.1f%%" % (g, pct[c][g]) for g in top)))
    lines.append("    enriched:      %s" % ", ".join("%s %+.2f" % (g, log2e[c][g]) for g in enr))
    lines.append("    lowest log2:   %s" % ", ".join("%s %+.2f" % (g, log2e[c][g]) for g in dep))
if mixed:
    lines.append("")
    lines.append("note: groups with more than one state colour (modal colour used): %s" % ", ".join(mixed))
open(os.path.join(OUT, "chrom_states_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


x = np.arange(len(groups))
k = len(clusters)
ANNOT_NAME = os.path.basename(REF.rstrip("/"))
SUB = (" — excluding %s, renormalised" % ", ".join(EXCL)) if EXCL else ""


def panel_title(c):
    """Show what the exclusion cost this cluster: keeping the denominator visible stops the
    renormalised composition from being read as if it covered the whole cluster."""
    if EXCL:
        return "cluster %d   (%d loops, %d of %d anchor loci = %.0f%%)" % (
            c, int((lab == c).sum()), len(sets[c]), n_before[c],
            len(sets[c]) / max(n_before[c], 1) * 100)
    return "cluster %d   (%d loops, %d anchor loci)" % (c, int((lab == c).sum()), len(sets[c]))

# ------------------------------------------------- 1. raw composition -------
ymax = max(pct[c][g] for c in clusters for g in groups) * 1.15
fig, axes = plt.subplots(k, 1, figsize=(9, 2.1 * k + 0.6), sharex=True)
axes = np.atleast_1d(axes)
for ax, c in zip(axes, clusters):
    ax.bar(x, [pct[c][g] for g in groups], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("% anchors")
    ax.set_title(panel_title(c), fontsize=10, loc="left")
    ax.grid(axis="y", ls=":", alpha=0.4)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
fig.suptitle("Chromatin-state composition of loop-cluster anchors (%s, rule=%s)%s"
             % (ANNOT_NAME, RULE, SUB), fontsize=11, y=0.995)
fig.tight_layout()
save(fig, "states_histogram")

# ------------------------------------------------------------ 2. pies -------
ncol = min(3, k)
nrow = int(np.ceil(k / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
axes = np.atleast_1d(axes).ravel()
# Merge sub-1 % slices only when the vocabulary is large enough to need it; with a dozen states
# every slice is shown so the pie carries the same categories, in the same order, as the bars.
MERGE = 1.0 if len(groups) > 14 else 0.0
for ax, c in zip(axes, clusters):
    keep = [g for g in groups if pct[c][g] >= MERGE]
    vals = [pct[c][g] for g in keep]
    cols = [GCOLOR.get(g, "#d3d3d3") for g in keep]
    rest = 100 - sum(vals)
    if rest > 1e-9:
        keep, vals, cols = keep + ["other <%g%%" % MERGE], vals + [rest], cols + ["#bbbbbb"]
    # dark wedge edges: several full-stack groups are white or near-white (quiescent) and
    # would otherwise vanish against the page
    ax.pie(vals, colors=cols, startangle=90, counterclock=False,
           autopct=lambda v: ("%.0f%%" % v) if v >= 5 else "",
           wedgeprops={"edgecolor": "#444444", "linewidth": 0.5}, textprops={"fontsize": 8})
    ax.set_title(panel_title(c).replace("   ", "\n"), fontsize=9)
for ax in axes[k:]:
    ax.axis("off")
handles = [Patch(facecolor=GCOLOR.get(g, "#d3d3d3"), edgecolor="black", linewidth=0.4, label=g)
           for g in groups]
if MERGE > 0:
    handles.append(Patch(facecolor="#bbbbbb", edgecolor="black", linewidth=0.4,
                         label="other <%g%%" % MERGE))
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Anchor chromatin-state composition per cluster (%s, rule=%s)%s"
             % (ANNOT_NAME, RULE, SUB), fontsize=11)
fig.tight_layout()
save(fig, "states_pie")

# ------------------------------------------------------ 3. enrichment -------
lim = max(abs(log2e[c][g]) for c in clusters for g in groups) * 1.15
fig, axes = plt.subplots(k, 1, figsize=(9, 2.1 * k + 0.6), sharex=True)
axes = np.atleast_1d(axes)
for ax, c in zip(axes, clusters):
    ax.bar(x, [log2e[c][g] for g in groups], color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("log2 obs/exp")
    ax.set_title("cluster %d" % c, fontsize=10, loc="left")
    ax.grid(axis="y", ls=":", alpha=0.4)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
fig.suptitle("Enrichment vs matched 5 kb-window background (%s, rule=%s)%s"
             % (ANNOT_NAME, RULE, SUB), fontsize=11, y=0.995)
fig.tight_layout()
save(fig, "states_enrichment_log2")

print("\n-> %s" % OUT)
