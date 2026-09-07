"""Rebuild balanced_gmm_test_report.pdf from saved artifacts (no retraining, no TF).

Pages 2-5 are recomputed from latent.npz + candidate_metrics.json; page 1's loss
curve is lifted out of the previous PDF's own vector path, because the AE training
history is not persisted anywhere else.

The plotting bodies are copied from 04_test_real_pretrain_gmm.py (post-rasterize
patch); importing that module would drag in tensorflow, which is not needed here.
"""
import json
import os
import re
import sys
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

BASE = '/mnt/ceph-hdd/projects/SCC_UKLA_papantonis/Yajie/LoopBin'
SD = os.path.join(BASE, 'scripts', 'gmm_metric_validation')
METRIC_NAMES = ['min_weight', 'separation', 'cov_health', 'assignment_confidence', 'log_likelihood']
RANDOM_STATES = (0, 7, 13, 21, 42, 73, 101, 123, 777, 2024)


def extract_loss_curve(pdf_path):
    """Recover (epoch, loss) from page 1's polyline using its own axis tick labels."""
    d = open(pdf_path, 'rb').read()
    m = re.search(rb'(?<![0-9])9 0 obj\s*<<.*?>>\s*stream\r?\n', d, re.S)
    t = zlib.decompress(d[m.end():d.find(b'endstream', m.end())]).decode('latin1')

    tm = re.findall(r'q\s+[\d.-]+ [\d.-]+ [\d.-]+ [\d.-]+ ([\d.-]+) ([\d.-]+) cm\s*BT(.*?)ET', t, re.S)
    yticks = sorted((float(y), float(txt)) for _, y, blk in tm
                    for txt in [''.join(re.findall(r'\((.*?)\)', blk))]
                    if re.fullmatch(r'0\.\d+', txt))
    (ya, va), (yb, vb) = yticks[0], yticks[-1]
    sl = (vb - va) / (yb - ya)

    path = max(re.finditer(r'([\d.-]+) ([\d.-]+) m\s*((?:[\d.-]+ [\d.-]+ l\s*)+)', t),
               key=lambda x: len(x.group(3)))
    pts = [(float(path.group(1)), float(path.group(2)))]
    pts += [(float(a), float(b)) for a, b in re.findall(r'([\d.-]+) ([\d.-]+) l', path.group(3))]

    x0 = pts[0][0]
    dx = (pts[-1][0] - x0) / 199.0
    ep = [int(round((x - x0) / dx)) for x, _ in pts]
    val = [va + (y - ya) * sl for _, y in pts]
    keep = [i for i in range(len(ep)) if i == 0 or ep[i] != ep[i - 1]]
    return [ep[i] for i in keep], [val[i] for i in keep]


def plot_loss_page(pdf, epochs, values):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values)
    ax.set_title('AE pretraining loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_candidate_metrics_page(pdf, rows):
    random_states = [r['random_state'] for r in rows]
    fig, axes = plt.subplots(len(METRIC_NAMES), 1, figsize=(9, 12), sharex=True)
    for ax, metric in zip(axes, METRIC_NAMES):
        values = [r[metric] for r in rows]
        colors = ['#2a7f62' if r['is_balanced_selected'] else
                  '#c0392b' if r['is_naive_selected'] else '#4c72b0' for r in rows]
        ax.bar([str(rs) for rs in random_states], values, color=colors)
        ax.set_ylabel(metric)
    axes[-1].set_xlabel('random_state (green = balanced pick, red = naive best-log-likelihood pick)')
    fig.suptitle('Per-candidate diagnostic metrics on the real pretrained latent space')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_heatmap_page(pdf, matrix, random_states, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    n = len(random_states)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(random_states)
    ax.set_yticklabels(random_states)
    for i in range(n):
        for j in range(n):
            color = 'white' if matrix[i, j] < 0.6 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('random_state')
    ax.set_ylabel('random_state')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_pca_page(pdf, z, balanced_labels, naive_labels):
    coords = PCA(n_components=2).fit_transform(z)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, labels, title in zip(axes, [balanced_labels, naive_labels],
                                  ['balanced selection', 'naive best-log-likelihood']):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=6, alpha=0.7,
                             rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    fig.suptitle('Real latent space (PCA), colored by hard cluster assignment')
    fig.colorbar(scatter, ax=axes, shrink=0.7, label='cluster')
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def fit_labels(z, rs, n_centroid, covariance_type):
    """Same GaussianMixture parameters as vade_model._fit_gmm_candidates."""
    g = mixture.GaussianMixture(
        n_components=n_centroid, covariance_type=covariance_type,
        init_params='kmeans', n_init=1, reg_covar=1e-4, random_state=rs,
    )
    g.fit(z)
    return g.predict(z)


def regen(outdir):
    pdf_path = os.path.join(outdir, 'balanced_gmm_test_report.pdf')
    meta = json.load(open(os.path.join(outdir, 'candidate_metrics.json')))
    z = np.load(os.path.join(outdir, 'latent.npz'))['z']
    cfg = meta['config']

    epochs, values = extract_loss_curve(pdf_path)
    print(f'{outdir}\n  recovered {len(epochs)} loss points, {min(values):.4f}..{max(values):.4f}',
          flush=True)

    rows = meta['candidates']
    balanced_rs = next(r['random_state'] for r in rows if r['is_balanced_selected'])
    naive_rs = next(r['random_state'] for r in rows if r['is_naive_selected'])
    print(f'  refitting GMM random_state={balanced_rs} (balanced) and {naive_rs} (naive)', flush=True)

    bal = fit_labels(z, balanced_rs, cfg['num_clusters'], cfg['covariance_type'])
    nai = fit_labels(z, naive_rs, cfg['num_clusters'], cfg['covariance_type'])

    # the refit must reproduce the stored agreement number, or the labels are not
    # the same ones the original report plotted
    got = adjusted_rand_score(bal, nai)
    want = meta['summary']['ari_balanced_vs_naive']
    assert abs(got - want) < 1e-9, f'refit mismatch: {got} vs {want}'
    print(f'  refit check ok: ARI(balanced,naive)={got:.6f}', flush=True)

    ari = np.array(meta['pairwise_ari'])
    nmi = np.array(meta['pairwise_nmi'])
    rstates = meta['random_states']

    tmp = pdf_path + '.new'
    with PdfPages(tmp) as pdf:
        plot_loss_page(pdf, epochs, values)
        plot_candidate_metrics_page(pdf, rows)
        plot_heatmap_page(pdf, ari, rstates, 'Pairwise ARI across the 10 GMM candidates')
        plot_heatmap_page(pdf, nmi, rstates, 'Pairwise NMI across the 10 GMM candidates')
        plot_pca_page(pdf, z, bal, nai)
    print(f'  wrote {tmp} ({os.path.getsize(tmp)} bytes)', flush=True)


if __name__ == '__main__':
    for arm in sys.argv[1:]:
        regen(os.path.join(SD, 'results', 'real_latent_' + arm))
