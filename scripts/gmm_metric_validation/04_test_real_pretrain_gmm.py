### Confirms select_balanced_gmm's toy-data metric-selection result (see
### 03_plot_metric_selection.py) against a real pretrain run on real loop data. Real data has
### no ground-truth cluster labels, so ARI/NMI here are candidate-*agreement* metrics: pairwise
### across the 10 GMM candidates _fit_gmm_candidates fits on the real pretrained latent space,
### plus how much the balanced pick actually differs from sklearn's naive best-log-likelihood
### pick (the behavior select_balanced_gmm was written to replace).
import argparse
import csv
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'loopbin', 'model'))
sys.path.insert(0, SCRIPT_DIR)
from ae import AE  # noqa: E402
from vade_model import _fit_gmm_candidates, select_balanced_gmm  # noqa: E402
from latent_diagnostics import (  # noqa: E402
    compute_latent_diagnostics, format_table, CORRELATION_LEGEND)

import tensorflow as tf  # noqa: E402

METRIC_NAMES = ['min_weight', 'separation', 'cov_health', 'assignment_confidence', 'log_likelihood']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=os.path.join(REPO_ROOT, 'data', '02_process', 'merged_log_data.npy'))
    parser.add_argument('--n-samples', type=int, default=10000,
                         help='0 or negative = use the full dataset (what main.py:pretrain_ae does)')
    parser.add_argument('--num-clusters', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--latent-activation', default='relu',
                         help="AE bottleneck activation: 'relu' (production default, "
                              "src/model/ae.py) or 'linear'. See run_report.md's 2026-08-27 entry.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--covariance-type', default='diag')
    parser.add_argument('--metric-weights', default='1.0,1.0,1.0',
                         help='comma-separated (separation, assignment_confidence, log_likelihood) weights')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def subsample(data_path, n_samples, seed):
    X = np.load(data_path)
    if n_samples is None or n_samples <= 0 or n_samples >= X.shape[0]:
        return X  # full dataset, matching main.py:pretrain_ae
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    return X[idx]


def pretrain_ae(X, epochs, seed, latent_activation='relu'):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    d_input = X.shape[1]
    ae = AE(d_input, latent_activation=None if latent_activation == 'linear' else latent_activation)
    ae(np.zeros((10, d_input)))
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002))
    history = ae.fit(X, shuffle=True, batch_size=256, epochs=epochs)
    return ae, history


def build_candidate_table(candidates, balanced_random_state, naive_random_state):
    rows = []
    for c in candidates:
        rows.append({
            'random_state': c['random_state'],
            **{m: c[m] for m in METRIC_NAMES},
            'is_balanced_selected': c['random_state'] == balanced_random_state,
            'is_naive_selected': c['random_state'] == naive_random_state,
        })
    return rows


def pairwise_matrix(label_sets, score_fn):
    n = len(label_sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = score_fn(label_sets[i], label_sets[j])
    return mat


def plot_loss_page(pdf, history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history['loss'])
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
    # rasterize the point cloud: tens of thousands of vector markers make this
    # page unusably slow to render in PDF viewers
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    metric_weights = tuple(float(w) for w in args.metric_weights.split(','))

    X = subsample(args.data, args.n_samples, args.seed)
    np.save(os.path.join(args.output, 'subsampled_data.npy'), X)

    ae, history = pretrain_ae(X, args.epochs, args.seed, args.latent_activation)
    ae.save(os.path.join(args.output, 'pretrained_ae'))

    z = ae.encoder.predict(X)
    np.savez(os.path.join(args.output, 'latent.npz'), z=z)

    selected_gmm = select_balanced_gmm(z, args.num_clusters, covariance_type=args.covariance_type,
                                        metric_weights=metric_weights)
    candidates = _fit_gmm_candidates(z, args.num_clusters, covariance_type=args.covariance_type)

    # Latent health on the REAL pretrained space -- the question the toy harness
    # can only proxy. No ground truth here, so the within-cluster statistic is
    # grouped by the selected GMM's own hard assignments (see compute_latent_diagnostics'
    # docstring: that makes the within-cluster number optimistic, not pessimistic).
    diagnostics = compute_latent_diagnostics(z, selected_gmm.predict(z))
    diagnostics['latent_activation'] = args.latent_activation
    diagnostics['grouping'] = 'selected GMM hard assignments (no ground truth on real data)'
    with open(os.path.join(args.output, 'latent_diagnostics.json'), 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f'\n=== Latent-space health (latent_activation={args.latent_activation}) ===')
    print(CORRELATION_LEGEND)
    print(format_table([(os.path.basename(args.data), diagnostics)]))

    naive_best = max(candidates, key=lambda c: c['log_likelihood'])
    balanced_best = next(c for c in candidates if c['random_state'] == selected_gmm.random_state)

    for c in candidates:
        c['labels'] = c['gmm'].predict(z)
    label_sets = [c['labels'] for c in candidates]
    random_states = [c['random_state'] for c in candidates]

    ari_matrix = pairwise_matrix(label_sets, adjusted_rand_score)
    nmi_matrix = pairwise_matrix(label_sets, normalized_mutual_info_score)

    balanced_idx = random_states.index(balanced_best['random_state'])
    naive_idx = random_states.index(naive_best['random_state'])
    ari_balanced_vs_naive = float(ari_matrix[balanced_idx, naive_idx])
    mean_ari_balanced_to_others = float(np.mean(np.delete(ari_matrix[balanced_idx], balanced_idx)))
    mean_ari_naive_to_others = float(np.mean(np.delete(ari_matrix[naive_idx], naive_idx)))

    candidate_rows = build_candidate_table(candidates, balanced_best['random_state'],
                                            naive_best['random_state'])

    metrics_out = {
        'config': {
            'data': args.data, 'n_samples': int(X.shape[0]), 'num_clusters': args.num_clusters,
            'epochs': args.epochs, 'seed': args.seed, 'covariance_type': args.covariance_type,
            'metric_weights': metric_weights, 'latent_activation': args.latent_activation,
        },
        'latent_diagnostics': diagnostics,
        'candidates': candidate_rows,
        'pairwise_ari': ari_matrix.tolist(),
        'pairwise_nmi': nmi_matrix.tolist(),
        'random_states': random_states,
        'summary': {
            'ari_balanced_vs_naive': ari_balanced_vs_naive,
            'mean_ari_balanced_to_others': mean_ari_balanced_to_others,
            'mean_ari_naive_to_others': mean_ari_naive_to_others,
        },
    }
    with open(os.path.join(args.output, 'candidate_metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    with open(os.path.join(args.output, 'candidate_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['random_state'] + METRIC_NAMES +
                                 ['is_balanced_selected', 'is_naive_selected'])
        writer.writeheader()
        writer.writerows(candidate_rows)

    pdf_path = os.path.join(args.output, 'balanced_gmm_test_report.pdf')
    with PdfPages(pdf_path) as pdf:
        plot_loss_page(pdf, history)
        plot_candidate_metrics_page(pdf, candidate_rows)
        plot_heatmap_page(pdf, ari_matrix, random_states, 'Pairwise ARI across the 10 GMM candidates')
        plot_heatmap_page(pdf, nmi_matrix, random_states, 'Pairwise NMI across the 10 GMM candidates')
        plot_pca_page(pdf, z, balanced_best['labels'], naive_best['labels'])

    print(f'balanced pick: random_state={balanced_best["random_state"]}')
    print(f'naive pick:    random_state={naive_best["random_state"]}')
    print(f'ARI(balanced, naive) = {ari_balanced_vs_naive:.4f}')
    print(f'mean ARI(balanced, others) = {mean_ari_balanced_to_others:.4f}')
    print(f'mean ARI(naive, others)    = {mean_ari_naive_to_others:.4f}')
    print(f'wrote {pdf_path}')


if __name__ == '__main__':
    main()
