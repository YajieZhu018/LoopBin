"""Latent-space health diagnostics, shared by 02_evaluate_metric_weights.py (toy
data, ground-truth labels) and 04_test_real_pretrain_gmm.py (real loop data, fitted
GMM labels). Lives in its own module because both callers have digit-leading
filenames that can't be imported normally.

See run_report.md's 2026-08-27 entry for why these three quantities in particular.
"""
import numpy as np

# Per-dim variance at/below this counts a latent dim as "dead". Also the cutoff for
# excluding a dim from the correlation matrices, where zero variance gives 0/0 = NaN.
DEAD_VAR_THRESHOLD = 1e-8


def _offdiag_abs_corr(sub):
    """(max, mean) absolute off-diagonal correlation of `sub`'s columns, or
    (None, None) if there aren't enough usable columns/rows to define one. NaN
    entries -- a column with no variance *within this subset* -- are dropped rather
    than propagated, so one degenerate dim doesn't wipe out the whole statistic (the
    trap that left cov_health reported as a bare NaN in earlier runs)."""
    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return None, None
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(sub, rowvar=False)
    values = np.abs(corr[np.triu_indices_from(corr, k=1)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return float(values.max()), float(values.mean())


def compute_latent_diagnostics(z, labels):
    """
    How usable is this latent space at all, independently of any GMM fit quality?

      - dead dims: per-dim variance <= DEAD_VAR_THRESHOLD. Under a ReLU bottleneck
        these are units whose pre-activation went negative for every sample, so they
        output exactly 0 and receive exactly 0 gradient forever.
      - effective_rank: participation ratio of cov(z)'s eigenvalues,
        (sum L)^2 / sum L^2. A continuous "how many dimensions is this really", and a
        strictly better measure than the dead-dim count because it also penalizes
        *redundant* alive dims -- 4 alive dims that are near-copies of each other
        score ~1, not 4.
      - within-cluster correlation: both select_balanced_gmm (covariance_type='diag')
        and VADE's GMM layer (lambda_p is per-dim, not a full matrix) assume latent
        dims are uncorrelated *within* a component. Global correlation across the
        whole latent space is NOT the test for that -- it is expected to be high
        whenever clusters separate along a diagonal direction, which is signal, not
        pathology. So the correlation that matters is computed within each group of
        `labels` and averaged.

    `labels` groups the rows for the within-cluster statistic. On toy data that's the
    ground truth. On real data there is no ground truth, so callers pass the fitted
    GMM's own hard assignments -- note that makes the within-cluster number somewhat
    OPTIMISTIC, since those clusters were chosen by a diagonal-covariance model that
    prefers axis-aligned groups. A high value there is therefore strong evidence of a
    problem; a low one is weaker evidence of health.
    """
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels)
    variances = z.var(axis=0)
    alive = variances > DEAD_VAR_THRESHOLD

    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0.0, None)
    eig_sum = float(eigenvalues.sum())
    sq_sum = float(np.square(eigenvalues).sum())
    effective_rank = eig_sum ** 2 / sq_sum if sq_sum > 1e-30 else 0.0

    z_alive = z[:, alive]
    global_max, global_mean = _offdiag_abs_corr(z_alive)

    per_cluster = {}
    for label in np.unique(labels):
        max_abs, mean_abs = _offdiag_abs_corr(z_alive[labels == label])
        per_cluster[str(int(label))] = {'max_abs_corr': max_abs, 'mean_abs_corr': mean_abs}
    cluster_maxes = [v['max_abs_corr'] for v in per_cluster.values() if v['max_abs_corr'] is not None]
    cluster_means = [v['mean_abs_corr'] for v in per_cluster.values() if v['mean_abs_corr'] is not None]

    return {
        'n_dims': int(z.shape[1]),
        'n_alive_dims': int(alive.sum()),
        'n_dead_dims': int((~alive).sum()),
        'effective_rank': float(effective_rank),
        'per_dim_variance': [float(v) for v in variances],
        'global_max_abs_corr': global_max,
        'global_mean_abs_corr': global_mean,
        'within_cluster_max_abs_corr': float(np.mean(cluster_maxes)) if cluster_maxes else None,
        'within_cluster_mean_abs_corr': float(np.mean(cluster_means)) if cluster_means else None,
        'n_clusters_valid': len(cluster_means),
        'n_clusters_total': len(per_cluster),
        'per_cluster_corr': per_cluster,
    }


def fmt(value):
    return 'n/a' if value is None else '%.3f' % value


CORRELATION_LEGEND = (
    "within|r| is the number that matters for covariance_type='diag' -- it should be near 0.\n"
    "global|r| is context: when it sits well ABOVE within|r|, the correlation is just clusters\n"
    "separating along a diagonal (harmless). When the two are equal, the correlation is\n"
    "intrinsic to each component, which is the case 'diag' cannot represent."
)


def format_table(rows):
    """rows: iterable of (name, diagnostics dict) -> printable table string."""
    lines = ['%-18s%9s%10s%11s%11s' % ('dataset', 'alive', 'eff_rank', 'within|r|', 'global|r|')]
    for name, d in rows:
        lines.append('%-18s%9s%10.2f%11s%11s' % (
            name,
            '%d/%d' % (d['n_alive_dims'], d['n_dims']),
            d['effective_rank'],
            fmt(d['within_cluster_mean_abs_corr']),
            fmt(d['global_mean_abs_corr']),
        ))
    return '\n'.join(lines)
