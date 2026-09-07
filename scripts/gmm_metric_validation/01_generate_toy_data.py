### Generates synthetic, biological-data-like datasets with known ground-truth cluster
### labels, used (together with 02_evaluate_metric_weights.py) to validate which of
### select_balanced_gmm's five candidate-selection metrics (balance, separation,
### covariance health, assignment confidence, fit quality) actually predict high
### clustering accuracy once real AE pretraining is in the loop.
###
### Unlike a direct latent-space blob generator, this samples cluster structure in a
### low-dim "true biological state" space, then nonlinearly maps it to a per-feature
### Poisson rate and samples counts from it -- i.e. the same kind of stochastic
### counting process that produces real Micro-C/ChIP-seq signal (sparse, right-skewed,
### heteroskedastic), not a dense symmetric distribution a GMM could already separate
### directly. This choice isn't cosmetic: an earlier version used a smooth
### tanh-projection + additive Gaussian noise + min-max normalize (dense, ~0.5-mean,
### symmetric per feature), and the real AE (src/model/ae.py, plain ReLU MLP,
### Glorot-uniform init, no batch norm) reliably collapsed to a dead 10-D latent space
### on it (0% ARI on every config). Real pretrained data (data/02_process/
### merged_log_data.npy) has mean~0.05 and 68% of values <0.05 -- switching to a
### Poisson-count generative process + log1p + min-max (matching that skewed, sparse
### profile) is what keeps the encoder's ReLU units alive. See run_report.md for the
### full diagnosis.
import json
import os

import numpy as np
from scipy.spatial.distance import pdist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'toy_data')

LATENT_FACTOR_DIM = 10   # true generative "biological state" dim; matches src/model/ae.py's bottleneck
HIDDEN_DIM = 128          # width of the nonlinear projection into raw space
RAW_DIM = 256 + 32 * 4    # nods to the real pipeline's 16x16 Micro-C + 4-protein flat layout
N_SAMPLES = 30000
N_CENTROID_VALUES = (4, 6, 10)  # matches the real range used in main.py / trials

# Blur level -> target separation between cluster centers, in pooled-std units
# (mild = well-separated, severe = heavily overlapping/continuous boundary).
BLUR_LEVELS = {
    'mild': 6.0,
    'moderate': 3.0,
    'severe': 1.5,
}
# Shifts the per-feature log-rate distribution so most sampled counts are low/zero
# (tuned empirically against real AE training behavior -- see module docstring;
# -1.5 gave ~75% zero counts, close to real data's low-signal-dominated profile,
# and was the best-performing of {-1.5, -0.5, 0.0} tested).
RATE_BIAS = -1.5


def make_means(n_centroid, target_separation, rng):
    means = rng.randn(n_centroid, LATENT_FACTOR_DIM)
    means *= target_separation / pdist(means).mean()
    return means


def make_covariances(n_centroid, rng):
    """Per-cluster covariance in factor space: a random rotation of a diagonal
    variance matrix, so clusters are elliptical/correlated rather than isotropic."""
    covs = []
    for _ in range(n_centroid):
        variances = rng.uniform(0.5, 1.5, size=LATENT_FACTOR_DIM)
        R, _ = np.linalg.qr(rng.randn(LATENT_FACTOR_DIM, LATENT_FACTOR_DIM))
        covs.append(R @ np.diag(variances) @ R.T)
    return covs


def make_projection(rng):
    W1 = rng.randn(LATENT_FACTOR_DIM, HIDDEN_DIM) / np.sqrt(LATENT_FACTOR_DIM)
    b1 = rng.randn(HIDDEN_DIM) * 0.1
    W2 = rng.randn(HIDDEN_DIM, RAW_DIM) / np.sqrt(HIDDEN_DIM)
    b2 = rng.randn(RAW_DIM) * 0.1 + RATE_BIAS
    return W1, b1, W2, b2


def sample_counts(factors, projection, rng):
    """Nonlinearly maps factors to a per-feature Poisson rate and samples counts --
    the sparse, right-skewed, heteroskedastic (variance=mean) signal this produces is
    what a real AE can actually learn from without its ReLU bottleneck collapsing
    (see module docstring)."""
    W1, b1, W2, b2 = projection
    hidden = np.tanh(factors @ W1 + b1)
    log_rate = hidden @ W2 + b2
    rate = np.exp(np.clip(log_rate, -10, 3))  # clip avoids float overflow in exp
    return rng.poisson(rate).astype(np.float64)


def log_minmax_normalize(raw_counts):
    """log1p + per-feature min-max to [0,1], matching the real pipeline's
    log_epi/log_microc + normalize steps (src/fn/processing.py)."""
    logx = np.log1p(raw_counts)
    lo, hi = logx.min(axis=0), logx.max(axis=0)
    span = np.where(hi - lo < 1e-10, 1.0, hi - lo)
    return (logx - lo) / span


def sample_dataset(n_centroid, blur_level, seed):
    rng = np.random.RandomState(seed)
    weights = rng.uniform(0.9, 1.1, size=n_centroid)
    weights /= weights.sum()

    means = make_means(n_centroid, BLUR_LEVELS[blur_level], rng)
    covariances = make_covariances(n_centroid, rng)
    projection = make_projection(rng)

    counts = np.round(weights * N_SAMPLES).astype(int)
    counts[-1] += N_SAMPLES - counts.sum()  # absorb rounding remainder

    factor_parts, y_parts = [], []
    for k in range(n_centroid):
        n_k = max(int(counts[k]), 0)
        if n_k == 0:
            continue
        factor_parts.append(rng.multivariate_normal(means[k], covariances[k], size=n_k))
        y_parts.append(np.full(n_k, k, dtype=int))

    factors = np.concatenate(factor_parts, axis=0)
    y_true = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(len(y_true))
    factors, y_true = factors[perm], y_true[perm]

    raw_counts = sample_counts(factors, projection, rng)
    X = log_minmax_normalize(raw_counts)
    return X, y_true, means, covariances


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []
    seed = 0
    for blur_level in BLUR_LEVELS:
        for n_centroid in N_CENTROID_VALUES:
            X, y_true, true_factor_means, true_factor_covs = sample_dataset(
                n_centroid, blur_level, seed)

            filename = f'{blur_level}_k{n_centroid}.npz'
            np.savez(os.path.join(OUTPUT_DIR, filename), X=X, y_true=y_true,
                     true_factor_means=true_factor_means,
                     true_factor_covs=np.stack(true_factor_covs),
                     seed=seed, n_centroid=n_centroid)

            manifest.append({
                'filename': filename, 'config_name': blur_level,
                'n_centroid': n_centroid, 'n_samples': N_SAMPLES,
                'raw_dim': RAW_DIM, 'latent_factor_dim': LATENT_FACTOR_DIM, 'seed': seed,
            })
            observed_weights = np.round(np.bincount(y_true) / len(y_true), 4)
            print(f'saved {filename}: X.shape={X.shape}, observed_weights={observed_weights}')
            seed += 1

    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote manifest with {len(manifest)} datasets to {manifest_path}')


if __name__ == '__main__':
    main()
