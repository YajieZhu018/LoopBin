### Evaluates select_balanced_gmm's five candidate-selection metrics (balance,
### separation, covariance health, assignment confidence, fit quality) against ARI/NMI
### on the toy datasets from 01_generate_toy_data.py, whose ground-truth labels are
### known. Unlike a direct-on-raw-data evaluation, this first pretrains an AE on each
### dataset (same architecture/hyperparameters as main.py's pretrain_ae) and evaluates
### GMM candidates on the resulting latent space -- the same space select_balanced_gmm
### operates on in the real pipeline (VADE.load_pretrained_weights). Reuses
### _fit_gmm_candidates (src/model/vade_model.py) to fit the same candidate pool once
### per dataset's latent space, then:
###   1. correlates each raw metric with ARI, pooled across all fitted candidates
###   2. compares the mean ARI achieved by several metric_weights combinations
###      (including the current default) against an oracle upper bound, a random-pick
###      baseline, and a "pick by raw log-likelihood only" baseline (mimicking
###      sklearn's default single-best-fit behavior, the thing select_balanced_gmm
###      was written to improve on).
### The pretrained AE (SavedModel dir) and its latent embedding are saved per dataset
### under pretrained/ so they can be reloaded later without rerunning pretraining.
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOY_DATA_DIR = os.path.join(SCRIPT_DIR, 'toy_data')
MANIFEST_PATH = os.path.join(TOY_DATA_DIR, 'manifest.json')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, 'pretrained')

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'loopbin', 'model'))
sys.path.insert(0, SCRIPT_DIR)
from ae import AE  # noqa: E402
from vade_model import _fit_gmm_candidates  # noqa: E402
from latent_diagnostics import (  # noqa: E402
    compute_latent_diagnostics, format_table, CORRELATION_LEGEND)

PRETRAIN_SEED = 0     # matches main.py:pretrain_ae's fixed seed
PRETRAIN_EPOCHS = 100  # smaller than main.py's production default of 200, for sweep speed

# None = linear bottleneck. src/model/ae.py still defaults to 'relu' (what production
# main.py:pretrain_ae and every trials/ run use), so this is an opt-in experiment here
# rather than a change to the production model: does removing the ReLU on the 10-unit
# bottleneck fix the dead-latent-dim problem that has capped every run in run_report.md
# so far (8/10 dead at N=3000, 6/10 at N=10000, 5-6/10 at N=30000), and does that in
# turn make cov_health measurable? Set back to 'relu' to reproduce the earlier runs.
LATENT_ACTIVATION = None

MIN_WEIGHT_THRESHOLD = 0.02
COLLAPSE_RATIO_THRESHOLD = 0.05

# metric_weights = (separation, assignment_confidence, log_likelihood), same order
# select_balanced_gmm uses.
WEIGHT_COMBOS = {
    'separation_only': (1.0, 0.0, 0.0),
    'confidence_only': (0.0, 1.0, 0.0),
    'loglik_only': (0.0, 0.0, 1.0),
    'separation_confidence': (1.0, 1.0, 0.0),
    'separation_loglik': (1.0, 0.0, 1.0),
    'confidence_loglik': (0.0, 1.0, 1.0),
    'equal_current_default': (1.0, 1.0, 1.0),
}

METRIC_NAMES = ['min_weight', 'separation', 'cov_health', 'assignment_confidence', 'log_likelihood']


def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    return np.zeros_like(values) if std < 1e-10 else (values - values.mean()) / std


def composite_select(candidates, survivors, weights):
    """Mirrors select_balanced_gmm's post-fit selection logic exactly, so results
    here reflect what the real function would pick under each weight combo."""
    pool = survivors if survivors else candidates
    w_sep, w_conf, w_ll = weights
    scores = (w_sep * zscore([c['separation'] for c in pool])
              + w_conf * zscore([c['assignment_confidence'] for c in pool])
              + w_ll * zscore([c['log_likelihood'] for c in pool]))
    if survivors:
        return pool[int(np.argmax(scores))]
    best_idx = max(range(len(pool)), key=lambda i: (pool[i]['min_weight'], pool[i]['cov_health'], scores[i]))
    return pool[best_idx]


def pretrain_ae(X, epochs, seed, latent_activation=LATENT_ACTIVATION):
    """Replicates main.py's pretrain_ae architecture/hyperparameters (AE(d_input),
    Adam lr=0.002, batch_size=256, seed=0) as a standalone helper, so this script
    doesn't depend on main.py's CLI-oriented pretrain_ae/argument handling. The one
    deliberate deviation is latent_activation (see LATENT_ACTIVATION above)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    d_input = X.shape[1]
    ae = AE(d_input, latent_activation=latent_activation)
    ae(np.zeros((10, d_input)))
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002))
    history = ae.fit(X, shuffle=True, batch_size=256, epochs=epochs, verbose=0)
    return ae, history



def evaluate_dataset(entry):
    data = np.load(os.path.join(TOY_DATA_DIR, entry['filename']))
    X, y_true, n_centroid = data['X'], data['y_true'], int(entry['n_centroid'])
    dataset_name = entry['config_name'] + f"_k{n_centroid}"

    ae, history = pretrain_ae(X, PRETRAIN_EPOCHS, PRETRAIN_SEED)
    loss_curve = [float(x) for x in history.history['loss']]
    final_loss = loss_curve[-1]

    dataset_pretrain_dir = os.path.join(PRETRAINED_DIR, dataset_name)
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    ae.save(dataset_pretrain_dir)
    z = ae.encoder.predict(X, verbose=0)
    np.savez(os.path.join(PRETRAINED_DIR, f'{dataset_name}_latent.npz'), z=z, y_true=y_true)

    diagnostics = compute_latent_diagnostics(z, y_true)
    print(f"  latent: {diagnostics['n_alive_dims']}/{diagnostics['n_dims']} alive, "
          f"effective_rank={diagnostics['effective_rank']:.2f}, "
          f"within-cluster mean|r|={diagnostics['within_cluster_mean_abs_corr']}, "
          f"global mean|r|={diagnostics['global_mean_abs_corr']}")

    candidates = _fit_gmm_candidates(z, n_centroid)
    for c in candidates:
        y_pred = c['gmm'].predict(z)
        c['ari'] = float(adjusted_rand_score(y_true, y_pred))
        c['nmi'] = float(normalized_mutual_info_score(y_true, y_pred))

    survivors = [c for c in candidates if c['min_weight'] >= MIN_WEIGHT_THRESHOLD
                 and c['cov_health'] >= COLLAPSE_RATIO_THRESHOLD]

    oracle = max(candidates, key=lambda c: c['ari'])
    naive_loglik = max(candidates, key=lambda c: c['log_likelihood'])
    random_pick_ari = float(np.mean([c['ari'] for c in candidates]))

    row = {
        'dataset': entry['filename'], 'config': entry['config_name'],
        'n_centroid': n_centroid, 'n_candidates': len(candidates),
        'n_survivors': len(survivors), 'ae_final_loss': final_loss,
        'oracle_ari': oracle['ari'], 'naive_loglik_ari': naive_loglik['ari'],
        'random_pick_ari': random_pick_ari,
        # headline latent-health numbers inline too, so they land in
        # per_dataset_summary.json next to the ARIs they help explain
        'n_dead_dims': diagnostics['n_dead_dims'],
        'effective_rank': diagnostics['effective_rank'],
        'within_cluster_mean_abs_corr': diagnostics['within_cluster_mean_abs_corr'],
    }
    for combo_name, weights in WEIGHT_COMBOS.items():
        selected = composite_select(candidates, survivors, weights)
        row[f'{combo_name}_ari'] = selected['ari']

    candidate_rows = [
        {'dataset': entry['filename'], 'config': entry['config_name'],
         'random_state': c['random_state'], **{m: c[m] for m in METRIC_NAMES},
         'ari': c['ari'], 'nmi': c['nmi']}
        for c in candidates
    ]
    return row, candidate_rows, loss_curve, diagnostics


def compute_within_dataset_correlations(all_candidate_rows):
    """Per-dataset Spearman rho of each metric vs. ARI, averaged across datasets --
    as opposed to the pooled correlation (all datasets' candidates combined), which
    is what select_balanced_gmm actually needs (it only ever compares candidates
    fit on the *same* dataset against each other). Pooling raw metric values across
    datasets whose scale differs with n_centroid/blur level can make a metric look
    predictive purely because "easier" datasets happen to have both higher raw
    metric values and higher ARI ceilings, with zero actual within-dataset
    discriminative power -- see run_report.md's 2026-08-26 entry for a worked
    example (assignment_confidence: pooled rho 0.47, mean within-dataset rho -0.14).
    """
    by_dataset = {}
    for row in all_candidate_rows:
        by_dataset.setdefault(row['dataset'], []).append(row)

    result = {}
    for m in METRIC_NAMES:
        per_dataset_rho = {}
        for dataset, rows in by_dataset.items():
            values = np.array([r[m] for r in rows], dtype=float)
            aris = np.array([r['ari'] for r in rows], dtype=float)
            if np.std(values) < 1e-10:
                per_dataset_rho[dataset] = None  # metric constant within this dataset -- undefined
            else:
                rho, _ = spearmanr(values, aris)
                per_dataset_rho[dataset] = None if np.isnan(rho) else float(rho)
        valid = [r for r in per_dataset_rho.values() if r is not None]
        result[m] = {
            'mean_rho': float(np.mean(valid)) if valid else None,
            'median_rho': float(np.median(valid)) if valid else None,
            'n_datasets_valid': len(valid),
            'n_datasets_total': len(by_dataset),
            'per_dataset_rho': per_dataset_rho,
        }
    return result


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    summary_rows, all_candidate_rows = [], []
    training_curves, latent_diagnostics = {}, {}
    for entry in manifest:
        print(f"evaluating {entry['filename']} ...")
        row, candidate_rows, loss_curve, diagnostics = evaluate_dataset(entry)
        summary_rows.append(row)
        all_candidate_rows.extend(candidate_rows)
        training_curves[row['dataset']] = loss_curve
        latent_diagnostics[row['dataset']] = diagnostics

    with open(os.path.join(RESULTS_DIR, 'per_dataset_summary.json'), 'w') as f:
        json.dump(summary_rows, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'per_candidate_metrics.json'), 'w') as f:
        json.dump(all_candidate_rows, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'training_curves.json'), 'w') as f:
        json.dump(training_curves, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'latent_diagnostics.json'), 'w') as f:
        json.dump({'latent_activation': str(LATENT_ACTIVATION),
                   'per_dataset': latent_diagnostics}, f, indent=2)

    print(f'\n=== Latent-space health (latent_activation={LATENT_ACTIVATION}) ===')
    print(CORRELATION_LEGEND)
    print(format_table(latent_diagnostics.items()))

    aris = np.array([r['ari'] for r in all_candidate_rows])
    print(f'\n=== Spearman correlation of each raw metric with ARI '
          f'(POOLED across all {len(all_candidate_rows)} fitted candidates -- inflated/deflated by '
          f'cross-dataset scale differences, see within-dataset numbers below for the trustworthy figure) ===')
    correlations = {}
    for m in METRIC_NAMES:
        values = np.array([r[m] for r in all_candidate_rows])
        rho, p = spearmanr(values, aris)
        correlations[m] = {'spearman_rho': float(rho), 'p_value': float(p)}
        print(f'{m}: rho={rho:.4f} (p={p:.2e})')

    within_dataset_correlations = compute_within_dataset_correlations(all_candidate_rows)
    print(f'\n=== Spearman correlation of each raw metric with ARI '
          f'(WITHIN-DATASET, mean over each of the {len(summary_rows)} datasets\' own candidate pool -- '
          f'this is what select_balanced_gmm actually needs) ===')
    for m in METRIC_NAMES:
        r = within_dataset_correlations[m]
        mean_str = f'{r["mean_rho"]:.4f}' if r['mean_rho'] is not None else 'n/a'
        median_str = f'{r["median_rho"]:.4f}' if r['median_rho'] is not None else 'n/a'
        print(f'{m}: mean_rho={mean_str}, median_rho={median_str} '
              f'({r["n_datasets_valid"]}/{r["n_datasets_total"]} datasets had non-constant metric values)')
    with open(os.path.join(RESULTS_DIR, 'within_dataset_correlations.json'), 'w') as f:
        json.dump(within_dataset_correlations, f, indent=2)

    print(f'\n=== Mean ARI by selection strategy (averaged over {len(summary_rows)} toy datasets) ===')
    strategy_means = {}
    for key in ['oracle_ari', 'naive_loglik_ari', 'random_pick_ari'] + [f'{name}_ari' for name in WEIGHT_COMBOS]:
        vals = [r[key] for r in summary_rows]
        strategy_means[key] = float(np.mean(vals))
        print(f'{key}: {np.mean(vals):.4f}')

    with open(os.path.join(RESULTS_DIR, 'correlations.json'), 'w') as f:
        json.dump(correlations, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'strategy_mean_ari.json'), 'w') as f:
        json.dump(strategy_means, f, indent=2)

    print(f'\nresults saved to {RESULTS_DIR}')
    print(f'pretrained models/latent embeddings saved to {PRETRAINED_DIR}')


if __name__ == '__main__':
    main()
