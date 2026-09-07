### Turns the JSON results already produced by 02_evaluate_metric_weights.py into a single PDF
### report, so the "which of the 5 raw metrics should select_balanced_gmm's composite score
### weight" decision is visual instead of just numbers in JSON.
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

METRIC_NAMES = ['min_weight', 'separation', 'cov_health', 'assignment_confidence', 'log_likelihood']

# strategy keys as written by 02_evaluate_metric_weights.py's strategy_mean_ari.json /
# per_dataset_summary.json (each is f'{key}_ari' there except oracle/naive/random which
# already include '_ari' in WEIGHT_COMBOS' sibling keys)
STRATEGY_KEYS = [
    'oracle_ari', 'naive_loglik_ari', 'random_pick_ari',
    'separation_only_ari', 'confidence_only_ari', 'loglik_only_ari',
    'separation_confidence_ari', 'separation_loglik_ari', 'confidence_loglik_ari',
    'equal_current_default_ari',
]


def load_results():
    with open(os.path.join(RESULTS_DIR, 'correlations.json')) as f:
        correlations = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'strategy_mean_ari.json')) as f:
        strategy_mean_ari = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'per_dataset_summary.json')) as f:
        per_dataset_summary = pd.DataFrame(json.load(f))
    with open(os.path.join(RESULTS_DIR, 'per_candidate_metrics.json')) as f:
        per_candidate_metrics = pd.DataFrame(json.load(f))

    # Both optional -- absent for runs from before these were added, so older
    # results/ folders still render (just without the pages that need them).
    within_dataset_correlations = None
    within_path = os.path.join(RESULTS_DIR, 'within_dataset_correlations.json')
    if os.path.exists(within_path):
        with open(within_path) as f:
            within_dataset_correlations = json.load(f)

    training_curves = None
    curves_path = os.path.join(RESULTS_DIR, 'training_curves.json')
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            training_curves = json.load(f)

    latent_diagnostics = None
    latent_path = os.path.join(RESULTS_DIR, 'latent_diagnostics.json')
    if os.path.exists(latent_path):
        with open(latent_path) as f:
            latent_diagnostics = json.load(f)

    return (correlations, strategy_mean_ari, per_dataset_summary, per_candidate_metrics,
            within_dataset_correlations, training_curves, latent_diagnostics)


def plot_correlation_page(pdf, correlations):
    names = sorted(METRIC_NAMES, key=lambda m: -abs(correlations[m]['spearman_rho']))
    rhos = [correlations[m]['spearman_rho'] for m in names]
    pvals = [correlations[m]['p_value'] for m in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2a7f62' if r >= 0 else '#c0392b' for r in rhos]
    bars = ax.barh(names, rhos, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Spearman rho vs. ARI (pooled across all fitted candidates)')
    ax.set_title('Which raw metrics actually predict ARI? (pooled -- see next page: this inflates/deflates '
                  'metrics via cross-dataset scale confounds)')
    for bar, rho, p in zip(bars, rhos, pvals):
        x = bar.get_width()
        offset = 0.02 if x >= 0 else -0.02
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                 f'rho={rho:.3f}, p={p:.1e}', va='center', ha=ha, fontsize=8)
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_scatter_page(pdf, per_candidate_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    configs = sorted(per_candidate_metrics['config'].unique())
    cmap = plt.get_cmap('tab10')
    palette = {config: cmap(i % 10) for i, config in enumerate(configs)}

    for ax, metric in zip(axes, METRIC_NAMES):
        for config in configs:
            sub = per_candidate_metrics[per_candidate_metrics['config'] == config]
            ax.scatter(sub[metric], sub['ari'], s=12, alpha=0.6,
                       color=palette[config], label=config)
        x = per_candidate_metrics[metric].to_numpy(dtype=float)
        y = per_candidate_metrics['ari'].to_numpy(dtype=float)
        if np.std(x) > 1e-10:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel(metric)
        ax.set_ylabel('ARI')
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='center', title='config', fontsize=8)
    fig.suptitle(f'Metric value vs. ARI, per candidate '
                 f'({len(per_candidate_metrics)} candidates, colored by stress config)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_within_vs_pooled_page(pdf, correlations, within_dataset_correlations):
    """The headline page: pooling raw metric values across datasets of very different
    difficulty/scale can make a metric look predictive purely from a cross-dataset
    confound (easier datasets have both higher raw values AND higher ARI, unrelated
    to whether the metric actually distinguishes candidates within one dataset).
    Within-dataset rho -- averaging a per-dataset Spearman computed only among that
    dataset's own candidates -- is what select_balanced_gmm actually needs, since it
    only ever compares candidates fit on the same data. See run_report.md."""
    names = sorted(METRIC_NAMES, key=lambda m: -abs(correlations[m]['spearman_rho']))
    pooled = [correlations[m]['spearman_rho'] for m in names]
    within = [within_dataset_correlations[m]['mean_rho']
              if within_dataset_correlations[m]['mean_rho'] is not None else 0.0
              for m in names]

    y = np.arange(len(names))
    height = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + height / 2, pooled, height=height, color='#9b59b6', label='pooled across datasets (confounded)')
    ax.barh(y - height / 2, within, height=height, color='#2a7f62', label='mean within-dataset (trustworthy)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Spearman rho vs. ARI')
    ax.set_title('Pooled correlation can overstate (or invert) predictive power --\n'
                  'within-dataset rho is the number to trust')
    ax.set_xlim(-1, 1)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_training_curves_page(pdf, training_curves):
    datasets = sorted(training_curves.keys())
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, dataset in zip(axes_flat, datasets):
        losses = training_curves[dataset]
        ax.plot(range(1, len(losses) + 1), losses, linewidth=1, color='#4c72b0')
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel('epoch', fontsize=8)
        ax.set_ylabel('AE loss', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(datasets):]:
        ax.axis('off')
    fig.suptitle('AE pretraining loss vs. epoch, per dataset')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_latent_diagnostics_page(pdf, latent_diagnostics):
    """Three views of latent-space health: how many dimensions survive, how many are
    *independently* useful, and whether the dims are correlated within a cluster (the
    assumption covariance_type='diag' rests on). See compute_latent_diagnostics in
    02_evaluate_metric_weights.py for what each quantity means."""
    activation = latent_diagnostics.get('latent_activation', 'unknown')
    per_dataset = latent_diagnostics['per_dataset']
    datasets = sorted(per_dataset.keys())
    x = np.arange(len(datasets))
    n_dims = per_dataset[datasets[0]]['n_dims']

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    ax = axes[0]
    alive = [per_dataset[d]['n_alive_dims'] for d in datasets]
    eff_rank = [per_dataset[d]['effective_rank'] for d in datasets]
    ax.bar(x - 0.2, alive, width=0.4, color='#4c72b0', label='alive dims (variance > 0)')
    ax.bar(x + 0.2, eff_rank, width=0.4, color='#dd8452',
           label='effective rank (participation ratio)')
    ax.axhline(n_dims, color='#2a7f62', linestyle='--', linewidth=1,
               label=f'bottleneck width ({n_dims})')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('dimensions')
    ax.legend(fontsize=8)
    ax.set_title('Usable latent dimensions. Effective rank below the alive count means the '
                 'surviving dims are redundant with each other.', fontsize=10)

    ax = axes[1]
    within = [per_dataset[d]['within_cluster_mean_abs_corr'] for d in datasets]
    glob = [per_dataset[d]['global_mean_abs_corr'] for d in datasets]
    within_plot = [0 if v is None else v for v in within]
    glob_plot = [0 if v is None else v for v in glob]
    ax.bar(x - 0.2, within_plot, width=0.4, color='#c0392b',
           label="within-cluster mean |r|  (this is what 'diag' assumes is ~0)")
    ax.bar(x + 0.2, glob_plot, width=0.4, color='#999999',
           label='global mean |r|  (expected to be higher; not a problem)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('mean |correlation|')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Latent dimension correlation. High within-cluster |r| breaks the diagonal-covariance "
                 "assumption in select_balanced_gmm and VADE's GMM layer.", fontsize=10)

    ax = axes[2]
    for d in datasets:
        variances = np.sort(np.array(per_dataset[d]['per_dim_variance']))[::-1]
        ax.plot(np.arange(1, len(variances) + 1), np.maximum(variances, 1e-12),
                marker='o', markersize=3, linewidth=1, label=d)
    ax.set_yscale('log')
    ax.set_xlabel('latent dim (sorted by variance, descending)')
    ax.set_ylabel('variance (log scale)')
    ax.legend(fontsize=7, ncol=3)
    ax.set_title('Per-dimension variance spectrum. A cliff to the floor marks dead dims.', fontsize=10)

    fig.suptitle(f'Latent-space health (latent_activation={activation})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def plot_strategy_pooled_page(pdf, strategy_mean_ari, n_datasets):
    ordered = sorted(STRATEGY_KEYS, key=lambda k: -strategy_mean_ari[k])
    values = [strategy_mean_ari[k] for k in ordered]
    labels = [k.replace('_ari', '') for k in ordered]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#7f7f7f' if lbl in ('oracle', 'random_pick') else '#4c72b0' for lbl in labels]
    ax.bar(labels, values, color=colors)
    ax.axhline(strategy_mean_ari['oracle_ari'], color='#2a7f62', linestyle='--',
               linewidth=1, label='oracle')
    ax.axhline(strategy_mean_ari['random_pick_ari'], color='#c0392b', linestyle='--',
               linewidth=1, label='random pick')
    ax.set_ylabel(f'Mean ARI (pooled across {n_datasets} toy datasets)')
    ax.set_title('Selection strategy vs. oracle / random baselines')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_strategy_by_config_page(pdf, per_dataset_summary):
    grouped = per_dataset_summary.groupby('config')[STRATEGY_KEYS].mean()
    grouped = grouped.loc[grouped['oracle_ari'].sort_values().index]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    n_strategies = len(STRATEGY_KEYS)
    x = np.arange(len(grouped.index))
    width = 0.8 / n_strategies
    for i, key in enumerate(STRATEGY_KEYS):
        ax.bar(x + i * width, grouped[key], width=width, label=key.replace('_ari', ''))
    ax.set_xticks(x + width * (n_strategies - 1) / 2)
    ax.set_xticklabels(grouped.index, rotation=30, ha='right')
    ax.set_ylabel('Mean ARI (averaged over n_centroid=4,6,10)')
    ax.set_title('Strategy ARI by stress config — where does the metric choice matter?')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_summary_page(pdf, correlations, strategy_mean_ari, within_dataset_correlations=None):
    ranked_metrics = sorted(METRIC_NAMES, key=lambda m: -abs(correlations[m]['spearman_rho']))
    ranked_strategies = sorted(STRATEGY_KEYS, key=lambda k: -strategy_mean_ari[k])

    lines = ['Metric → ARI correlation ranking (pooled, |rho| descending):', '']
    for m in ranked_metrics:
        c = correlations[m]
        lines.append(f'  {m:<22} rho={c["spearman_rho"]:+.3f}  p={c["p_value"]:.1e}')

    if within_dataset_correlations is not None:
        ranked_within = sorted(
            METRIC_NAMES,
            key=lambda m: -abs(within_dataset_correlations[m]['mean_rho'] or 0.0))
        lines += ['', 'Metric → ARI correlation ranking (WITHIN-DATASET mean, |rho| descending -- trust this one):', '']
        for m in ranked_within:
            r = within_dataset_correlations[m]
            mean_str = f'{r["mean_rho"]:+.3f}' if r['mean_rho'] is not None else 'n/a'
            lines.append(f'  {m:<22} mean_rho={mean_str}  ({r["n_datasets_valid"]}/{r["n_datasets_total"]} datasets)')

    lines += ['', 'Top 3 selection strategies by pooled mean ARI:', '']
    for k in ranked_strategies[:3]:
        lines.append(f'  {k.replace("_ari", ""):<22} mean ARI={strategy_mean_ari[k]:.4f}')
    lines += ['', f'  {"oracle":<22} mean ARI={strategy_mean_ari["oracle_ari"]:.4f}  (upper bound)']
    lines += [f'  {"random_pick":<22} mean ARI={strategy_mean_ari["random_pick_ari"]:.4f}  (lower bound)']

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', fontsize=11, family='monospace')
    ax.set_title('Metric selection summary')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_csv(strategy_mean_ari):
    ordered = sorted(STRATEGY_KEYS, key=lambda k: -strategy_mean_ari[k])
    out_path = os.path.join(RESULTS_DIR, 'metric_selection_summary.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'strategy', 'mean_ari'])
        for rank, key in enumerate(ordered, start=1):
            writer.writerow([rank, key.replace('_ari', ''), f'{strategy_mean_ari[key]:.6f}'])
    return out_path


def main():
    (correlations, strategy_mean_ari, per_dataset_summary, per_candidate_metrics,
     within_dataset_correlations, training_curves, latent_diagnostics) = load_results()
    n_datasets = len(per_dataset_summary)

    pdf_path = os.path.join(RESULTS_DIR, 'metric_selection_report.pdf')
    with PdfPages(pdf_path) as pdf:
        # Latent health first: every metric downstream is computed *in* this space,
        # so if it's degenerate the rest of the report can't be read at face value.
        if latent_diagnostics:
            plot_latent_diagnostics_page(pdf, latent_diagnostics)
        plot_correlation_page(pdf, correlations)
        if within_dataset_correlations is not None:
            plot_within_vs_pooled_page(pdf, correlations, within_dataset_correlations)
        plot_scatter_page(pdf, per_candidate_metrics)
        plot_strategy_pooled_page(pdf, strategy_mean_ari, n_datasets)
        plot_strategy_by_config_page(pdf, per_dataset_summary)
        plot_summary_page(pdf, correlations, strategy_mean_ari, within_dataset_correlations)
        if training_curves:
            plot_training_curves_page(pdf, training_curves)

    csv_path = write_summary_csv(strategy_mean_ari)
    print(f'wrote {pdf_path}')
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
