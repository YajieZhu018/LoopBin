### Focused re-plot of 03's scatter page for a single stress config (default: mild) in a
### single results run (default: n30000_linear). The pooled-across-all-configs scatter mixes
### datasets of very different difficulty, so a metric can look predictive purely from that
### cross-dataset confound; restricting to one config removes the severity axis and asks
### whether the metrics track ARI *within* a comparable difficulty regime.
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

METRIC_NAMES = ['min_weight', 'separation', 'cov_health', 'assignment_confidence', 'log_likelihood']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', default='n30000_linear',
                        help='subfolder of results/ holding per_candidate_metrics.json')
    parser.add_argument('-c', '--config', default='mild',
                        help='stress config to keep (mild/moderate/severe)')
    args = parser.parse_args()

    run_dir = os.path.join(SCRIPT_DIR, 'results', args.run)
    with open(os.path.join(run_dir, 'per_candidate_metrics.json')) as f:
        df = pd.DataFrame(json.load(f))
    sub = df[df['config'] == args.config].copy()
    if sub.empty:
        raise SystemExit(f'no candidates with config={args.config} in {run_dir}')

    datasets = sorted(sub['dataset'].unique())
    cmap = plt.get_cmap('tab10')
    palette = {d: cmap(i % 10) for i, d in enumerate(datasets)}

    pooled = {}
    within = {}
    for metric in METRIC_NAMES:
        x = sub[metric].to_numpy(dtype=float)
        y = sub['ari'].to_numpy(dtype=float)
        pooled[metric] = spearmanr(x, y) if np.std(x) > 1e-10 else (np.nan, np.nan)
        rhos = []
        for d in datasets:
            s = sub[sub['dataset'] == d]
            xd = s[metric].to_numpy(dtype=float)
            yd = s['ari'].to_numpy(dtype=float)
            if np.std(xd) > 1e-10 and np.std(yd) > 1e-10:
                rhos.append(spearmanr(xd, yd).correlation)
        within[metric] = (float(np.mean(rhos)) if rhos else np.nan, rhos)

    out_pdf = os.path.join(run_dir, f'ari_vs_metrics_{args.config}_only.pdf')
    with PdfPages(out_pdf) as pdf:
        # page 1: scatter of each metric vs ARI, mild candidates only
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        for ax, metric in zip(axes, METRIC_NAMES):
            for d in datasets:
                s = sub[sub['dataset'] == d]
                ax.scatter(s[metric], s['ari'], s=28, alpha=0.75,
                           color=palette[d], label=d, edgecolors='none')
            x = sub[metric].to_numpy(dtype=float)
            y = sub['ari'].to_numpy(dtype=float)
            if np.std(x) > 1e-10:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + intercept, color='black', linewidth=1, linestyle='--')
            rho, p = pooled[metric]
            mean_rho = within[metric][0]
            ax.set_title(f'{metric}\npooled rho={rho:+.3f} (p={p:.2g})   '
                         f'within-dataset mean rho={mean_rho:+.3f}', fontsize=9)
            ax.set_xlabel(metric)
            ax.set_ylabel('ARI')
        axes[-1].axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[-1].legend(handles, labels, loc='center', title='dataset', fontsize=9)
        fig.suptitle(f'{args.run}: metric vs. ARI for {args.config} datasets only '
                     f'({len(sub)} candidates, {len(datasets)} datasets)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: rho ranking, pooled-within-config vs mean within-dataset
        order = sorted(METRIC_NAMES, key=lambda m: -abs(pooled[m][0]))
        idx = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(idx + 0.2, [pooled[m][0] for m in order], height=0.4,
                color='#2a7f62', label=f'pooled within {args.config}')
        ax.barh(idx - 0.2, [within[m][0] for m in order], height=0.4,
                color='#c0392b', label='mean within-dataset')
        ax.set_yticks(idx)
        ax.set_yticklabels(order)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Spearman rho vs. ARI')
        ax.set_title(f'{args.config}-only correlation ranking '
                     f'(within-dataset rho is the confound-free one)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    out_json = os.path.join(run_dir, f'correlations_{args.config}_only.json')
    with open(out_json, 'w') as f:
        json.dump({m: {'pooled_rho': None if np.isnan(pooled[m][0]) else float(pooled[m][0]),
                       'pooled_p': None if np.isnan(pooled[m][1]) else float(pooled[m][1]),
                       'within_dataset_mean_rho': None if np.isnan(within[m][0]) else within[m][0],
                       'per_dataset_rho': dict(zip(datasets, within[m][1]))}
                   for m in METRIC_NAMES}, f, indent=2)

    print(f'wrote {out_pdf}')
    print(f'wrote {out_json}')
    for m in sorted(METRIC_NAMES, key=lambda m: -abs(pooled[m][0])):
        print(f'  {m:<22} pooled rho={pooled[m][0]:+.3f} p={pooled[m][1]:.2g}   '
              f'within mean rho={within[m][0]:+.3f}')


if __name__ == '__main__':
    main()
