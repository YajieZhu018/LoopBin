### Reports cluster occupancy for one or more VaDE runs' labels.npy, so "some clusters
### collapse into one" is a measured number rather than an impression. Pure numpy --
### no TensorFlow import, so it is safe to run on the login node.
import argparse
import os

import numpy as np


def describe(path, n_clusters=None):
    labels = np.load(path)
    k = n_clusters or int(labels.max()) + 1
    counts = np.bincount(labels.astype(int), minlength=k)
    frac = counts / counts.sum()
    nonempty = int((counts > 0).sum())
    # normalized entropy: 1.0 = perfectly even, 0.0 = everything in one cluster
    nz = frac[frac > 0]
    even = float(-(nz * np.log(nz)).sum() / np.log(k)) if k > 1 else 1.0
    return labels, counts, frac, nonempty, even, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+', help='one or more labels.npy paths')
    parser.add_argument('--n-clusters', type=int, default=None)
    args = parser.parse_args()

    for path in args.labels:
        if not os.path.exists(path):
            print(f'\n=== {path}\n    MISSING (run not finished?)')
            continue
        labels, counts, frac, nonempty, even, k = describe(path, args.n_clusters)
        print(f'\n=== {path}')
        print(f'    n={labels.size}  nominal_k={k}  non-empty clusters={nonempty}/{k}  '
              f'evenness={even:.3f}  largest={frac.max():.1%}  smallest={frac.min():.1%}')
        order = np.argsort(-counts)
        print('    ' + '  '.join(f'c{i}:{counts[i]}({frac[i]:.1%})' for i in order))
        starved = [i for i in range(k) if frac[i] < 0.02]
        if starved:
            print(f'    below 2% of the data: {starved}')


if __name__ == '__main__':
    main()
