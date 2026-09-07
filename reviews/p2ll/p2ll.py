"""
Compute P2LL (Peak-to-Low-Left) scores for Hi-C loop matrices.

Assumptions:
- micro.npy has shape (N, 256), where each row is a flattened 16x16 matrix.
- label.npy has shape (N,), containing labels 0-5.
- P2LL is defined as:

      center_peak / mean(lower_left_region)

  where:
    center_peak = matrix[8, 8] for a 16x16 matrix
    lower_left_region = matrix[12:16, 0:4]

If your lab uses a different P2LL definition, modify the
compute_p2ll() function accordingly.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ranksums


def compute_p2ll(mat):
    """Compute Peak-to-Low-Left score for one 16x16 Hi-C matrix."""
    peak = np.mean(mat[7:10, 7:10])

    low_left = mat[8:16, 0:8]
    low_left_mean = np.mean(low_left)

    if low_left_mean == 0:
        return np.nan

    return peak / low_left_mean


def main():
    args = sys.argv[1:]
    micro = np.load(args[0])      # (N, 256)
    labels = np.load(args[1])     # (N,)

    if micro.shape[1] != 256:
        raise ValueError(
            f"Expected micro.npy shape (N,256), got {micro.shape}"
        )

    if len(labels) != len(micro):
        #raise ValueError(
        #    f"Number of labels ({len(labels)}) != number of loops ({len(micro)})"
        #)
        labels = labels[:len(micro)]

    # --------------------------------------------------
    # (1) Compute P2LL for every loop
    # --------------------------------------------------
    p2ll_scores = []

    for row in micro:
        mat = row.reshape(16, 16)
        p2ll_scores.append(compute_p2ll(mat))

    p2ll_scores = np.array(p2ll_scores)

    np.save("p2ll_scores.npy", p2ll_scores)

    # --------------------------------------------------
    # (2) Merge with labels and save dataframe
    # --------------------------------------------------
    df = pd.DataFrame({
        "loop_id": np.arange(len(p2ll_scores)),
        "label": labels,
        "p2ll": p2ll_scores
    })

    df.to_csv("p2ll_with_labels.csv", index=False)

    # --------------------------------------------------
    # (3) Pairwise statistics among labels 0-5
    # --------------------------------------------------
    results = []

    unique_labels = sorted(np.unique(labels))

    for g1, g2 in itertools.combinations(unique_labels, 2):
        x = df.loc[df["label"] == g1, "p2ll"].dropna().values
        y = df.loc[df["label"] == g2, "p2ll"].dropna().values

        stat, pval = ranksums(x, y)

        results.append({
            "group1": int(g1),
            "group2": int(g2),
            "n1": len(x),
            "n2": len(y),
            "statistic": stat,
            "pvalue": pval
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("pairwise_p2ll_tests.csv", index=False)

    with open("pairwise_p2ll_tests.txt", "w") as f:
        f.write("Pairwise Wilcoxon rank-sum tests on P2LL scores\n")
        f.write("=" * 60 + "\n\n")

        for _, r in results_df.iterrows():
            f.write(
                f"Label {r['group1']} vs Label {r['group2']} | "
                f"n1={r['n1']} n2={r['n2']} | "
                f"stat={r['statistic']:.6f} | "
                f"p={r['pvalue']:.6e}\n"
            )

    print("Saved:")
    print("  p2ll_scores.npy")
    print("  p2ll_with_labels.csv")
    print("  pairwise_p2ll_tests.csv")
    print("  pairwise_p2ll_tests.txt")


if __name__ == "__main__":
    main()

