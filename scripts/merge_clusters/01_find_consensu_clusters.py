import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# import data
data_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1'
out_path = '/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1/'
results = [np.load(f'{data_path}_run{i}/labels.npy') for i in range(1, 6)]
# Assuming `results` is a list of 5 arrays, each containing the cluster labels of one run
n_samples = len(results[0])
n_runs = len(results)

# Step 1: Create the co-occurrence matrix
co_occurrence_matrix = np.zeros((n_samples, n_samples))

# Vectorized calculation for the co-occurrence matrix
for result in results:
    # Create an indicator matrix where each element is 1 if two samples share a cluster
    indicator_matrix = (result[:, None] == result[None, :]).astype(float)
    co_occurrence_matrix += indicator_matrix

# Normalize by the number of runs
co_occurrence_matrix /= n_runs

# Step 2: Apply hierarchical clustering on the co-occurrence matrix
# Use linkage on the co-occurrence matrix to create the hierarchy
Z = linkage(co_occurrence_matrix, method='average')

# Define the final number of clusters (e.g., 6) or use a distance threshold
final_clusters = AgglomerativeClustering(
    n_clusters=7, affinity='precomputed', linkage='average'
    ).fit_predict(1 - co_occurrence_matrix)  # 1 - matrix as dissimilarity
# save final clusters
np.save(f'{out_path}consensus_cluster.npy', final_clusters)

# Optional: Plot the dendrogram
# Sample a subset of data
sample_size = 5000  # Adjust based on your needs
indices = np.random.choice(len(Z), sample_size, replace=False)
sampled_Z = linkage(Z[indices], method='average')
plt.figure(figsize=(10, 7))
dendrogram(sampled_Z)
plt.title("Dendrogram of Consensus Clustering")
plt.xlabel("Sample index")
plt.ylabel("Co-occurrence distance")
plt.savefig(f'{out_path}consensus_clusters.pdf')

