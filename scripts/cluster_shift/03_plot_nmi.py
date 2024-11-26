import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
# args
data_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_with_added_noises'
output_path = '/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_with_added_noises/'
# Load label files
labels = [np.load(f'{data_path}_run{i}/labels.npy').astype(int) for i in range(1, 6)]
# Initialize a 5x5 matrix for storing NMI values
nmi_matrix = np.zeros((5, 5))
# Calculate pairwise NMI
for i in range(5):
    for j in range(5):
        # Compute NMI between labels from run i and run j
        nmi_matrix[i, j] = normalized_mutual_info_score(labels[i], labels[j])
# Plot the NMI matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(nmi_matrix, annot=True, cmap='viridis', xticklabels=[f'Run {i+1}' for i in range(5)], yticklabels=[f'Run {i+1}' for i in range(5)])
plt.title("Pairwise Normalized Mutual Information (NMI) Between Runs")
plt.xlabel("Runs")
plt.ylabel("Runs")
plt.savefig(f'{output_path}/nmi.pdf')
plt.close()