# make heatmap of percentage of each cluster in the matched labels
import numpy as np
# plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/01_sorted/'
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/03_plot/'
# create a 6x3 matrix
matrix = np.zeros((6,3))
#matrix_over_clusters = np.zeros((6,3))
matrix_over_manual_labels = np.zeros((6,3))
for cond in ['control','degron']:
    # calculate the number of line whose 7th and 8th columns are 'T' and 'T', or 'T' and 'F'/ 'F' and 'T' or 'F' and 'F'
    with open(f'{il}{cond}_labels_loops.bedpe','r') as f:
        lines = f.readlines()
        for line in lines:
            if cond=='control':
                i = int(line.split()[10])
            else:
                i = int(line.split()[8])
            if line.split()[6] == 'T' and line.split()[7] == 'T':
                matrix[i][0] += 1
            elif (line.split()[6] == 'T' and line.split()[7] == 'F') or (line.split()[6] == 'F' and line.split()[7] == 'T'):
                matrix[i][1] += 1
            else:
                matrix[i][2] += 1
    ## calculate the percentage of each cluster in 'TT','TF' and 'FF'
    #for i in range(3):
    #    matrix_over_clusters[:,i] = matrix[:,i]/np.sum(matrix[:,i])
    #sns.heatmap(matrix_over_clusters,square=True,annot=True,cmap='coolwarm',xticklabels=['TT','TF','FF'],yticklabels=[f'cluster{i}' for i in range(1,7)])
    ## save the heatmap
    #plt.savefig(f'{ol}{cond}_heatmap_perc_over_clusters.pdf')
    #plt.close()
    # calculate the percentage of 'TT','TF' and 'FF' in each cluster
    for i in range(matrix.shape[0]):
        matrix_over_manual_labels[i] = matrix[i]/np.sum(matrix[i])
    sns.heatmap(matrix_over_manual_labels,square=True,annot=True,cmap='coolwarm',vmin=0.1, vmax=0.55, xticklabels=['TT','TF','FF'],yticklabels=[f'cluster{i}' for i in range(matrix.shape[0])])
    # save the heatmap
    plt.savefig(f'{ol}{cond}_heatmap_CTCF_perc_over_manual_labels.pdf')
    plt.close()
    ## plot counts
    #sns.heatmap(matrix, square=True,annot=True, fmt='g', cmap='coolwarm', xticklabels=['TT', 'TF', 'FF'], yticklabels=[f'cluster{i}' for i in range(1, 7)])
    ## save the heatmap
    #plt.savefig(f'{ol}{cond}_heatmap_counts_manual_labels.pdf')
    #plt.close()


