# make heatmap of percentage of each cluster in the matched labels
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/01_sorted/'
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/intersect/03_plot/'
# create a 6x3 matrix
matrix = np.zeros((6,3))
#matrix_over_clusters = np.zeros((6,3))
matrix_over_manual_labels = np.zeros((6,3))
for cond in ['control']:
    # calculate the number of line whose 9th and 10th columns are 'T' and 'T', or 'T' and 'F'/ 'F' and 'T' or 'F' and 'F'
    with open(f'{il}{cond}_labels_loops.bedpe','r') as f:
        lines = f.readlines()
        for line in lines:
            i = int(line.split()[10])
            if line.split()[8] == 'T' and line.split()[9] == 'T':
                matrix[i][0] += 1
            elif (line.split()[8] == 'T' and line.split()[9] == 'F') or (line.split()[8] == 'F' and line.split()[9] == 'T'):
                matrix[i][1] += 1
            else:
                matrix[i][2] += 1
    
    ## plot heatmap
    #sns.heatmap(matrix,square=True,annot=True,fmt='g',cmap='coolwarm',xticklabels=['TT','TF','FF'],yticklabels=[f'cluster{i}' for i in range(1,7)])
    ## save the heatmap
    #plt.savefig(f'{ol}{cond}_heatmap_H3K27ac_counts_manual_labels.pdf')
    #plt.close()

    ## calculate the percentage of each cluster in 'TT','TF' and 'FF'
    #for i in range(3):
    #    matrix_over_clusters[:,i] = matrix[:,i]/np.sum(matrix[:,i])
    #sns.heatmap(matrix_over_clusters,square=True,annot=True,cmap='coolwarm',xticklabels=['TT','TF','FF'],yticklabels=[f'cluster{i}' for i in range(1,7)])
    ## save the heatmap
    #plt.savefig(f'{ol}{cond}_heatmap_H3K27ac_perc_over_clusters.pdf')
    #plt.close()
    
    # calculate the percentage of 'TT','TF' and 'FF' in each cluster
    for i in range(matrix.shape[0]):
        matrix_over_manual_labels[i] = matrix[i]/np.sum(matrix[i])
    sns.heatmap(matrix_over_manual_labels,square=True,annot=True,cmap='coolwarm',xticklabels=['TT','TF','FF'],yticklabels=[f'cluster{i}' for i in range(matrix.shape[0])])
    plt.savefig(f'{ol}{cond}_heatmap_H3K27ac_perc_over_manual_labels.pdf')
    plt.close()