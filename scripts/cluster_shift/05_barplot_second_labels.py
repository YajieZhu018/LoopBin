import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statannot import add_stat_annotation
il = '/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/'
ol = '/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/second_labels/'
# make dir
os.makedirs(ol, exist_ok=True)
inFile = f'{il}run1_to_run4.bedpe'
prob_file1 = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/control_rep1/prob.npy'
prob_file2 = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run4/control_rep1/prob.npy'
loops = pd.read_csv(inFile, header=None, sep="\t")
loops.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'cluster1', 'cluster2']
prob1 = np.load(prob_file1)
prob2 = np.load(prob_file2)
# get the index of the second maximum value
second_max_index1 = np.argsort(-prob1, axis=1)[:, 1]
second_max_index2 = np.argsort(-prob2, axis=1)[:, 1]
loops['second_label1'] = second_max_index1
loops['second_label2'] = second_max_index2
#cluster1 = np.array([0,1,5,3,2,4])
#cluster2= np.array([2,3,5,1,0,4,6])
#pairs1 = [(0,2),(0,3),(1,5),(5,1),(3,0),(2,4),(4,6)]
old_pairs = [(0,5),(3,1),(4,4),(2,2),(4,2),(4,5)]
# rename the clusters with the dict
dict_cluster1 = {0: 'active dot', 1: 'active domain', 5: 'repressive dot', 3: 'repressive domain', 2: 'lonely dot', 4: 'lonely domain'}
dict_cluster2 = {2: 'weak active dot', 3: 'strong active dot', 5: 'active domain', 1: 'repressive dot', 0: 'repressive domain', 4: 'lonely dot', 6: 'lonely domain'}
loops['cluster1'] = loops['cluster1'].map(dict_cluster1)
loops['cluster2'] = loops['cluster2'].map(dict_cluster2)
loops['second_label1'] = loops['second_label1'].map(dict_cluster1)
loops['second_label2'] = loops['second_label2'].map(dict_cluster2)
# map pairs
pairs = [(dict_cluster1[cluster1], dict_cluster2[cluster2]) for (cluster1, cluster2) in old_pairs]
# for (cluster1, cluster2) in each pair of pairs, boxplot the percentage of counts of second_label1
for (cluster1, cluster2) in pairs:
    subset = loops[(loops['cluster1'] == cluster1) & (loops['cluster2'] == cluster2)]
    for key in ['second_label1', 'second_label2']:
        # get the second labels and counts of each second label 
        counts = subset[key].value_counts(normalize=True)
        counts = counts.reset_index()
        counts.columns = ['second_label', 'percentage']
        # barplot
        plt.figure(figsize=(8, 6))
        sns.barplot(x='second_label', y='percentage', data=counts)
        plt.title(f'Boxplot for Cluster {cluster1} and Cluster {cluster2}')
        pdf = f'{ol}boxplot_cluster_{cluster1}_{cluster2}_{key}.pdf'
        plt.savefig(pdf)
        plt.close()