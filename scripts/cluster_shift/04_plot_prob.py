import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
il = '/usr/users/yzhu1/LoopBin/trials/cluster_shift/vade_10clusters_merged_control_degron_rep1_cov_dia/'
inFile = f'{il}run1_to_run4.bedpe'
prob_file1 = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run1/control_rep1/prob.npy'
prob_file2 = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run4/control_rep1/prob.npy'
loops = pd.read_csv(inFile, header=None, sep="\t")
loops.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'cluster1', 'cluster2']
prob1 = np.load(prob_file1)
prob2 = np.load(prob_file2)
loops['prob1'] = list(np.max(prob1,axis = 1))
loops['prob2'] = list(np.max(prob2,axis = 1))
#cluster1 = np.array([0,1,5,3,2,4])
#cluster2= np.array([2,3,5,1,0,4,6])
pairs1 = [(0,2),(0,3),(1,5),(5,1),(3,0),(2,4),(4,6)]
pairs2 = [(0,5),(3,1),(4,4),(2,2),(4,2),(4,5)]
# set the column 'cond', if (cluster1, cluster2) is in pairs1, then 'cond' is 'same', if in pairs2, 'minor shift', otherwise 'cond' is 'major shifted'
loops['cond'] = 'major shifted'
for i in range(len(pairs1)):
    loops.loc[(loops['cluster1'] == pairs1[i][0]) & (loops['cluster2'] == pairs1[i][1]), 'cond'] = 'same'
for i in range(len(pairs2)):
    loops.loc[(loops['cluster1'] == pairs2[i][0]) & (loops['cluster2'] == pairs2[i][1]), 'cond'] = 'minor shift'
# create a violin plot
plt.figure(figsize=(8, 6))
#sns.boxplot(x='cond', y='prob1', data=loops)
sns.violinplot(x='cond', y='prob1', data=loops, cut=0)
# add the p-value
add_stat_annotation(ax=plt.gca(), data=loops, x='cond', y='prob1', box_pairs=[('same', 'minor shift'),('same', 'major shifted'),('minor shift', 'major shifted')], test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# Display the plot
plt.title('Prob of clusters')
# save as pdf
pdf = f'{il}violinplot_prob_clusters_prob_control_run_1_4.pdf'
plt.savefig(pdf)
plt.close()