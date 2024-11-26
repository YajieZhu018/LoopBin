# violin plot of the loop length in each cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/01_sorted/'
ol =  '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/02_plot/'
conds = ['control', 'degron']
dic_data = {}
for cond in conds:
    in_file = f'{il}{cond}_loops_labels_genes_log2FC.bed'
    # read into a dataframe
    df = pd.read_csv(in_file, sep='\t')
    # add cond
    df['cond'] = cond
    # save into a dic
    dic_data[cond] = df
# concatenate two df
df = pd.concat([dic_data['control'], dic_data['degron']], ignore_index=True)
# save into tsv
df.to_csv(f'{il}loops_labels_genes_log2FC.tsv',sep='\t',index=False)
# violin plot
ax = sns.violinplot(x='cluster', y=f'log2FC', hue='cond', data=df)
plt.savefig(f'{ol}violinplot_loops_clusters_log2FC.pdf')
plt.close()



    
