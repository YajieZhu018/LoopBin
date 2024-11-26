# violin plot of the loop length in each cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/02_intersect/'
ol =  '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/03_plot/'
cond = 'control'
in_file = f'{il}{cond}_loops_genes_log2FC.bed'
# read into a dataframe
df = pd.read_csv(in_file, sep='\t')
# violin plot
ax = sns.violinplot(x='cluster', y=f'log2FC', data=df)
plt.savefig(f'{ol}violinplot_loops_shifted_clusters_log2FC.pdf')
plt.close()



    
