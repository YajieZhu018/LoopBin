# violin plot of the loop length in each cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import sys
folder = sys.argv[1]
il = f'{folder}unique/01_unique/'
ol =  f'{folder}unique/02_plot/'
conds = ['control', 'degron']
dic_data = {}
for cond in conds:
    in_file = f'{il}{cond}_unique_loops_labels_log2FC.bedpe'
    # read into a dataframe
    df = pd.read_csv(in_file, sep='\t')
    # add cond
    df['cond'] = cond
    # save into a dic
    dic_data[cond] = df
# concatenate two df
df = pd.concat([dic_data['control'], dic_data['degron']], ignore_index=True)
# save into tsv
df.to_csv(f'{il}unique_loops_labels_genes_log2FC.tsv',sep='\t',index=False)
# violin plot
ax = sns.violinplot(x='cluster', y=f'log2FC', hue='cond', data=df)
n = 6
box_pairs = [((i,'control'),(i,'degron')) for i in range(n)]
order = [i for i in range(n)]
add_stat_annotation(ax, data=df, x="cluster", y="log2FC", hue="cond",
                    box_pairs=box_pairs,
		            test='Mann-Whitney',order = order,text_format='star', loc='inside', verbose=2)
plt.savefig(f'{ol}violinplot_unique_loops_clusters_log2FC.pdf')
plt.close()



    
