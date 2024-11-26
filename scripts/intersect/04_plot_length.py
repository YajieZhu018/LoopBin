# violin plot of the loop length in each cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/01_sorted/'
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/03_plot/'
conds = ['control', 'degron']
# create a dataframe with column 'condition', 'loop length' and 'cluster number' to store the loop length
df = pd.DataFrame(columns=['condition', 'log_length', 'cluster'])
# loop through control and degron
for cond in conds:
    # loop through the clusters
    for i in range(1,7):
        file = f'{il}{cond}_loops_label_{str(i)}.bed'
        # get the loop length as 5th column - 2th column
        loops = pd.read_csv(file, sep='\t', header=None)
        # append the log length to the dataframe 'length'; repeat cond len(loops) times
        df = df.append(pd.DataFrame({'condition': [cond]*len(loops), 'log_length': np.log(loops[4]-loops[1]), 'cluster': [str(i)]*len(loops)}))
# output df as csv
df.to_csv(f'{ol}loop_length.csv', index=False)
# plot the violin of the length with cluster as x-axis and condition as hue
ax = sns.violinplot(x='cluster', y='log_length', hue='condition', data=df)
box_pairs = [((str(i),'control'),(str(i),'degron')) for i in range(1,7)]
order = [str(i) for i in range(1,7)]
add_stat_annotation(ax, data=df, x="cluster", y="log_length", hue="condition",
                    box_pairs=box_pairs,
		    test='Mann-Whitney-ls',order = order,text_format='star', loc='inside', verbose=2) #stats_params={'alternative': "two-sided"} 
plt.savefig(f'{ol}loop_length_violin_mann_whitney_less.pdf')
plt.close()



    
