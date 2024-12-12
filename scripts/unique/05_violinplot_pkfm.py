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
for cond in conds:
    dic_data = {}
    in_file = f'{il}{cond}_unique_loops_labels_fkpm.bedpe'
    # read into a dataframe
    df = pd.read_csv(in_file, sep='\t')
    # set 99 percentile to cap extreme values on fpkm
    #cap = df[f'fpkm_{cond}'].quantile(0.99)
    #df.loc[df[f'fpkm_{cond}'] > cap, f'fpkm_{cond}'] = cap
    for subcond in conds:
        sub_df = pd.DataFrame()
        # apply log2 to fpkm
        sub_df['log2fpkm'] = np.log2(df[f'fpkm_{subcond}']+1)
        sub_df['cond'] = subcond
        sub_df['cluster'] = df['cluster']
        dic_data[subcond] = sub_df
    #ax = sns.violinplot(x='cluster', y=f'fpkm_{cond}', data=df)
    ## set y-axis label
    #ax.set_ylabel(f'log2(fpkm_{cond})')
    #plt.savefig(f'{ol}{cond}_log2_pkfm_cap0.99_violin.pdf')
    #plt.close()
    # concatenate two df
    df = pd.concat([dic_data['control'], dic_data['degron']], ignore_index=True)
    # violin plot
    ax = sns.violinplot(x='cluster', y=f'log2fpkm', hue='cond', data=df)
    n = 6
    box_pairs = [((i,'control'),(i,'degron')) for i in range(n)]
    order = [i for i in range(n)]
    add_stat_annotation(ax, data=df, x="cluster", y="log2fpkm", hue="cond",
                    box_pairs=box_pairs,
		            test='Mann-Whitney',order = order,text_format='star', loc='inside', verbose=2)
    plt.savefig(f'{ol}violinplot_{cond}_unique_loops_clusters_log2fpkm.pdf')
    plt.close()



    
