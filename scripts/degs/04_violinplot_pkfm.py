# violin plot of the loop length in each cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import sys
folder = sys.argv[1]
ol = f'{folder}degs/'
conds = ['control', 'degron']
for cond in conds:
    in_file = f'{ol}{cond}_loops_labels_genes_fkpm.bed'
    # read into a dataframe
    df = pd.read_csv(in_file, sep='\t')
    # set 99 percentile to cap extreme values on fpkm
    cap = df[f'fpkm_{cond}'].quantile(0.99)
    df.loc[df[f'fpkm_{cond}'] > cap, f'fpkm_{cond}'] = cap
    # apply log2 to fpkm
    df[f'fpkm_{cond}'] = np.log2(df[f'fpkm_{cond}']+1)
    # violin plot 'pkfm' vs cluster
    ax = sns.violinplot(x='cluster', y=f'fpkm_{cond}', data=df)
    # set y-axis label
    ax.set_ylabel(f'log2(fpkm_{cond})')
    plt.savefig(f'{ol}{cond}_log2_pkfm_cap0.99_violin.pdf')
    plt.close()



    
