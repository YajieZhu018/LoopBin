# barplot the percentage of unique loops with down/up regulated genes in each cluster; compare control vs degron
import pandas as pd
# csv with fold changes of degs
degs_file = '/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_DLD1_async_factory_gene_irnaseq_GRCh38_qval0.05n.csv'
# read into a dataframe; removing quatoation marks
degs = pd.read_csv(degs_file, quotechar='"')
# get the gene names of those with log2foldchange > 0 into a list 'up'
up_genes = degs[degs['log2FoldChange'] > 0]['gene_name'].tolist()
# get the gene names of those with log2foldchange < 0 into a list 'down'
down_genes = degs[degs['log2FoldChange'] < 0]['gene_name'].tolist()
# create a dictionary to store the data
data = {}
# loop conditions
for cond in ['control', 'degron']:
    # loop files
    loop_file = f'/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/01_sorted/{cond}_loops_label_genes.bed'
    # read into a dataframe without header
    loops = pd.read_csv(loop_file, header=None, sep='\t')
    # get the total number of loops in each cluster; the cluster number is the 8th column
    total_loops = loops[7].value_counts().sort_index()
    # filter the loops with up genes into a new dataframe
    up_loops = loops[loops[9].isin(up_genes)]
    # get counts in each cluster
    up_counts = up_loops[7].value_counts().sort_index()
    # calculate the percentage of loops with up genes in each cluster
    up_percentage = up_counts / total_loops * 100
    # filter the loops with down genes into a new dataframe
    down_loops = loops[loops[9].isin(down_genes)]
    # get counts in each cluster
    down_counts = down_loops[7].value_counts().sort_index()
    # calculate the percentage of loops with down genes in each cluster
    down_percentage = down_counts / total_loops * 100
    # save the down and up into a dictionary
    data[cond] = {'up': up_percentage, 'down': down_percentage}
# plot the dictionary; up and down as subplot left and right; cluster as x-axis; 
# conditions as color legend; two conditions of the same cluster stay together for better comparison
import matplotlib.pyplot as plt
import numpy as np
for de in ['down', 'up']:
    # set bar width
    barWidth = 0.25
    # set the x locations
    x = np.arange(len(data['control'][de]))
    # set the figure size
    plt.figure(figsize=(15, 5))
    # plot the bars
    plt.bar(x, data['control'][de], color='b', width=barWidth, label='control')
    plt.bar(x + barWidth, data['degron'][de], color='r', width=barWidth, label='degron')
    # set the x ticks
    plt.xticks(x + barWidth / 2, data['control'][de].index)
    # set the x label
    plt.xlabel('Cluster')
    # set the y label
    plt.ylabel(f'Percentage of loops')
    # set the title
    plt.title(f'Percentage of loops with {de} regulated genes in each cluster')
    # set the legend
    plt.legend()
    # save the plot
    plt.savefig(f'/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/degs/barplot_unique_loops_degs_{de}.pdf')
