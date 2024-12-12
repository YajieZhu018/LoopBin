# barplot the percentage of loops with promoters in each cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# get the number of all loops in each cluster calculating the frequency of cluster in the 7th column of the loop file
# in the directory of the loop files
folder = sys.argv[1]
il = f'{folder}degs/' 
# loop through control and degron
for cond in ["control", "degron"]:
    loop_file = f'{folder}intersect/01_sorted/{cond}_labels_loops_formatted.bedpe'
    # read into a dataframe
    df = pd.read_csv(loop_file, sep='\t', header=None)
    # get the frequency of each cluster in a dictionary
    cluster_freq = df[6].value_counts().to_dict()
    # calculate the loops without promoters in each cluster
    loop_no_promoter_file = f'{il}{cond}_loops_labels_no_genes.bed'
    # read into a dataframe
    df = pd.read_csv(loop_no_promoter_file, sep='\t', header=None)
    # get the frequency of each cluster
    cluster_freq_no_promoter = df[6].value_counts().to_dict()
    # calculate the percentage of loops with promoters in each cluster
    cluster_perc = {}
    for cluster in cluster_freq:
        cluster_perc[cluster] = 1 - cluster_freq_no_promoter[cluster] / cluster_freq[cluster]
    # write the percentage of loops with promoters in each cluster to a file
    out_file = f'{il}{cond}_loops_labels_perc_promoter.txt'
    with open(out_file, 'w') as out:
        # add header
        out.write('cluster\tperc\n')
        for cluster in cluster_perc:
            out.write(f'{cluster}\t{cluster_perc[cluster]}\n')
    # barplot the percentage of loops with promoters in each cluster
    # read into a dataframe
    df = pd.read_csv(out_file, sep='\t')
    # barplot
    ax = sns.barplot(x='cluster', y='perc', data=df)
    # set y-axis label
    ax.set_ylabel('percentage of loops with promoters')
    plt.savefig(f'{il}{cond}_loops_labels_perc_promoter.pdf')
    plt.close()


