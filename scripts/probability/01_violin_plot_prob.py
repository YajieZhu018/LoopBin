import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
il_prefix = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run'
for i in range(1,2):
    il = f'{il_prefix}{str(i)}/'
    inFile = f'{il}prob.npy'
    prob = np.load(inFile)
    # get the max prob
    max_prob = np.max(prob,axis = 1)
    # get the index of the max prob
    clusters = np.argmax(prob, axis = 1)
    # create a panda frame
    df = pd.DataFrame({
        'probability': max_prob,
        'clusters': clusters
    })
    df["quantile"] = pd.qcut(df["probability"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    # Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='clusters', y='probability', hue = 'quantile', data=df)
    # Display the plot
    plt.title('Prob of clusters')
    # save as pdf
    pdf = f'{il}prob_clusters_with_quantile.pdf'
    plt.savefig(pdf)
    plt.close()