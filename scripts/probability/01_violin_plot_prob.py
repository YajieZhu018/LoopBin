import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
il_prefix = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_run'
for i in range(1,6):
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
    # Create a violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='clusters', y='probability', data=df)
    # Display the plot
    plt.title('Prob of clusters')
    # save as pdf
    pdf = f'{il}prob_clusters.pdf'
    plt.savefig(pdf)
    plt.close()