## make a heatmap to show the intersection number of each pair of clusters in two runs
import pandas as pd
import numpy as np  
import sys
# input
input_file = sys.argv[1] # chrom1, start1, end1, chrom2, start2, end2, cluster1, cluster2
output_file = sys.argv[2]
# read the input file
data = pd.read_csv(input_file, sep='\t', header=None)
# get the number of clusters
cluster1 = np.sort(data[6].unique())
cluster2 = np.sort(data[7].unique())
# create a matrix of (cluster1, cluster2) to store the intersection number of each pair of clusters
m = np.zeros((len(cluster1), len(cluster2)))
# fill the matrix
for i in range(len(cluster1)):
    for j in range(len(cluster2)):
        m[i,j] = len(data[(data[6] == cluster1[i]) & (data[7] == cluster2[j])]) 
# create a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
sns.heatmap(m, annot=True, fmt=".0f", xticklabels=cluster2, yticklabels=cluster1)
plt.xlabel('Cluster2')
plt.ylabel('Cluster1')
plt.title('Intersection number of each pair of clusters')
# output
plt.savefig(output_file)
plt.close()