# make a heatmap of the number of overlapped loops between control and degron clusters
import numpy as np
import os
# input folder
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/02_intersect/'
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/03_plot/'
# create output folder if it does not exist
if not os.path.exists(ol):
    os.makedirs(ol)
# get the line number in each rep1_loops_control_"i"_degron_"j".bed and save the number in to the entry row i and column j in the matrix
def get_num(file):
    with open(file) as f:
        return len(f.readlines())
# save the number to the matrix
def save_num(matrix, i, j, num):
    matrix[i][j] = num
# loop through i and j
def loop_ij(matrix, a, b):
    for i in range(a):
        for j in range(b):
            file = il + 'loops_control_' + str(i+1) + '_degron_' + str(j+1) + '.bed'
            num = get_num(file)
            save_num(matrix, i, j, num)
# create the matrix
m = np.zeros((6, 6))
# calculate the number of overlapped loops between control and degron clusters
loop_ij(m, 6, 6)
# plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt
# plot the whole matrix
sns.heatmap(m, annot=True, fmt='g', cmap='viridis')
# label 1-6 to the center
plt.xticks(np.arange(6) + 0.5, np.arange(1, 7))
plt.yticks(np.arange(6) + 0.5, np.arange(1, 7))
# add control as y label and degron as x label
plt.ylabel('Control')
plt.xlabel('Degron')
# save to the output folder
plt.savefig(ol + 'heatmap_viridis.pdf')
plt.close()