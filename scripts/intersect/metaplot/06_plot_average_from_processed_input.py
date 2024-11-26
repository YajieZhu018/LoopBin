"""
Average plot of CUT&TAG from LoopBin processed data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
# create list to store average vectors and vmax
averages = []
labels = []
il = '/usr/users/yzhu1/LoopBin/result/intersect/temp/'
ol = '/usr/users/yzhu1/LoopBin/result/intersect/03_plot/'
# get the cut&tag data
# get the cut&tag data
for j in ['1', '3']:
    for cond in ['control', 'degron']:
        filename = f'/usr/users/yzhu1/LoopBin/result/intersect/temp/{cond}_cluster_{j}_5/save/log/log_epigenetic.npy'
        data = np.load(filename)
        # calcualte mean value
        data_average = np.mean(data,axis=0)
        # create list to store the max value of each condition
        sub_vmaxs = []
        # separate cut&tag
        for i in range(4):
            # merge two anchors
            sub_average = (data_average[i*32: i*32 + 16] + data_average[i*32 + 16: (i+1)*32]) / 2
            averages.append(sub_average)
            sub_vmaxs.append(np.max(sub_average))
        labels.append(f'{cond}_cluster_{j}_5')
# add a smooth function
smoothed_data_list = [gaussian_filter1d(data, sigma=2) for data in averages]
# Create a figure with a 3x4 grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
colors = ['blue', 'green', 'red', 'purple']
marks = ['CTCF','H3K27ac','H3K27me3','SMC1A']
for i, ax in enumerate(axes.flat):
    for j in range(4):
        ax.plot(smoothed_data_list[i+j*4], color = colors[j % 4],label = labels[j])
    ax.set_title(marks[i])
ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
pdf = f'{ol}average_plot_processed_input.pdf'
plt.savefig(pdf)
plt.close()

