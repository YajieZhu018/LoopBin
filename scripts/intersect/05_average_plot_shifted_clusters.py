# average plot of cluster 1 and 3 in control that shift to cluster 5
# loop through control and degron
# set the plot: 4 rows, 5 columns; 
import os
import sys
sys.path.append(os.path.expanduser('~/LoopBin/src/fn'))
import processing
import numpy as np
import matplotlib.pyplot as plt
il = sys.argv[1]
ol = f'{il}intersect/03_plot/average_plot_cluster_shift/'
for j,k in [('0','2')]: #[('0','1'), ('1','4'), ('2','4'), ('4','2'), ('5','2'), ('5','3')]:
    images = []
    vmin_loop = []
    vmax_loop = []
    for cond in ['control', 'degron']: # 
        # processed data
        processed_data_file = f'/usr/users/yzhu1/LoopBin/data/02_process/{cond}_rep1/log_data.npy'
        processed_data = np.load(processed_data_file)
        # bed file of shared loops with shifted clusters
        bed_cluster_file = f'{il}intersect/02_intersect/{cond}_intersect.bedpe'
        # bed files of all loops
        bed_file = f'{il}{cond}_rep1/labels_loops.bedpe'
        # read in to lists
        with open(bed_cluster_file,'r') as file:
            bed_cluster = file.read().splitlines()
        with open(bed_file, 'r') as file:
            bed = file.read().splitlines()
        # get element of bed into a list
        bed_dict = {tuple(element.split("\t")[0:6]):ind for ind, element in enumerate(bed)}
        # keep the indices from bed
        indices = [bed_dict[tuple(element.split("\t")[0:6])] for element in bed_cluster if (tuple(element.split("\t")[0:6]) in bed_dict) and (element.split("\t")[6]==j) and element.split("\t")[7]==k]
        # get the data with the indices
        data = processed_data[indices]
        # make it two dimension
        ori_micro_c = data[:,:256]
        ori_epigenetic = data[:,256:]
        x_data = processing.create_data(ori_epigenetic, ori_micro_c)
        # average plot over the 0 axis for each channel at the 1 axis
        a = np.mean(x_data, axis=0)
        vmin_sub = []
        vmax_sub = []
        for i in range(a.shape[2]):
            images.append(a[:,:,i])
            vmin_sub.append(np.min(a[:,:,i]))
            vmax_sub.append(np.max(a[:,:,i]))
        vmin_loop.append(vmin_sub)
        vmax_loop.append(vmax_sub)# show colorbar
    vmin_1 = np.min(vmin_loop,axis=0)
    vmax_1 = np.max(vmax_loop,axis=0)
    vmax_1 = [0.24,0.04,0.04,0.1,0.04]
    # use the same color bar as in the cluster plot
    #vmax_1 = [0.2,0.04,0.025,0.22,0.04]
    # plot each image in a_split to plot five images in one row of the same plot
    # Create a figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    # Loop through the images in a_split and plot them in the corresponding axes
    for i, image in enumerate(images):
        row = i // 5  # Calculate the row index
        col = i % 5   # Calculate the column index
        axes[row, col].imshow(image, cmap="jet", vmin=vmin_1[col], vmax=vmax_1[col])  # Plot the image in the corresponding axe
    # set the title for each column
    titles = ["Micro-C", "CTCF", "H3K27ac", "H3K27me3", "SMC1A"]
    for i, title in enumerate(titles):
        axes[0, i].set_title(title)
    # set the ylabel for each row
    groups = [f'control_{j}', f'degron_{k}']
    for i, group in enumerate(groups):
        axes[i, 0].set_ylabel(group, rotation=0, labelpad=35)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # save
    plt.savefig(f'{ol}average_plot_cluster_control_{j}_degron_{k}.pdf')
    plt.close()