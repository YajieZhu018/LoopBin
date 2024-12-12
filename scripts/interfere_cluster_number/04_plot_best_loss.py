# plot the mininum loss of each model
import json
import numpy as np
import matplotlib.pyplot as plt
output_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/calculate_g/merged_control_degron_rep1/'
json_file_name = f'{output_path}g_vs_actual_num_clusters.json'
with open(json_file_name) as json_file:
    data = json.load (json_file)
# get the minimum g in data['g'][cluster] for all cluster number
for key in ['g', 'g kl', 'g recon']:
    max_g = []
    for cluster_num, g in data[key].items():
        max_g.append(max(g))
    # plot the minimum g of each model
    plt.plot(list(data['g'].keys()), max_g, 'o-', label='max g')
    plt.xlabel("N of actual clusters")
    plt.ylabel(f'max {key}')
    plt.title(f'max {key} of each model')
    plt.legend()
    plt.savefig(f'{output_path}max_{key}_vs_num_cluster.pdf')
    plt.close()
