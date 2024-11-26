import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # it allows appending without initializing empty list
output_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/calculate_g/'
json_file_name = f'{output_path}generalizability.json'
with open(json_file_name) as json_file:
    data = json.load (json_file)
new_dict = defaultdict(list)
new_dict['g'] = defaultdict(list)
new_dict['g recon'] = defaultdict(list)
# Iterate over each cluster count
for cluster_num, losses in data['g'].items():
    n_clusters = data['N cluster'][cluster_num][0]
    loss = losses[0]
    # For each N cluster, append the corresponding loss
    for n, loss in zip(n_clusters, loss):
        new_dict['g'][n].append(loss)
for cluster_num, losses in data['g recon'].items():
    n_clusters = data['N cluster'][cluster_num][0]
    loss = losses[0]
    # For each N cluster, append the corresponding loss
    for n, loss in zip(n_clusters, loss):
        new_dict['g recon'][n].append(loss)       
# Convert defaultdict back to a regular dictionary (optional)
new_dict['g'] = dict(sorted(new_dict['g'].items()))
new_dict['g recon'] = dict(sorted(new_dict['g recon'].items()))
new_dict = dict(new_dict)
# save
out_file_name = f'{output_path}g_vs_actual_num_clusters.json'
with open(out_file_name, 'w') as json_file:
        json.dump(new_dict, json_file, indent=4)
# plotting g
dic_color = {'g': 'b', 'g recon':'r'}
for err in ['g', 'g recon']:
    x = list(new_dict[err].keys())  # Keys as x-axis labels
    y_means = [np.mean(values) for values in new_dict[err].values()]  # Mean of each list
    y_stds = [np.std(values) for values in new_dict[err].values()]    # Standard deviation of each list
    plt.errorbar(x, y_means, yerr=y_stds, fmt='o', capsize=5, capthick=2, marker='s', linestyle='-', color=dic_color[err], label=f'{err.capitalize()}')
plt.xlabel("N of actual clusters")
plt.ylabel('g')
plt.title("g with standard deviation of each cluster number")
plt.legend()
plt.savefig(f'{output_path}g_vs_actual_num_cluster.pdf')
plt.close()