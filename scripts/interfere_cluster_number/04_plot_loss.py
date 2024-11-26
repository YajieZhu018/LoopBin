import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # it allows appending without initializing empty list
output_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/calculate_g/'
json_file_name = f'{output_path}generalizability.json'
with open(json_file_name) as json_file:
    data = json.load (json_file)
new_dict = defaultdict(list)
new_dict['test recon loss'] = defaultdict(list)
new_dict['train recon loss'] = defaultdict(list)
# Iterate over each cluster count
for cluster_num, losses in data['train recon loss'].items():
    n_clusters = data['N cluster'][cluster_num][0]
    loss = losses[0]
    # For each N cluster, append the corresponding loss
    for n, loss in zip(n_clusters, loss):
        new_dict['train recon loss'][n].append(loss)
for cluster_num, losses in data['test recon loss'].items():
    n_clusters = data['N cluster'][cluster_num][0]
    loss = losses[0]
    # For each N cluster, append the corresponding loss
    for n, loss in zip(n_clusters, loss):
        new_dict['test recon loss'][n].append(loss)       
# Convert defaultdict back to a regular dictionary (optional)
new_dict['test recon loss'] = dict(sorted(new_dict['test recon loss'].items()))
new_dict['train recon loss'] = dict(sorted(new_dict['train recon loss'].items()))
new_dict = dict(new_dict)
# save
out_file_name = f'{output_path}recon_loss_vs_actual_num_cluster.json'
with open(out_file_name, 'w') as json_file:
        json.dump(new_dict, json_file, indent=4)
# plotting training and testing error
for subkey in [' recon ']:
    dic_color = {f'train{subkey}loss': 'b', f'test{subkey}loss':'r'}
    for err in [f'train{subkey}loss', f'test{subkey}loss']:
        x = list(new_dict[err].keys())  # Keys as x-axis labels
        y_means = [np.mean(values) for values in new_dict[err].values()]  # Mean of each list
        y_stds = [np.std(values) for values in new_dict[err].values()]    # Standard deviation of each list
        plt.errorbar(x, y_means, yerr=y_stds, fmt='o', capsize=5, capthick=2, marker='s', linestyle='-', color=dic_color[err], label=f'{err.capitalize()}')
    plt.xlabel("N of actual clusters")
    plt.ylabel(f'{subkey}loss')
    plt.title("loss with standard deviation of each cluster number")
    plt.legend()
    subname=subkey.strip()
    plt.savefig(f'{output_path}{subname}_loss_vs_actual_num_cluster.pdf')
    plt.close()