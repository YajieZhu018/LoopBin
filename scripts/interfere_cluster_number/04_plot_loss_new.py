import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # it allows appending without initializing empty list
output_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/calculate_g/control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/'
json_file_name = f'{output_path}g_vs_actual_num_clusters.json'
with open(json_file_name) as json_file:
    new_dict = json.load (json_file)
# plotting training and testing error
for subkey in [' kl ', ' ']:
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