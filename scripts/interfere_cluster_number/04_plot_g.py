import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # it allows appending without initializing empty list
output_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/calculate_g/merged_control_degron_rep1/'
json_file_name = f'{output_path}g_vs_actual_num_clusters.json'
with open(json_file_name) as json_file:
    new_dict = json.load (json_file)
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