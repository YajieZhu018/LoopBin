import numpy as np
il_prefix = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_with_added_noises_run'
for i in range(1,6):
    il = f'{il_prefix}{str(i)}/'
    # probability file
    prob_file = f'{il}prob.npy'
    prob = np.load(prob_file)
    # get the max prob
    max_prob = np.max(prob,axis = 1)
    # get the cluster whose prob > 0.95 as boolean vector
    boolean_vector = (max_prob > 0.95)
    # labeled loop file
    loop_file = f'{il}labels_loops.bedpe'
    # read loop file
    with open(loop_file, 'r') as file:
        lines = file.readlines()
    # filter loops whose prob > 0.95 as True values in the boolean vector
    filtered_lines = [line for i, line in enumerate(lines) if boolean_vector[i]]
    # write the filtered lines to a new file 
    filtered_loop_file = f'{il}labels_loops_prob0.95.bedpe'
    with open(filtered_loop_file, 'w') as outfile:
        outfile.writelines(filtered_lines)