from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

# load data
output_file = '/usr/users/yzhu1/LoopBin/trials/consensus_cluster/sil_scores.txt'  # copy and paste to the output
data_file = '/usr/users/yzhu1/LoopBin/data/02_process/merged_log_data.npy'
#data_file = '/usr/users/yzhu1/LoopBin/data/02_process/control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/merged_log_data.npy'
x = np.load(data_file)
for i in range(8,9):
    model_file = f'/usr/users/yzhu1/LoopBin/trials/saved_models/vade_10clusters_merged_control_degron_rep1_cov_dia_run{i}/'
    #model_file = f'/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1_run{i}/'
    label_file = f'{model_file}labels.npy'
    # load the model and labels
    clusters = np.load(label_file)
    model = load_model(model_file)
    # subselect 8000 points
    x_shuffled, clusters_shuffled = shuffle(x, clusters, random_state=42)
    data = x_shuffled[:8000]
    z,_,_ = model.encoder.predict(data)
    labels = clusters_shuffled[:8000]
    # Calculate scores
    sil_score = silhouette_score(z, labels)
    ch_score = calinski_harabasz_score(z, labels)
    print(f"run{i}")
    print(f"Silhouette Score: {sil_score}")
    print(f"Calinski-Harabasz Index: {ch_score}")












