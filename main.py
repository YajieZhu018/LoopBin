"""
LoopBin clusters chromatin loops with a VADE model.
It provides functions for preprocessing, processing, pretraining, training, and clustering.

Author: Yajie Zhu, Alexis Bel
Date: 2024-12-08
"""
import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.fn import function
from src.fn import processing
from src.plot import plotting
from src.model.ae import AE
from src.model.vade_model import VADE
from sklearn.cluster import KMeans
import tensorflow as tf
import random
tf.keras.backend.set_floatx('float32')
#from sklearn.mixture import GaussianMixture

def parse_arguments():
    """
    The `parse_arguments()` function is used to obtain command line arguments for a VADE model script.
    Return: The function `parse_arguments()` returns the parsed arguments obtained from the command line.
    """
    """Obtain the arguments from the command line"""
    parser = argparse.ArgumentParser(
        description="VADE model for direct usage",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-f",
        dest="flag",
        help="Functions to run\n"
        "The available functions are:\n"
        "preprocess\n"
        "process\n"
        "normalize\n"
        "pretrain\n"
        "train\n"
        "cluster\n"
        "merge\n"
        "Usage:\n"
        "preprocess: preprocess input epigenetic data, used with -b <bigwig file> -g <output folder of bedgraph> -n <name of epigenetic data>\n"
        "processing: process bedgraph files and a cool file to generate the input for the model,\
              used with -l <loop file> -c <cool file> -g <folder of bedgraph> -o <output folder> -r <number of processors>\n"
        "normalize: merge and co-normalize processed data of different conditions, used with -e <cond1,cond2> -u <output folder>\n"
        "pretrain: pretrain the autencoder (AE) model, used with -d <processed data> -u <output folder>\n"
        "train: train the VADE model and clustering, used with -num <number of clusters> -d <processed_data> -if_pre <True/False> -pre <path to the pretrained model> \
            -u <output folder>, -ep <epoch number> -p <epig_name1,epig_name2>\n"
        "cluster: predict the cluster with a trained model, used with -d <processed data> -m <model path> -u <output folder>\n"
        "merge: merge small clusters, used with -d <processed data> -u <output folder> -k <clusters to merge>\n",
        nargs="?",
        const=0
    )
    parser.add_argument(
        "-b",
        dest="preprocess",
        help="Path to the bigwig files.",
        nargs="?",
        const="_"
    )
    parser.add_argument(
        "-n",
        dest="name",
        help="Name of the CUT&TAG data. such as CTCF, H3K27ac, H3K27me3, and SMC1A",
        nargs="?",
        const=0
    )
    parser.add_argument(
        "-g",
        dest="bedgraph_folder",
        help="Path to the bedgraph files of the epigenetic features.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-p",
        dest="proteins",
        help="Name of the CUT&TAG separated by comma",
        nargs="?",
        const="CTCF,H3K27ac,H3K27me3,SMC1A"
    )
    parser.add_argument(
        "-e",
        dest="conditions",
        help="list of conditions separated by comma, such as control,treatment",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-l",
        dest="list_loop",
        help="Path to the bedpe file containing the chromatin loops.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-c",
        dest="cool_file",
        help="Path to the mcool file of micro-C data. The resolution of 8kb is used.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-d",
        dest="file",
        help="Path to the processed data",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-r",
        dest="nbr_cpu",
        help="Number of cpus.",
        nargs="?",
        const=1
    )
    parser.add_argument(
        "-u",
        dest="folder",
        help="Folder to the output.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-num",
        dest="cluster_number",
        help="Number of cluster",
        nargs="?",
        const=10
    )
    parser.add_argument(
        "-ep",
        dest="epoch_number",
        help="Number of epochs for the training",
        nargs="?",
        const=1000
    )
    parser.add_argument(
        "-pre",
        dest="pretrained_model",
        help="Path to the pretrained AE model",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-if_pre",
        dest="if_pretrain",
        help="True if the pretrain model is used",
        nargs="?",
        const='True'
    )
    parser.add_argument(
        "-m",
        dest="model",
        help="Path to the VADE model",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-k",
        dest="clusters",
        help="clusters to merge separated by comma such as 2,3",
        nargs="?",
        const=None
    )
    return parser.parse_args()

def preprocess(args):
    """Preprocessing step"""
    # Verification of the argument use
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Extract the directory path of the current script (parent directory)
    script_directory = os.path.dirname(current_script_path)

    # Build the absolute path to the "preprocessing" folder based on the script's location
    preprocessing_folder = os.path.join(script_directory, 'src')
    preprocessing_command = os.path.join(script_directory, 'src', 'fn')

    function.verif_preprocess(args)
    # Preprocessing
    os.system(f'sh {preprocessing_folder}/preprocess_local.sh {args.preprocess} {args.name} {args.bedgraph_folder} {preprocessing_command}')
    sys.exit()


def process(args):
    """Processing step"""
    # Verification of the argument use
    cool_file = function.verif_process(args)
    function.verif_folder(args.bedgraph_folder)
    # Process
    processing.process(args.list_loop, cool_file, args.bedgraph_folder, args.proteins,
                       args.nbr_cpu, args.folder)
    sys.exit()

def process_all_groups(args):
    """merged and log processed data step"""
    # Process
    processing.process_all_groups(args.conditions, args.folder)
    sys.exit()

def cluster_data_inner_func(data, vade, loop_path, output_path, list_epic):
    # predict the latent space of the data
    z_mean,_,z = vade.encoder.predict(data)
    # get the probability of the data; shape(orignal data shape, number of clusters)
    prob = vade.gmm(z_mean)
    # get the cluster of the data
    cluster = np.argmax(prob,axis=1)
    # remove non-existing cluster 
    labels = np.unique(cluster)
    # Convert labels to TensorFlow tensor
    labels = tf.convert_to_tensor(np.unique(cluster), dtype=tf.int32)
    # Use TensorFlow indexing
    prob = tf.gather(prob, labels, axis=1)
    # recalculate the prob so the sum = 1
    prob = prob / tf.reduce_sum(prob, axis=1, keepdims=True)
    cluster = np.argmax(prob,axis=1)
    # save the probability
    np.save(f'{output_path}/prob.npy', prob)
    # save the cluster
    np.save(f'{output_path}/labels.npy', cluster)
    # add the label to the end of loops
    loop_file = f'{loop_path}/loop_file_analysis.bedpe'
    out_loop_file = f'{output_path}/labels_loops.bedpe'
    # Read the TSV file
    with open(loop_file, 'r') as f:
        lines = f.readlines()
    # Write the updated content to a new file
    with open(out_loop_file, 'w') as f:
        for i, line in enumerate(lines):
            line = line.strip()  # Remove newline or extra spaces
            new_line = f"{line}\t{cluster[i]}"  # Append the NumPy array value as a new column
            f.write(new_line + '\n')  # Write the new line with the appended column
    # get the reconstruction of the data
    recon = vade.decoder(z_mean)
    # save the reconstruction
    np.save(f'{output_path}/recon.npy', recon)
    # plot the average plot of each cluster
    ori_micro_c = data[:,:256]
    ori_epigenetic = data[:,256:]
    x_data = processing.create_data(ori_epigenetic, ori_micro_c)
    #x_data = np.load(input_data_path)
    # get the reconstructed data
    recon_micro_c = recon[:,:256]
    recon_epigenetic = recon[:,256:]
    x_recon = processing.create_data(recon_epigenetic, recon_micro_c)
    dict_ori = function.sep_cluster(x_data, cluster)
    dict_recon = function.sep_cluster(x_recon, cluster)
    plotting.plot_cluster(dict_ori, cluster, dict_recon, output_path, list_epic)
    plotting.plot_all_clusters(dict_ori, cluster, output_path, list_epic)
    # pie plot of the cluster
    plotting.plot_pie(dict_ori, cluster, output_path)
    # plot the tsne of the latent space
    plotting.plot_tsne(z_mean, cluster, output_path)

# Custom callback to save every 200 epochs
class SaveEveryNEpoch(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, save_path):
        super(SaveEveryNEpoch, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  # Save every `save_freq` epochs
            save_filepath = os.path.join(self.save_path, f'model_epoch_{epoch + 1}/')
            self.model.save(save_filepath)
            print(f"Checkpoint saved: {save_filepath}")

def pretrain_ae(args):
    """
    Pretrain the AE model
    """
    input_data_path = args.file
    ol = args.folder
    # load the input data
    X = np.load(input_data_path)
    # shuffle the data
    np.random.seed(0)
    np.random.shuffle(X)
    # set ae model
    d_input = X.shape[1]
    ae = AE(d_input)
    ae(np.zeros((10, d_input)))
    # set ae model
    adam_nn= tf.keras.optimizers.Adam(learning_rate=0.002)
    ae.compile(optimizer=adam_nn)
    # fit the model with X_train as training and X_test as validation
    history = ae.fit(X, shuffle=True, batch_size=256, epochs=200)
    # save the model
    ae.save(ol)
    # plot loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    # save it
    pdf = f'{ol}pretrain_ae_loss.pdf'
    plt.savefig(pdf)
    plt.close()
    # predict latent space of X_test
    z = ae.encoder(X)
    plotting.plot_score(z, ol)
    mcluster = function.set_kmeans(z)
    with open(f'{ol}/model_cluster.pkl', "wb") as file_pointer:
        pickle.dump(mcluster, file_pointer)

def save_model_each_200_epochs(args):
    """
    Train the VADE model
    """
    # get input
    n_clusters = int(args.cluster_number)
    data_path = args.file
    pretrain_model_path = args.pretrained_model
    output_path = args.folder
    list_epic = args.proteins.split(',')
    # get True if pretrain model is used
    if_pretrain = args.if_pretrain
    epochs = int(args.epoch_number)
    gmm_name = output_path.split('/')[-2]
    # generate random seed
    #seed = random.randint(1,100)
    seed = 73
    print(f'seed = {seed}')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    # load the data
    X = np.load(data_path)
    # no test data for unsupervised learning
    X_train = X
    # set vade model
    d_input = X_train.shape[1]
    vade = VADE(d_input,n_clusters)
    vade(np.zeros((10, d_input)))
    #vade.summary()
    # load the pretrain model
    if if_pretrain == 'True':
        vade.load_pretrained_weights(pretrain_model_path,X_train,gmm_name)
    ## Define a learning rate scheduler
    decay_nn = 0.9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: max(0.0002, 0.002 * decay_nn ** (epoch//10))  # Apply decay to the learning rate
    )
    # Directory for saving checkpoints
    checkpoint_dir = f'{output_path}checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Instantiate the custom callback
    save_callback = SaveEveryNEpoch(save_freq=200, save_path=checkpoint_dir)
    # set ae model
    adam_nn= tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-4)
    vade.compile(optimizer=adam_nn)
    history = vade.fit(X_train, shuffle=True, batch_size=256, epochs=epochs, callbacks=[lr_scheduler, save_callback],verbose=2)
    # save the model
    #vade.save(output_path)


def train_vade(args):
    """
    Train the VADE model
    """
    # get input
    n_clusters = int(args.cluster_number)
    data_path = args.file
    pretrain_model_path = args.pretrained_model
    output_path = args.folder
    list_epic = args.proteins.split(',')
    # get True if pretrain model is used
    if_pretrain = args.if_pretrain
    epochs = int(args.epoch_number)
    gmm_name = output_path.split('/')[-2]
    # generate random seed
    #seed = random.randint(1,100)
    seed = 73
    print(f'seed = {seed}')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    #tf.config.threading.set_inter_op_parallelism_threads(8)
    #tf.config.threading.set_intra_op_parallelism_threads(8)
    # load the data
    X = np.load(data_path)
    # no test data for unsupervised learning
    X_train = X
    # set vade model
    d_input = X_train.shape[1]
    vade = VADE(d_input,n_clusters)
    vade(np.zeros((10, d_input)))
    #vade.summary()
    # load the pretrain model
    if if_pretrain == 'True':
        vade.load_pretrained_weights(pretrain_model_path,X_train,gmm_name)
    ## Define a learning rate scheduler
    decay_nn = 0.9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: max(0.0002, 0.002 * decay_nn ** (epoch//10))  # Apply decay to the learning rate
    )
    # set ae model
    adam_nn= tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-4)
    vade.compile(optimizer=adam_nn)
    history = vade.fit(X_train, shuffle=True, batch_size=256, epochs=epochs, callbacks=[lr_scheduler],verbose=2)
    # save the model
    vade.save(output_path)
    # plot loss of the model
    plotting.plot_train_loss(history, output_path)
    # predict latent space of X
    loop_path = os.path.dirname(data_path)
    cluster_data_inner_func(X_train, vade, loop_path, output_path, list_epic)


def cluster_data(args):
    # get the data path, model path from argv
    data_path = args.file
    model_path = args.model
    output_path = args.folder
    list_epic = args.proteins.split(',')
    data = np.load(data_path)
    loop_path = os.path.dirname(data_path)
    # load the model
    model = tf.keras.models.load_model(model_path)
    cluster_data_inner_func(data, model, loop_path, output_path, list_epic)


def plot_result(x_data, reconstructed_data, lat_space, labels, plot_folder):
    """Separate cluster data and plottet it"""
    # Separate the data by cluster
    dict_clust = function.sep_cluster(x_data, labels)
    dict_rec = function.sep_cluster(reconstructed_data, labels)

    # Plot the result
    plotting.plot_pie(dict_clust, labels, plot_folder)
    plotting.plot_cluster(dict_clust, labels, dict_rec, plot_folder)
    plotting.plot_tsne(lat_space, labels, plot_folder)

def train_vade_with_test(args):
    """
    Train the VADE model
    """
    # get input
    n_clusters = int(args.cluster_number)
    data_path = args.file
    pretrain_model_path = args.pretrained_model
    output_path = args.folder
    list_epic = args.proteins.split(',')
    # get True if pretrain model is used
    if_pretrain = args.if_pretrain
    epochs = int(args.epoch_number)
    gmm_name = output_path.split('/')[-2]
    # set random seed to ensure the reproducibility
    #random.seed(0)
    # load the data
    X = np.load(data_path)
    # shuffle the data
    #tf.random.set_seed(42)
    X_shuffled = tf.random.shuffle(X)
    # split the data into train and test with tensorflow
    split_index = int(0.8 * X.shape[0])
    X_train = X_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    # save the training and test data
    os.makedirs(output_path, exist_ok=True)
    np.save(f'{output_path}/X_train.npy', X_train)
    np.save(f'{output_path}/X_test.npy', X_test)
    # set vade model
    d_input = X_train.shape[1]
    vade = VADE(d_input,n_clusters)
    vade(np.zeros((10, d_input)))
    #vade.summary()
    # load the pretrain model
    if if_pretrain == 'True':
        vade.load_pretrained_weights(pretrain_model_path,X_train,gmm_name)
    ## Define a learning rate scheduler
    decay_nn = 0.9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: max(0.0002, 0.002 * decay_nn ** (epoch//10))  # Apply decay to the learning rate
    )
    # set ae model
    adam_nn= tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-4)
    vade.compile(optimizer=adam_nn)
    history = vade.fit(X_train, shuffle=True, batch_size=256, epochs=epochs, callbacks=[lr_scheduler],verbose=1, validation_data=(X_test, X_test))
    # save the model
    vade.save(output_path)
    # plot loss of the model
    plotting.plot_loss(history, output_path)
    ## concatenate data and folder pairs into a list
    #data_folders = [(X_train, "train"), (X_test, "test")]
    ## create the subfolder
    #for _, folder in data_folders:
    #    os.makedirs(f'{output_path}/{folder}', exist_ok=True)
    ## predict on both training and testing data
    #loop_path = os.path.dirname(data_path)
    #for data, folder in data_folders:
    #    cluster_data_inner_func(data, vade, loop_path, f'{output_path}/{folder}', list_epic)


def calculate_generalizability(args):
    """
    determine the cluster number based on generalizability
    """
    from sklearn.model_selection import KFold
    # get input
    data_path = args.file
    pretrain_model_path = args.pretrained_model
    output_path = args.folder
    list_epic = args.proteins.split(',')
    # get True if pretrain model is used
    if_pretrain = args.if_pretrain
    epochs = int(args.epoch_number)
    gmm_name = output_path.split('/')[-2]
    #seed = random.randint(1,100)
    #print(f'seed = {seed}')
    #random.seed(seed)
    #np.random.seed(seed)
    #tf.random.set_seed(seed)
    # set random seed to ensure the reproducibility
    X = np.load(data_path)
    np.random.shuffle(X)
    # set vade model
    d_input = X.shape[1]
    ## Define a learning rate scheduler
    decay_nn = 0.9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: max(0.0002, 0.002 * decay_nn ** (epoch//10))  # Apply decay to the learning rate
    )
    # initialize dic to store generalizability
    dic_g = {'g':{}, 'g recon':{}, 'g kl':{}, 'train loss':{}, 'test loss':{}, 
             'train recon loss':{}, 'test recon loss':{}, 'train kl loss':{}, 'test kl loss':{}, 'N cluster':{}, 'N all cluster':{}}
    for n_clusters in range(4,11):
        # create n_cluster as key and empty list as value
        dic_g['g'][n_clusters] = []
        dic_g['g recon'][n_clusters] = []
        dic_g['g kl'][n_clusters] = []
        dic_g['train loss'][n_clusters] = []
        dic_g['test loss'][n_clusters] = []
        dic_g['train recon loss'][n_clusters] = []
        dic_g['test recon loss'][n_clusters] = []
        dic_g['train kl loss'][n_clusters] = []
        dic_g['test kl loss'][n_clusters] = []
        dic_g['N cluster'][n_clusters] = []
        dic_g['N all cluster'][n_clusters] = []
        vade = VADE(d_input,n_clusters)
        vade(np.zeros((10, d_input)))
        #vade.summary()
        # load the pretrain model
        if if_pretrain == 'True':
            vade.load_pretrained_weights(pretrain_model_path,X,gmm_name)
        # set ae model
        adam_nn= tf.keras.optimizers.Adam(learning_rate=0.002,epsilon=1e-4)
        vade.compile(optimizer=adam_nn)

        # Define cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # Initialize dict to store errors
        dic_err = {'train':[], 'test':[], 'train_recon':[],'test_recon':[], 'train_kl':[],'test_kl':[], 'N_cluster':[], 'N_all_cluster':[]}
        # Perform cross-validation
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            history = vade.fit(X_train, shuffle=True, batch_size=256, epochs=epochs, callbacks=[lr_scheduler],verbose=1,validation_data=(X_test, X_test))
            for data, key in [(X_train,'train'), (X_test,'test')]:
                z_mean, z_log_var, z = vade.encoder.predict(data)
                reconstruction = vade.decoder(z)
                reconstruction_loss = loss_fn(data, reconstruction)*d_input
                # calculate vae loss according to the vae loss function
                kl_loss = vade.calculate_kl_loss(z, z_mean, z_log_var)
                loss = reconstruction_loss + kl_loss
                #scalar_loss = np.mean(loss.numpy())
                dic_err[key].append(loss.numpy().astype(float))
                dic_err[f'{key}_recon'].append(reconstruction_loss.numpy().astype(float))
                dic_err[f'{key}_kl'].append(kl_loss.numpy().astype(float))
                from collections import Counter
                def filter_elements_by_frequency(lst, threshold_percent=2):
                    # Count the occurrences of each element
                    element_counts = Counter(lst)
                    # Total number of elements in the list
                    total_elements = len(lst)
                    # Calculate the threshold
                    threshold = (threshold_percent / 100) * total_elements
                    # Get elements whose frequency is greater than the threshold
                    result = [element for element, count in element_counts.items() if count > threshold]
                    return result
                if (key == 'train'):
                    # get the real cluster number
                    prob = vade.gmm(z_mean)
                    # get the cluster of the data
                    cluster = np.argmax(prob,axis=1)
                    labels = filter_elements_by_frequency(list(cluster), threshold_percent=2)
                    # save the real cluster number
                    dic_err['N_all_cluster'].append(len(np.unique(cluster)))
                    dic_err['N_cluster'].append(len(labels))
        # Calculate generalization as train error / test error for each fold
        generalization = list(np.array(dic_err['train']) / np.array(dic_err['test']))
        g_recon = list(np.array(dic_err['train_recon']) / np.array(dic_err['test_recon']))
        g_kl = list(np.array(dic_err['train_kl']) / np.array(dic_err['test_kl']))
        # add to dic_g
        dic_g['g'][n_clusters] = generalization
        dic_g['g recon'][n_clusters] = g_recon
        dic_g['g kl'][n_clusters] = g_kl
        dic_g['train loss'][n_clusters] = dic_err['train']
        dic_g['test loss'][n_clusters] = dic_err['test']
        dic_g['train recon loss'][n_clusters] = dic_err['train_recon']
        dic_g['test recon loss'][n_clusters] = dic_err['test_recon']
        dic_g['train kl loss'][n_clusters] = dic_err['train_kl']
        dic_g['test kl loss'][n_clusters] = dic_err['test_kl']
        dic_g['N cluster'][n_clusters] = dic_err['N_cluster']
        dic_g['N all cluster'][n_clusters] = dic_err['N_all_cluster']
    # save g
    import json
    with open(f'{output_path}/generalizability_vs_set_num_clusters.json', 'w') as json_file:
        json.dump(dic_g, json_file, indent=4)
    # get actual cluster number
    from collections import defaultdict  # it allows appending without initializing empty list
    new_dict = defaultdict(list)
    for key in dic_g:
        if key != 'N cluster':
            new_dict[key] = defaultdict(list)
            # Iterate over each cluster count
            for cluster_num, values in dic_g[key].items():
                n_clusters = dic_g['N cluster'][cluster_num]
                # For each N cluster, append the corresponding the value
                for n, value in zip(n_clusters, values):
                    new_dict[key][n].append(value)
            # Convert defaultdict back to a regular dictionary (optional)
            new_dict[key] = dict(sorted(new_dict[key].items()))
    new_dict = dict(new_dict)
    # save
    out_file_name = f'{output_path}g_vs_actual_num_clusters.json'
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


def calcualte_NMI(args):
    from sklearn.metrics import normalized_mutual_info_score
    import seaborn as sns
    # args
    data_path = args.file  # where the labels locate
    output_path = args.folder
    # Load label files
    labels = [np.load(f'{data_path}_run{i}/labels.npy') for i in range(1, 6)]
    # Initialize a 5x5 matrix for storing NMI values
    nmi_matrix = np.zeros((5, 5))
    # Calculate pairwise NMI
    for i in range(5):
        for j in range(5):
            # Compute NMI between labels from run i and run j
            nmi_matrix[i, j] = normalized_mutual_info_score(labels[i], labels[j])
    # Plot the NMI matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(nmi_matrix, annot=True, cmap='viridis', xticklabels=[f'Run {i+1}' for i in range(5)], yticklabels=[f'Run {i+1}' for i in range(5)])
    plt.title("Pairwise Normalized Mutual Information (NMI) Between Runs")
    plt.xlabel("Runs")
    plt.ylabel("Runs")
    plt.savefig(f'{output_path}/nmi.pdf')
    plt.close()

def merge_small_clusters(args):
    # list of clusters to merge
    small_clusters = args.clusters.split(',')
    small_clusters = list(map(int,small_clusters))
    data_path = args.file
    output_path = args.folder
    list_epic = args.proteins.split(',')
    # create subfolders
    new_folder = f'{output_path}/cluster_merged/'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    # delete the small clusters
    prob = np.load(f'{output_path}prob.npy')
    mask = np.ones(prob.shape[1],dtype=bool)
    mask[small_clusters] = False
    new_prob = prob[:,mask]
    # recalculate the prob so the sum = 1
    new_prob = new_prob / new_prob.sum(axis=1, keepdims=True)
    np.save(f'{new_folder}prob.npy', new_prob)
    # get the labels
    cluster = np.argmax(new_prob, axis=1)
    np.save(f'{new_folder}labels.npy', cluster)
    # load the data
    data = np.load(data_path)
    # plot the average plot of each cluster
    ori_micro_c = data[:,:256]
    ori_epigenetic = data[:,256:]
    x_data = processing.create_data(ori_epigenetic, ori_micro_c)
    dict_ori = function.sep_cluster(x_data, cluster)
    plotting.plot_all_clusters(dict_ori, cluster, new_folder, list_epic)
    # pie plot of the cluster
    plotting.plot_pie(dict_ori, cluster, new_folder)

def find_consensu_cluster(args):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    data_path = args.file
    output_path = args.folder
    list_epic = args.proteins.split(',')
    model_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_H3K27ac_H3K27me3_SMC1A_H3K4me1'
    os.makedirs(output_path, exist_ok=True)
    results = [np.load(f'{model_path}_run{i}/labels.npy') for i in range(1, 6)]
    # Assuming `results` is a list of 5 arrays, each containing the cluster labels of one run
    n_samples = len(results[0])
    n_runs = len(results)
    # Step 1: Create the co-occurrence matrix
    co_occurrence_matrix = np.zeros((n_samples, n_samples))
    # Vectorized calculation for the co-occurrence matrix
    for result in results:
        # Create an indicator matrix where each element is 1 if two samples share a cluster
        indicator_matrix = (result[:, None] == result[None, :]).astype(float)
        co_occurrence_matrix += indicator_matrix
    # Normalize by the number of runs
    co_occurrence_matrix /= n_runs
    # Step 2: Apply hierarchical clustering on the co-occurrence matrix
    # Define the final number of clusters (e.g., 6) or use a distance threshold
    cluster = AgglomerativeClustering(
        n_clusters=7, affinity='precomputed', linkage='average'
        ).fit_predict(1 - co_occurrence_matrix)  # 1 - matrix as dissimilarity
    # save final clusters
    np.save(f'{output_path}consensus_cluster.npy', cluster)
    # load the data
    data = np.load(data_path)
    # plot the average plot of each cluster
    ori_micro_c = data[:,:256]
    ori_epigenetic = data[:,256:]
    x_data = processing.create_data(ori_epigenetic, ori_micro_c)
    dict_ori = function.sep_cluster(x_data, cluster)
    plotting.plot_all_clusters(dict_ori, cluster, output_path, list_epic)
    # pie plot of the cluster
    plotting.plot_pie(dict_ori, cluster, output_path)
    # Optional: Plot the dendrogram
    # Use linkage on the co-occurrence matrix to create the hierarchy
    Z = linkage(co_occurrence_matrix, method='average')
    # Sample a subset of data
    sample_size = 5000  # Adjust based on your needs
    indices = np.random.choice(len(Z), sample_size, replace=False)
    sampled_Z = linkage(Z[indices], method='average')
    plt.figure(figsize=(10, 7))
    dendrogram(sampled_Z)
    plt.title("Dendrogram of Consensus Clustering")
    plt.xlabel("Sample index")
    plt.ylabel("Co-occurrence distance")
    plt.savefig(f'{output_path}consensus_clusters.pdf')
    

def main(args):
    """Main function"""
    if args.flag == 'preprocess':
        preprocess(args)
    elif args.flag == 'process':
        process(args)
    elif args.flag == 'normalize':
        process_all_groups(args)
    elif args.flag == 'pretrain':
        pretrain_ae(args)
    elif args.flag == 'train':
        train_vade(args)
    elif args.flag == 'cluster':
        cluster_data(args)
    elif args.flag == 'merge':
        merge_small_clusters(args)
    elif args.flag == 'calculateg':
        calculate_generalizability(args)
    elif args.flag == 'nmi':
        calcualte_NMI(args)
    else:
        sys.exit("Invalid flag.")
    

if __name__ == '__main__':
    # Get the arguments
    command_args = parse_arguments()
    # Run the main function
    main(command_args)
