"""
This script provides a VAE model to clusterise chromatine loops.

The script supports preprocessing, processing, clustering, and data generation.
It uses command line arguments to determine the functionality to execute.

Author: Yajie Zhu, Alexis Bel
Date: 2024-07-13
"""
import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.fn import function
from src.fn import processing
from src.fn import init_data
from src.fn import processing_pca
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
    The `parse_arguments()` function is used to obtain command line arguments for a VAE model script.
    :return: The function `parse_arguments()` returns the parsed arguments obtained from the command
    line.
    """
    """Obtain the arguments from the command line"""
    parser = argparse.ArgumentParser(
        description="VAE model for direct usage",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-f",
        dest="flag",
        help="Flag to run different functions of the script\n"
        "1. Preprocessing, use with -b, -g, and -n\n"
        "2. Processing, use with -l , -c and -g. Optional -r \
for multiprocessing and -u the folder\n"
        "2.1 Merge processed data, use with -e for the condition groups and -u for the output folder, "
        "3. Pretrain the AE model and save it. With -d for the \
process data and -u for the folder \n"
        "4. Train the VADE model and clustering, use with -num for the number of cluster \
            -d for the processed data, -pre for the pretrained model path, -u for the output folder, -ep for the epoch number \
            -if_pre for if pretrain \n"
        "5. Predict the cluster with a trained model, use with -d for the processed data, -m for the model path, -u for the output folder\n"
        "6. Generate 100 random samples. The sample generated are bad\n"
        "7. Generate PCs of each datasets and combine as input for clustering, use with -o, -u",
        nargs="?",
        const=0
    )
    parser.add_argument(
        "-b",
        dest="preprocess",
        help="File bigwig to preprocess. In the case you don't have the \
bigwig of a feature, you can write \"empty\" and it \
will generate empty bedgraph",
        nargs="?",
        const="_"
    )
    parser.add_argument(
        "-n",
        dest="name",
        help="Name of the preprocessing feature. Only four available: CTCF, \
H3K27ac, H3K27me3, and SMC1A",
        nargs="?",
        const=0
    )
    parser.add_argument(
        "-g",
        dest="bedgraph_folder",
        help="Folder containing the bedgraph files from epigenetic features \
or where to put the bedgraph files.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-e",
        dest="conditions",
        help="list of conditions.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-o",
        dest="log_nornalized_input_folder",
        help="Folder containing the log min-max normalized files from microC and epigenetic features",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-l",
        dest="list_loop",
        help="Bedpe file containing  the localisation of the chromatin loop.",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-c",
        dest="cool_file",
        help="Cool file containing  micro-C data. Will automatically use the \
resolution 8000",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-d",
        dest="file",
        help="File of the processed data",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-r",
        dest="nbr_cpu",
        help="Number of cpu use for the processing",
        nargs="?",
        const=1
    )
    parser.add_argument(
        "-u",
        dest="folder",
        help="Folder to put the output.For Processing and result of \
clustering",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-fo",
        dest="weight_folder",
        help="Folder of the VAE model",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-num",
        dest="cluster_number",
        help="Number of cluster for the clustering",
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
        help="Path to the pretrained model",
        nargs="?",
        const=None
    )
    parser.add_argument(
        "-if_pre",
        dest="if_pretrain",
        help="True if pretrain model is used",
        nargs="?",
        const='True'
    )
    parser.add_argument(
        "-m",
        dest="model",
        help="Path to the model",
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
    processing.process(args.list_loop, cool_file, args.bedgraph_folder,
                       args.nbr_cpu, args.folder)
    sys.exit()

def process_all_groups(args):
    """merged and log processed data step"""
    # Process
    processing.process_all_groups(args.conditions, args.folder)
    sys.exit()

def process_pca(args):
    """Process the input to get first several PCs and combine into one vector"""
    function.verif_folder(args.log_nornalized_input_folder)
    # Process
    processing_pca.process_pca(args.log_nornalized_input_folder, args.folder)
    sys.exit()

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

def train_vade(args):
    """
    Train the VADE model
    """
    # get input
    n_clusters = int(args.cluster_number)
    data_path = args.file
    pretrain_model_path = args.pretrained_model
    output_path = args.folder
    # get True if pretrain model is used
    if_pretrain = args.if_pretrain
    epochs = int(args.epoch_number)
    gmm_name = output_path.split('/')[-2]
    # set random seed to ensure the reproducibility
    random.seed(0)
    # load the data
    X = np.load(data_path)
    # no test data for unsupervised learning
    X_train = X
    # shuffle the data
    np.random.seed(0)
    np.random.shuffle(X)
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
    history = vade.fit(X_train, shuffle=True, batch_size=256, epochs=epochs, callbacks=[lr_scheduler],verbose=1)
    # save the model
    vade.save(output_path)
    # predict latent space of X_test
    z_mean,_,_ = vade.encoder(X_train)
    # get the probability of the data; shape(orignal data shape, number of clusters)
    prob = vade.gmm(z_mean)
    # get the cluster of the data
    cluster = np.argmax(prob,axis=1)
    # separate the cluster
    dict_clust = function.sep_cluster(X_train, cluster)
    # plot the tsne of the latent space
    plotting.plot_tsne(z_mean, cluster, output_path)
    # pie plot of the cluster
    plotting.plot_pie(dict_clust, cluster, output_path)
    # plot loss of the model
    plotting.plot_train_loss(history, output_path)

def cluster_data(args):
    # get the data path, model path from argv
    data_path = args.file
    model_path = args.model
    output_path = args.folder
    #input_data_path = sys.argv[4]
    # load the data
    data = np.load(data_path)
    #np.random.seed(0)
    #np.random.shuffle(data)
    #data = data[:10000]
    # load the model
    vade = tf.keras.models.load_model(model_path)
    # predict the latent space of the data
    z_mean,_,z = vade.encoder(data)
    # get the probability of the data; shape(orignal data shape, number of clusters)
    prob = vade.gmm(z_mean)
    # save the probability
    np.save(f'{output_path}/prob.npy', prob)
    # get the cluster of the data
    cluster = np.argmax(prob,axis=1)
    # save the cluster
    np.save(f'{output_path}/labels.npy', cluster)
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
    plotting.plot_cluster(dict_ori, cluster, dict_recon, output_path)
    # pie plot of the cluster
    plotting.plot_pie(dict_ori, cluster, output_path)
    # plot the tsne of the latent space
    plotting.plot_tsne(z_mean, cluster, output_path)


def plot_result(x_data, reconstructed_data, lat_space, labels, plot_folder):
    """Separate cluster data and plottet it"""
    # Separate the data by cluster
    dict_clust = function.sep_cluster(x_data, labels)
    dict_rec = function.sep_cluster(reconstructed_data, labels)

    # Plot the result
    plotting.plot_pie(dict_clust, labels, plot_folder)
    plotting.plot_cluster(dict_clust, labels, dict_rec, plot_folder)
    plotting.plot_tsne(lat_space, labels, plot_folder)


def load_model(weight_folder):
    """Load VAE model and cluster model"""
    #current_script_path = os.path.abspath(__file__)
    # Extract the directory path of the current script (parent directory)
    #script_directory = os.path.dirname(current_script_path)
    #weight_folder = os.path.join(script_directory, 'src','saved_model')
    #set_random()

    #vae_network = set_model(latent_size, 3, 0.00077,beta)
    #vae_network = set_model(40, 3, 0.00077)
    #vae_network = set_model(49, 2.9435442035310144e-08, 3, 0.0005081969815992698)
    #vae_network = set_model(49, 3, 0.001) #0.0005081969815992698
    # Load the model weights
    #try:
    #    vae_network.load_weights(f'{weight_folder}/my_TRUE_model_weights_clust.h5')
    #except (FileNotFoundError, IOError) as exception:
    #    sys.exit(f"Something went wrong when loading the weights of the\
    #        model: {exception}")
    try:
        vae_network = tf.keras.models.load_model(f'{weight_folder}my_VAE_model')
    except (FileNotFoundError, IOError) as exception:
        sys.exit(f"Something went wrong when loading the\
            model: {exception}")
    #Load the cluster (such as kmeans) model
    #try:
    #    with open(f"{weight_folder}/model_kmeans.pkl", "rb") as file_pointer:
    #        model_cluster = pickle.load(file_pointer)
    #except (FileNotFoundError, IOError) as exception:
    #    sys.exit(f"Something went wrong when loading the cluster model:\
    #        {exception}")

    return vae_network      #, model_cluster


def run_model(args):
    """Run the model and clustering step"""
    # Load model
    vae_network = load_model(args.weight_folder)
    # See if the file exist and load the data
    function.verif_file(args.file, ".npy")
    x_data = init_data.load_data(args.file)
    # Define where to put the output
    if args.folder is not None:
        result_folder, plot_folder = function.set_result_folders(args.folder)
    else:
        result_folder, plot_folder = function.set_default_folders()

    # Predict the latent space and reconstruction from the latent space
    #_, _, lat_space = vae_network.encoder.predict([x_data,x_data])
    lat_space = vae_network.encoder.predict([x_data,x_data])
    reconstructed_data = vae_network.decoder.predict(lat_space)
    function.save_latent(lat_space, os.path.join(result_folder,
                                                 "latent_space.npy"))
    # Predict the label of the latent space and save it
    # fit the cluster model with latent space
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(lat_space)
    labels = kmeans.labels_
    #model_cluster.fit(lat_space)
    #labels = model_cluster.predict(lat_space)
    with open(os.path.join(result_folder, "loops_labels.pkl"), "wb")as file_point:
        pickle.dump(labels, file_point)
    # Plot and save result
    plot_result(x_data, reconstructed_data, lat_space, labels, plot_folder)
    ## normalize lat space and plot
    ## Normalize the data across features
    #mean = np.mean(lat_space, axis=0, keepdims=True)
    #std = np.std(lat_space, axis=0, keepdims=True)
    #normalized_lat_space = (lat_space - mean) / std
    #function.save_latent(normalized_lat_space, os.path.join(result_folder,
    #                                             "normalized_latent_space.npy"))
    ## define a new kmeans cluster
    #new_model_cluster = KMeans(n_clusters=model_cluster.n_clusters, random_state=0)
    #new_model_cluster.fit(normalized_lat_space)
    #normalized_labels = new_model_cluster.predict(normalized_lat_space)
    #with open(os.path.join(result_folder, "normalized_loops_labels.pkl"), "wb")as file_point:
    #    pickle.dump(normalized_labels, file_point)
    ##os.system(f'mkdir -p {plot_folder}/normalized_latent_space')
    #plot_result(x_data, reconstructed_data, normalized_lat_space, normalized_labels, f'{plot_folder}/normalized_latent_space/')
    sys.exit()


def run_random(args):
    """Generate 100 new data"""
    # Load models
    current_script_path = os.path.abspath(__file__)
    vae_network, model_cluster = load_model()
    # Generate random latent space point
    array_shape = (100, 28)
    script_directory = os.path.dirname(current_script_path)
    weight_folder = os.path.join(script_directory, 'src','saved_model')
    variances = np.load(f"{weight_folder}/var_data.npy")
    lat_space = np.random.normal(loc=0, scale=variances, size=array_shape)

    # Define where to put the output
    if args.folder is not None:
        result_folder, plot_folder = function.set_result_folders(args.folder)
    else:
        result_folder, plot_folder = function.set_default_folders()

    # Create new data reconstruction from the latent space
    reconstructed_data = vae_network.decoder.predict(lat_space)
    function.save_latent(lat_space, os.path.join(result_folder,
                                                 "latent_space.npy"))

    # Predict the label of the latent space and save it
    labels = model_cluster.predict(lat_space.astype('float32'))
    with open(os.path.join(result_folder, "loops_labels"), "wb") as file_point:
        pickle.dump(labels, file_point)

    # Plot and save result
    plot_result(reconstructed_data, reconstructed_data,
                lat_space, labels, plot_folder)

def test_print(args):
    process_pca.test_hello()

def main(args):
    """Main function"""
    if args.flag == '1':
        preprocess(args)
    elif args.flag == '2':
        process(args)
    elif args.flag == '2.1':
        process_all_groups(args)
    elif args.flag == '3':
        pretrain_ae(args)
    elif args.flag == '4':
        train_vade(args)
    elif args.flag == '5':
        cluster_data(args)
    elif args.flag == '7':
        process_pca(args)
    elif args.flag == '6':
        run_random(args)
    else:
        sys.exit("Invalid flag.")


if __name__ == '__main__':
    # Get the arguments
    command_args = parse_arguments()
    # Run the main function
    main(command_args)
