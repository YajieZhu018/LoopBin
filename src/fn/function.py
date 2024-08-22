"""
This module contains utility functions for checking the files and
clustering using K-Means.

Functions:
- verif_preprocess(arguments): Perform preprocessing verification.
- verif_folder(folder_path): Verify the existence of a folder and create it if
it doesn't exist.
- check_bw_file_existence(file_path): Check if a file with .bw extension
exists.
- verif_file(name, extension): Verify the existence of a file with a
specific extension.
- verif_process(arguments): Perform process verification.
- save_latent(latent_space, name): Save latent space to a file.
- sep_cluster(data_conv, labels): Separate data by labels.
- set_result_folders(base_folder): Set result and plot folders.
- set_default_folders(): Set default result and plot folders.
- set_save_names(result_folder, base_folder=None): Set names for
saving results and plots.
- give_best(lat_space): Find the optimal number of clusters for K-Means.
- set_kmeans(lat_space): Set up K-Means clustering with the optimal number
of clusters.
- set_gmm (lat_space): set up GMM clustering. 
"""

import sys
import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator


def verif_preprocess(arguments):
    """
    Perform preprocessing verification.

    Args:
        arguments: Command-line arguments.

    Raises:
        SystemExit: If the required arguments are missing or invalid.
    """
    preprocess_file = arguments.preprocess
    database_folder = arguments.bedgraph_folder
    name = arguments.name
    if name == "empty":
        verif_folder(database_folder)
        return 0
    print(preprocess_file)
    if preprocess_file == "_" or name is None or database_folder is None:
        sys.exit("To preprocess, the 'preprocess', 'name', and\
                 'bedgraph_folder' arguments must be used. "
                 "Run the epigenetic feature one by one")

    if preprocess_file is None:
        sys.exit("Please indicate the file to preprocess")

    if not check_bw_file_existence(preprocess_file):
        sys.exit(f"{preprocess_file} not found or not in BW format")

    if name not in {"SMC1A", "CTCF", "H3K27ac", "H3K27me3"}:
        sys.exit("Only four names available: SMC1A, CTCF, H3K27ac, H3K27me3")

    verif_folder(database_folder)
    print(f"File {preprocess_file} detected. Bedgraph files chr?_{name}8K will\
           be put in {database_folder}")


def verif_folder(folder_path):
    """
    Verify the existence of a folder and create it if it doesn't exist.

    Args:
        folder_path (str): Path to the folder.
    """
    os.makedirs(folder_path, exist_ok=True)
    print(f"The new directory {folder_path} is created!")


def check_bw_file_existence(file_path):
    """
    Check if a file with .bw extension exists.

    Args:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the .bw file exists, False otherwise.
    """
    if file_path == "empty":
        print("Empty bedgraph will be generated")
        return True
    if file_path.endswith('.bw') and os.path.isfile(file_path):
        return True
    return False


def verif_file(name, extension):
    """
    Verify the existence of a file with a specific extension.

    Args:
        name (str): Name of the file.
        extension (str): Desired extension.

    Raises:
        SystemExit: If the file is not found or has an invalid extension.
    """
    if name is None:
        sys.exit("File not given")
    if not os.path.isfile(name) or not name.endswith(extension):
        sys.exit(f"File {name} not found or wrong file format {extension}\
                  needed")
    print(f"File {name} detected\n")


def verif_process(arguments):
    """
    Perform process verification.

    Args:
        arguments: Command-line arguments.

    Returns:
        str: The processed cool file with the resolution.

    Raises:
        SystemExit: If the required arguments are missing or invalid.
    """
    loop_list = arguments.list_loop
    cool_file = arguments.cool_file
    bedgraph_folder = arguments.bedgraph_folder

    if loop_list == "_" or cool_file is None or bedgraph_folder is None:
        sys.exit("To process, the 'list_loop', 'cool_file', and\
                'bedgraph_folder' arguments must be used. "
                 "Run the epigenetic feature one by one")
    verif_file(loop_list, ".bedpe")
    verif_file(cool_file, ".mcool")
    verif_folder(bedgraph_folder)
    return cool_file + "::/resolutions/8000"


def save_latent(latent_space, name):
    """
    Save latent space to a file.

    Args:
        latent_space (ndarray): Latent space array.
        name (str): Name of the file to save.
    """
    np.save(name, latent_space)


def sep_cluster(data_conv, labels):
    """
    Separate data by labels.

    Args:
        data_conv (ndarray): Input data array.
        labels (ndarray): Labels array.

    Returns:
        dict: Dictionary with separated arrays based on labels.
    """
    separated_arrays = {}

    for label, element in zip(labels, data_conv):
        separated_arrays.setdefault(label, []).append(element)
    return separated_arrays


def set_result_folders(base_folder):
    """
    Set result and plot folders.

    Args:
        base_folder (str): Base folder path.

    Returns:
        tuple: Result folder path, plot folder path.
    """
    result_folder = os.path.join(base_folder, "result")
    plot_folder = os.path.join(base_folder, "plot")
    verif_folder(result_folder)
    verif_folder(plot_folder)
    return result_folder, plot_folder


def set_default_folders():
    """
    Set default result and plot folders.

    Returns:
        tuple: Result folder path, plot folder path.
    """
    verif_folder("result")
    verif_folder("plot")
    return "result", "plot"


def set_save_names(result_folder, base_folder=None):
    """
    Set names for saving results and plots.

    Args:
        result_folder (str): Result folder path.
        base_folder (str, optional): Base folder path. Defaults to None.

    Returns:
        tuple: Save names for loops labels, latent space, and plot folder.
    """
    if base_folder is not None:
        save_name = os.path.join(result_folder, "loops_labels")
        save_name_2 = os.path.join(result_folder, "latent_space.npy")
        save_name_plot = f"{base_folder}/plot"
    else:
        save_name = "result/loops_labels"
        save_name_2 = "result/latent_space.npy"
        save_name_plot = "plot"
    return save_name, save_name_2, save_name_plot


#def give_best(lat_space):
#    """
#    Find the optimal number of clusters for K-Means.
#
#    Args:
#        lat_space (ndarray): Latent space data.
#
#    Returns:
#        int: Optimal number of clusters.
#    """
#    score = []
#    for i in range(2, 25):
#        kmeans = KMeans(n_clusters=i, random_state=0)
#        kmeans.fit(lat_space)
#        score.append(kmeans.inertia_)
#    best_nbr = KneeLocator(range(1, len(score) + 1), score, curve='convex',
#                           direction='decreasing')
#    return best_nbr.knee

def give_best(lat_space, method='kmeans'):
    """
    Find the optimal number of clusters for K-Means, AgglomerativeClustering, or DBSCAN.

    Args:
        lat_space (ndarray): Latent space data.
        method (str): Clustering method. Options are 'kmeans', 'agglomerative', or 'dbscan'. Default is 'kmeans'.

    Returns:
        int: Optimal number of clusters.
    """
    if method == 'kmeans':
        score = []
        for i in range(2, 25):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(lat_space)
            score.append(kmeans.inertia_)
        best_nbr = KneeLocator(range(1, len(score) + 1), score, curve='convex', direction='decreasing')
        return best_nbr.knee
    elif method == 'gmm':
        score = []
        for i in range(2, 25):
            gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=0)
            gmm.fit(lat_space)
            score.append(gmm.bic(lat_space))  # Bayesian Information Criterion (BIC) used for GMM
        best_nbr = KneeLocator(range(1, len(score) + 1), score, curve='convex', direction='decreasing')
        return best_nbr.knee
    elif method == 'agglomerative':
        score = []
        for i in range(2, 25):
            agglomerative = AgglomerativeClustering(n_clusters=i)
            labels = agglomerative.fit_predict(lat_space)
            score.append(np.mean(labels))
        best_nbr = KneeLocator(range(1, len(score) + 1), score, curve='convex', direction='decreasing')
        return best_nbr.knee
    elif method == 'dbscan':
        min_samples = min(5, len(lat_space) // 10)
        eps = 0.5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(lat_space)
        unique_labels = np.unique(labels)
        # If all points are assigned to noise, return 1
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            return 1
        # Otherwise, return the number of clusters excluding noise
        else:
            return len(unique_labels[unique_labels != -1])
    else:
        raise ValueError("Invalid method. Supported methods are 'kmeans', 'gmm', 'agglomerative', or 'dbscan'.")

def set_kmeans(lat_space):
    """
    Set up K-Means clustering with the optimal number of clusters.

    Args:
        lat_space (ndarray): Latent space data.

    Returns:
        KMeans: K-Means clustering model.
    """
    best_nbr = give_best(lat_space,method='kmeans')
    kmeans = KMeans(n_clusters=best_nbr, random_state=0)
    kmeans.fit(lat_space)
    return kmeans

def set_dbscan(lat_space):
    """
    Set up dbscan clustering with the optimal number of clusters.

    Args:
        lat_space (ndarray): Latent space data.

    Returns:
        dbscan: dbscan clustering model.
    """
    best_nbr = give_best(lat_space,method='dbscan')
    min_samples = min(5, len(lat_space) // 10)
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
    dbscan.fit(lat_space)
    return dbscan


def set_gmm (latent_space, covariance_type='full', random_state=0):
    """
    Cluster the latent space using Gaussian Mixture Models (GMM).
    
    Parameters:
    - latent_space (array-like): The latent space obtained from the encoder of the VAE.
    - n_components (int, optional): The number of mixture components. Default is 3.
    - covariance_type (str, optional): The type of covariance parameters to use. 
                                       Options are 'full', 'tied', 'diag', 'spherical'. Default is 'full'.
    - random_state (int, RandomState instance or None, optional): Determines random number generation for 
                                                                  initialization. Default is None.
    
    Returns:
    - labels (array-like): Cluster labels for each data point in the latent space.
    - gmm (GaussianMixture): Fitted GMM model.
    """
    # Initialize GMM
    n_components = give_best(latent_space,method='gmm')
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    # Fit GMM to the latent space
    gmm.fit(latent_space)
    # Predict cluster labels
    #labels = gmm.predict(latent_space)
    return gmm
