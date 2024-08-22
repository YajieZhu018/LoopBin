"""Module to prepare the cluster using PCs as input.
The functions include:
    - Extract submatrices of loops from Hi-C and epigenetics data
    - Extract PCs from each dataset
    - Concatenate PCs and save it
"""

import multiprocessing
import os
import sys
import cooler
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

def load_array(filename):
    """
    Loads a NumPy array from a file and returns it.

    Args:
        filename (str): The path to the file containing the array.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    with open(filename, 'rb') as file_handle:
        data = np.load(file_handle)
    return data

def load_array(filename):
    """
    Loads a NumPy array from a file and returns it.

    Args:
        filename (str): The path to the file containing the array.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    with open(filename, 'rb') as file_handle:
        data = np.load(file_handle)
    return data

def call_PCA(data, num_pc):
    """
    Call the first several PCs from the data. 
    The number of PCs is controled by num_pc.

    Args:
        data (numpy arr): the single dataset to call pc from. It has the shape (num_loop, 32) for epigenetics data or (num_loop, 256) for Hi-C
        num_pc (str): the number of PCs to call
    Yields:
        pc (numpy arr): the PCs with the shape (num_loop, num_pc)
    """
    # calculate pca
    pca = PCA(n_components=num_pc)
    pc = pca.fit(data).transform(data)
    return pc

def call_PCA_epigenetics(epigenetic_data, num_pc):
    """
    Call the first several PCs from the epigentic data and concatenate them. 
    The number of PCs is controled by num_pc.

    Args:
        epigenetic_data (numpy arr): the data containing four epigenetic marks as four channels. It has the shape (num_loop, 128)
        num_pc (str): the number of PCs to call
    Yields:
        pc (numpy arr): the PCs with the shape (num_loop, num_pc * 4)
    """
    # initial an empty list to store the pc
    epigenetic_pc_list = []
    # loop over all the channel
    for i in range(4):
        # separate each epigenetic file
        start = i * 32
        end = (i + 1) * 32
        data = epigenetic_data[:,start:end]
        pc = call_PCA(data, num_pc)
        epigenetic_pc_list.append(pc)
    epigenetic_pc = np.concatenate(tuple(epigenetic_pc_list), axis = 1)
    return epigenetic_pc

    

def save_pc(final_pc, dir_path, outfilename):
    """
    Save pc to files.

    Args:
        micro_c_pc (numpy.ndarray): The PCs called from micro_c data, shape: num_loop, num_pc
        epigenetic_pc (numpy.ndarray): The PCs called from all the epigenetic data, shape: num_loop, num_pc * 4
        dir_path (str): The directory path.
        outfilename (str): the name of the output .npy file
    """

    # create the out directory if it does not exist
    if not os.path.exists(f"{dir_path}"):
        os.makedirs(f"{dir_path}")
    with open(f'{dir_path}/{outfilename}', 'wb') as file_pointer:
        np.save(file_pointer, final_pc)
    print(f"Data \"pca.npy\" ready to use in {dir_path}")

def process_pca(input_directory, output_directory, num_pc_micro_c=4, num_pc_epigenetics=2, outfilename='pca.npy'):
    # load log min-max normalized micro-C
    micro_c_data = load_array(f'{input_directory}/log_micro_c.npy')
    # load log min-max normalized epigenetics
    epigenetic_data = load_array(f'{input_directory}/log_epigenetic.npy')
    # get pca
    micro_c_pc = call_PCA(micro_c_data, num_pc_micro_c)
    epigenetic_pc = call_PCA_epigenetics(epigenetic_data, num_pc_epigenetics)
    # concatenate the PCs from micro-C and epigenetics data
    final_pc = np.concatenate((micro_c_pc, epigenetic_pc), axis = 1)
    # save it 
    save_pc(final_pc, output_directory, outfilename)

def test_hello():
    print('hello')