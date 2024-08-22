"""Module to load the files and format files"""

import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import repeat

def add_noise(x_train, noise_level=0.01):
    """
    add random Gaussian noise to the input
    Args:
        x_train (numpy.ndarray): the training data
        noise_level (float): the stardard diviation of Gaussian noise
    Returns:
        x_train_noisy (numpy.ndarray): the training data with noise added
    """
    # create a Gaussian noise matrix of the same shape of x_train
    noise = np.random.normal(0, noise_level, x_train.shape)
    x_train_noisy = x_train + noise
    return x_train_noisy 

def construct_train(data):
    """
    Construct the training and validation sets from the given training data.

    Args:
        data (numpy.ndarray): The training data.

    Returns:
        tuple: A tuple containing the training data, validation data,
        training indices, and validation indices.
    """
    # add different level of noise
    #noise_levels = [0, 0.01, 0.02, 0.03, 0.04]
    #data = np.concatenate(list(map(add_noise, repeat(original_data), noise_levels)), axis = 0)
    indices = np.arange(data.shape[0])
    indices_train, indices_test = prepare_input(indices)
    x_train = data[indices_train]
    x_test = data[indices_test]
    # repeat original data to make it the same size as data
    #original_data = np.concatenate([original_data]*5, axis = 0)
    #y_train = original_data[indices_train]
    #y_test = original_data[indices_test]
    return x_train, x_test, indices_train, indices_test


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


def load_data(filename):
    """
    Loads data from a file.

    Args:
        filename (str): The path to the file containing the data.

    Returns:
        numpy.ndarray: The loaded data.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as file_handle:
            data = np.load(file_handle)
        return data
    raise FileNotFoundError(f"The file {filename} wasn't found")


def prepare_input(data, ratio=0.2):
    """
    Splits the data into training and test sets.

    Args:
        data (numpy.ndarray): The data to split.
        ratio (float): The proportion of data to use for testing.

    Returns:
        tuple: A tuple containing the training data, testing data,
                training labels, and testing labels.
    """
    return train_test_split(data, test_size=ratio, random_state=1)


def load_label(filename):
    """
    Loads label data from a file and returns it as a list of strings.

    Args:
        filename (str): The path to the file containing the labels.

    Returns:
        list: A list of label strings.
    """
    with open(filename, 'r', encoding="utf-8") as file_handle:
        label = file_handle.read()
    label = [label[i:i+2] for i in range(0, len(label), 2)]
    return label


def construct_input(file_name):
    """
    Load data from a file, add noise, split it into training and test sets,
    and return the resulting arrays.

    Args:
        file_name (str): The name of the file containing the data.

    Returns:
        tuple: A tuple containing four arrays:
            - x_train (numpy.ndarray): The training set.
            - x_test (numpy.ndarray): The test set.
            - indices_train (numpy.ndarray): The indices of the training set
                in the original data.
            - indices_test (numpy.ndarray): The indices of the test set
                in the original data.
    """
    data = load_data(file_name)
    return construct_train(data)

