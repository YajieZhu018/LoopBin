
"""Module for processing and analyzing Hi-C data.

This module provides functions for processing and analyzing Hi-C data,
including:
- Loading and preprocessing data from various file formats
- Extracting submatrices from a Hi-C matrix
- Extracting epigenetic signals from bedgraph files
- Normalizing and transforming data
- Saving and loading processed data
The main function is input_creator_r that will run the function task()
"""

import multiprocessing
import os
import sys
import cooler
import numpy as np
from tqdm import tqdm

#  Setting variables

# Part of the name for each proteine
name_epi = ["_CTCF8K.bedgraph", "_H3K27ac8K.bedgraph",
            "_H3K27me38K.bedgraph", "_SMC1A8K.bedgraph"]


def read_file(data_file):
    """
    Read a file as a generator.

    Args:
        data_file (str): The path to the file.

    Yields:
        str: Each line of the file.
    """
    with open(data_file, "r", encoding="utf-8") as file_pointer:
        for line in file_pointer:
            yield line


def log_max_min_normalize(arr):
    """
    Perform logarithmic max-min normalization on a numpy array.

    Args:
        arr (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The normalized array.
    """
    # Convert to float32
    arr = np.float32(arr)
    # Logarithmic transformation
    arr = np.log10(arr + 1)
    # Suppress/hide the warning Invalid value
    np.seterr(invalid='ignore')
    # Normalize to 0-1
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    # Replace NaN with 0
    arr = np.nan_to_num(arr, nan=np.float32(0))
    return arr


def list_chrom():
    """
    Create a list of chromosome names.

    Returns:
        list: A list of chromosome names.
    """
    chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
    return chromosomes


def calcualte_submatrix_region_with_loop_midpoint(midpoint):
    """
    Calculate the start and end positions of a submatrix based on the loop
    anchor midpoint.

    Args:
        midpoint (int): The midpoint of the loop anchor.

    Returns:
        tuple: The start and end positions of the submatrix as integers.
    """
    start = 8000 * (midpoint // 8000 - 16 / 2)
    end = 8000 * (midpoint // 8000 + 16 / 2)
    return int(start), int(end)


# To create the list of name for the epigenetic files
def create_list_epi(file_epi, epigenetics_path):
    """
    Create a list of epigenetic file names.

    Args:
        file_epi (list): The list of epigenetic file name suffixes.
        epigenetics_path (str): The path to the epigenetics folder.

    Returns:
        dict: A dictionary where each chromosome is a key and the value is
        a list of corresponding epigenetic file names.
    """
    chromosome = list_chrom()
    new_sample_name = {}
    for chrom in chromosome:
        new_sample_name[chrom] = []
        for name in file_epi:
            new_sample_name[chrom].append(epigenetics_path + "/" + chrom + name)
    return new_sample_name


def cool_to_matrix(hic, first_anchor, second_anchor):
    """
    Convert a .cool file to a matrix.

    Args:
        hic (cooler.Cooler): The .cool file object.
        chrom1 (str): The chromosome of the first anchor.
        start1 (int): The start position of the first anchor.
        end1 (int): The end position of the first anchor.
        chrom2 (str): The chromosome of the second anchor.
        start2 (int): The start position of the second anchor.
        end2 (int): The end position of the second anchor.

    Returns:
        numpy.ndarray or None: The matrix if conversion is successful,
        None otherwise.
    """
    chrom1, start1, end1 = first_anchor
    chrom2, start2, end2 = second_anchor
    try:
        mat = hic.matrix(balance=True, sparse=False).fetch(chrom1 + ":" +
                                                           str(start1) +
                                                           "-" + str(end1),
                                                           chrom2 + ":" +
                                                           str(start2) +
                                                           "-" + str(end2))
    except ValueError as exception:
        print(f"Error: {exception} \n {chrom1}:{start1}-{end1} {chrom2}:\
            {start2}-{end2}")
        return None
    # set a cutoff for nan number
    nan_cutoff = 16 * 16 / 2
    # don't keep poor submatrix whose number of nan > cutoff
    if np.count_nonzero(np.isnan(mat)) > nan_cutoff:
        return None
    return mat


def extract_signal_from_bedgraph(name_file, chrom, start, end):
    """
    Extract signal from a bedgraph file.

    Args:
        name_file (dict): A dictionary containing chromosome-wise
        lists of file names.
        chrom (str): The chromosome for signal extraction.
        start (int): The start position of the signal.
        end (int): The end position of the signal.

    Returns:
        numpy.ndarray: An array containing the extracted signal.
    """
    # calculate line index: from mth line to nth line
    starting = int(start / 8000)
    ending = int(end / 8000) - 1
    # generate an empty numpy array
    arr = np.empty([ending-starting+1, len(name_file[chrom])])
    for i, file in enumerate(name_file[chrom]):
        # read lines between m to n
        with open(file, "r", encoding="utf-8") as file_pointer:
            for pos, line in enumerate(file_pointer):
                if starting <= pos <= ending:
                    # get the signal of a certain line
                    arr[pos-starting, i] = line.strip().split("\t")[3]
                if pos > ending:
                    break
    return arr


def merge_matrices_to_multichannel_image(matrix_list):
    """
    Merge matrices of the same size into a multi-channel image.

    Args:
        matrix_list (List[numpy.ndarray]): List of matrices to merge.

    Returns:
        numpy.ndarray: The merged multi-channel image.

    Raises:
        ValueError: If the matrices have different sizes.
    """
    size_set = {mat.shape for mat in matrix_list}
    if len(size_set) != 1:
        raise ValueError("Les matrices doivent être de même taille")

    height = 16
    width = 16
    multichannel_matrix = np.zeros((height, width, 5), dtype='float')

    for channel, matrix in enumerate(matrix_list):
        multichannel_matrix[..., channel] = matrix
    return multichannel_matrix


def task(hic, first_anchor, second_anchor, name_file, i):
    """
    Execute a task with submatrices for multiprocessing

    Args:
        hic (cooler.Cooler): The .cool file object.
        chrom1 (str): The chromosome of the first anchor.
        start1 (int): The start position of the first anchor.
        end1 (int): The end position of the first anchor.
        chrom2 (str): The chromosome of the second anchor.
        start2 (int): The start position of the second anchor.
        end2 (int): The end position of the second anchor.
        name_file (dict): A dictionary containing chromosome-wise
        lists of file names.
        i (int): The index of the task.

    Returns:
        Tuple or None: A tuple containing the extracted signal,
        flattened matrix, and task index if successful,
        None otherwise.
    """
    chrom1, start1, end1 = first_anchor
    chrom2, start2, end2 = second_anchor
    mat = cool_to_matrix(hic, first_anchor, second_anchor)
    if mat is None:

        return None
        # get array of epi
    arr1 = extract_signal_from_bedgraph(name_file, chrom1, start1, end1)
    arr2 = extract_signal_from_bedgraph(name_file, chrom2, start2, end2)
    arr = np.concatenate((arr1, arr2))

    return arr, mat.flatten(), i


def check_cpu_count(nbr_cpu):
    """
    Check the available CPU count on the machine.

    Args:
        n (int): The desired number of CPUs.

    Returns:
        bool: True if the desired number of CPUs is available, False otherwise.

    Raises:
        SystemExit: If the desired number of CPUs is not available.
    """
    available_cpu_count = multiprocessing.cpu_count()
    if 0 < nbr_cpu <= available_cpu_count:
        return True
    sys.exit(f"Machine has {available_cpu_count} cpu but you asked for {nbr_cpu} cpu")


def input_creator_r(loop_file, cool_file, bedgraph, n_cpu):
    """
    Create input data from input files using multiprocessing.

    Args:
        loop_file (str): Path to the loop file.
        cool_file (str): Path to the .cool file.
        bedgraph (str): Path to the bedgraph files.
        n_cpu (int): Number of CPUs to use for multiprocessing.

    Returns:
        list: List of tuples containing the input data.

    Raises:
        SystemExit: If the specified number of CPUs is not availabl
    """
    check_cpu_count(n_cpu)
    # a list of the file names epigenetics
    name_file = create_list_epi(name_epi, bedgraph)
    # Load file .mcool
    # initiate the variables
    loop = read_file(loop_file)
    output = []
    i = -1
    hic = cooler.Cooler(cool_file)
    with multiprocessing.Pool(n_cpu) as pool:
        results = []
        for loop in read_file(loop_file):
            i += 1
            chrom1, loop_start1, loop_end1, chrom2, loop_start2, loop_end2 =\
                loop.strip().split("\t")[:6]
            # Calclating the midpoints
            midpoint1 = (int(loop_start1) + int(loop_end1)) // 2
            midpoint2 = (int(loop_start2) + int(loop_end2)) // 2

            # Taking the position around the midpoints
            start1, end1 = calcualte_submatrix_region_with_loop_midpoint(
                midpoint1)
            start2, end2 = calcualte_submatrix_region_with_loop_midpoint(
                midpoint2)

            # Filtering the position
            ranges1 = [(start1, end1), (start1 - 8000, end1 - 8000),
                       (start1 + 8000, end1 + 8000), (start1 - 8000 * 2, end1 -
                                                      8000 * 2), (start1 + 8000
                                                                  * 2, end1 +
                                                                  8000 * 2)]
            for start, end in ranges1:
                intersected = list(set(range(start, end, 8000)).intersection(
                    range(start2, end2, 8000)))
                if intersected:
                    break
            if intersected:
                continue
            ranges2 = [(start2, end2), (start2 - 8000, end2 - 8000),
                       (start2 + 8000, end2 + 8000),
                       (start2 - 8000 * 2, end2 - 8000 * 2),
                       (start2 + 8000 * 2, end2 + 8000 * 2)]
            for start, end in ranges2:
                intersected = list(set(range(start1, end1, 8000)).intersection(
                    range(start, end, 8000)))
                if intersected:
                    break
            if intersected:
                continue
            result = pool.apply_async(task, (hic, [chrom1, start1, end1],
                                             [chrom2, start2, end2],
                                             name_file, i))
            results.append(result)
        # Récupération des résultats
        for result in tqdm(results):
            final_result = result.get()
            if final_result is not None:
                output.append(final_result)
        return output


def mise_en_forme(data_filtered):
    """
    Format the filtered data.

    Args:
        data_filtered (list): List of filtered data.

    Returns:
        tuple: Tuple containing the formatted epigenetic data,
        micro_c data, and numero data.

    """
    epigenetic = np.ones((len(data_filtered), 32, 4))
    micro_c = np.ones((len(data_filtered), 256))
    numero = np.ones((len(data_filtered), 1), dtype='int')

    for i, data in enumerate(data_filtered):
        epigenetic[i] = data[0]
        micro_c[i] = data[1]
        numero[i] = data[2]
    # Tri des tableaux par rapport à numero
    indices_tri = np.argsort(numero.flatten())
    sorted_epigenetic = epigenetic[indices_tri]
    sorted_micro_c = micro_c[indices_tri]
    sorted_numero = numero[indices_tri]
    return sorted_epigenetic, sorted_micro_c, sorted_numero


def extract_line(loop_file, numbers, name):
    """
    Extract specific lines from a file and write them to another file.

    Args:
        loop_file (str): Path to the input file.
        numbers (list): List of line numbers to extract.
        name (str): Path to the output file.

    """
    with open(loop_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    with open(name, 'w', encoding="utf-8") as output_file:
        for index in numbers:
            if 0 <= index < len(lines):
                output_file.write(lines[int(index)])


def save_raw(epigenetic, micro_c, line_number, dir_path):
    """
    Save raw data to files.

    Args:
        epigenetic (numpy.ndarray): The raw epigenetic data.
        micro_c (numpy.ndarray): The raw micro_c data.
        line_number (numpy.ndarray): The line numbers.
        dir_path (str): The directory path.

    """
    if not os.path.exists(f"{dir_path}/raw"):
        os.makedirs(f"{dir_path}/raw")
    with open(f'{dir_path}/raw/raw_epigenetic.npy', 'wb') as file_pointer:
        np.save(file_pointer, epigenetic)
    with open(f'{dir_path}/raw/raw_micro_c.npy', 'wb') as file_pointer:
        np.save(file_pointer, micro_c)
    with open(f"{dir_path}/loop_number.npy", "wb") as file_pointer:
        np.save(file_pointer, line_number)


def save_log(epigenetic, micro_c, dir_path):
    """
    Save log-transformed data to files.

    Args:
        epigenetic (numpy.ndarray): The log-transformed epigenetic data.
        micro_c (numpy.ndarray): The log-transformed micro_c data.
        file_pointer (str): The directory path.

    """
    if not os.path.exists(f"{dir_path}/log"):
        os.makedirs(f"{dir_path}/log")
    with open(f'{dir_path}/log/log_epigenetic.npy', 'wb') as file_pointer:
        np.save(file_pointer, epigenetic)
    # save the hic
    with open(f'{dir_path}/log/log_micro_c.npy', 'wb') as file_pointer:
        np.save(file_pointer, micro_c)


def log_epi(sorted_epigenetic):
    """
    Perform logarithmic transformation on sorted epigenetic data.

    Args:
        sorted_epigenetic (numpy.ndarray): The sorted epigenetic data.

    Returns:
        numpy.ndarray: The logarithmically transformed epigenetic data.
    """
    ctcf, hac, hme, smc1 = splitting(sorted_epigenetic, 2)
    ctcf = np.reshape(ctcf, (len(sorted_epigenetic), 32))
    hac = np.reshape(hac, (len(sorted_epigenetic), 32))
    hme = np.reshape(hme, (len(sorted_epigenetic), 32))
    smc1 = np.reshape(smc1, (len(sorted_epigenetic), 32))
    log_ct = log_max_min_normalize(ctcf)
    log_hac = log_max_min_normalize(hac)
    log_hme = log_max_min_normalize(hme)
    log_sm = log_max_min_normalize(smc1)
    log_epigenetic = np.concatenate((log_ct, log_hac, log_hme, log_sm), axis=1)
    return log_epigenetic


def log_microc(sorted_micro_c):
    """
    Perform logarithmic transformation on sorted micro-C data.

    Args:
        sorted_micro_c (numpy.ndarray): The sorted micro-C data.

    Returns:
        numpy.ndarray: The logarithmically transformed micro-C data.
    """
    micro_c = np.nan_to_num(sorted_micro_c)
    log_micro_c = log_max_min_normalize(micro_c)
    return log_micro_c


def splitting(log_epigenetic, nbr_axe):
    """
    Split the log_epigenetic array into four parts along axis 1.

    Args:
        log_epigenetic (ndarray): Input array to be split.

    Returns:
        tuple: A tuple containing the split arrays log_ct, log_hac, log_hme, and log_sm.
    """
    split_result = np.split(log_epigenetic, 4, axis=nbr_axe)
    log_ct, log_hac, log_hme, log_sm, *_ = split_result
    return log_ct, log_hac, log_hme, log_sm


def create_data(log_epigenetic, log_micro_c):
    """
    Create multichannel matrices from log-transformed epigenetic
    and micro-C data.

    Args:
        log_epigenetic (numpy.ndarray): The log-transformed epigenetic data.
        log_micro_c (numpy.ndarray): The log-transformed micro-C data.

    Returns:
        numpy.ndarray: The multichannel matrices combining different channels.
    """
    mem_h3ac = np.empty((log_epigenetic.shape[0], 16, 16))
    mem_h3me = np.empty((log_epigenetic.shape[0], 16, 16))
    mem_smc1 = np.empty((log_epigenetic.shape[0], 16, 16))
    mem_ctcf = np.empty((log_epigenetic.shape[0], 16, 16))
    log_ct, log_hac, log_hme, log_sm = splitting(log_epigenetic, 1)
    # give the epigenetics data as matrix
    for nbr, i in enumerate(log_ct):
        mem_ctcf[nbr] = np.outer(i[16:], i[:16])
        mem_h3ac[nbr] = np.outer(log_hac[nbr][16:], log_hac[nbr][:16])
        mem_h3me[nbr] = np.outer(log_hme[nbr][16:], log_hme[nbr][:16])
        mem_smc1[nbr] = np.outer(log_sm[nbr][16:], log_sm[nbr][:16])

    log_micro_c = np.reshape(log_micro_c, (log_ct.shape[0], 16, 16))
    multichannel_matrix = np.zeros((log_ct.shape[0],  16, 16, 5),
                                   dtype='float')
    for i in range(log_ct.shape[0]):
        multichannel_matrix[i] = merge_matrices_to_multichannel_image(
            [log_micro_c[i], mem_ctcf[i], mem_h3ac[i], mem_h3me[i],
             mem_smc1[i]])
    return multichannel_matrix


def process(loop_file, hic, bedgraph, nbr_cpu, folder):
    """
    Process loop file data, generate multichannel matrices,
    and save the processed data.

    Args:
        loop_file (str): Path to the loop file.
        hic (str): Path to the Hi-C data file.
        bedgraph (str): Path to the bedgraph file.
        nbr_cpu (int): Number of CPU cores to use for processing.
        folder (str): Path to the directory to save the processed data.

    Returns:
        None
    """
    if folder is not None and os.path.exists(folder):
        dir_path = f"{folder}"
        name = f"{folder}/loop_file_analysis.bedpe"
    else:
        dir_path = "save"
        name = "loop_file_analysis.bedpe"

    nbr_cpu = nbr_cpu or 1

    data_filtered = input_creator_r(loop_file, hic, bedgraph, int(nbr_cpu))
    sorted_epigenetic, sorted_micro_c, sorted_numero = mise_en_forme(data_filtered)

    i = 0
    while os.path.exists(dir_path):
        i += 1
        dir_path += str(i)

    os.makedirs(dir_path)

    extract_line(loop_file, sorted_numero, name)
    save_raw(sorted_epigenetic, sorted_micro_c, sorted_numero, dir_path)

    #log_epigenetic = log_epi(sorted_epigenetic)
    #log_micro_c = log_microc(sorted_micro_c)
    #save_log(log_epigenetic, log_micro_c, dir_path)
#
    #multichannel_matrix = create_data(log_epigenetic, log_micro_c)
    #np.save(f"{dir_path}/merged_2D_data.npy", multichannel_matrix)


def process_all_groups(condstring, folder):
    """
    Merge the processed micro-C and cut&tag data and log normalize them together
    Separate the log normalized data into the respective conditions
    Args:
        condstring (str): condition list
        folder: input folder
    Yields:
        log_data.npy of merged and in each condition
    """
    # get the number of loops in each condition
    conds = condstring.split(",")
    list_raw_micro_c = [np.load(f'{folder}/{cond}/raw/raw_micro_c.npy') for cond in conds]
    list_raw_epig = [np.load(f'{folder}/{cond}/raw/raw_epigenetic.npy') for cond in conds]
    list_number_loops = [x.shape[0] for x in list_raw_micro_c]
    # merge conditions
    merged_micro_c = np.concatenate(tuple(list_raw_micro_c), axis=0)
    merged_epig = np.concatenate(tuple(list_raw_epig), axis=0)
    # log normalization
    log_micro_c = log_microc(merged_micro_c)
    log_epigenetic = log_epi(merged_epig)
    # generate merged data 
    merged = np.concatenate((log_micro_c, log_epigenetic), axis=1)
    np.save(f"{folder}/merged_log_data.npy", merged)
    print(f"Data \"merged_log_data.npy\" ready to use in {folder}")
    # separate conditions
    split_indices = np.cumsum(list_number_loops)
    submatrices = np.split(merged, split_indices, axis=0)
    # save to the condition subfolder
    for i in range(len(conds)):
        cond = conds[i]
        submatrix = submatrices[i]
        np.save(f'{folder}/{cond}/log_data.npy', submatrix)

