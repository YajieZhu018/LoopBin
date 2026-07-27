"""
LoopBin clusters chromatin loops with a VADE model.
It provides functions for preprocessing, processing, pretraining, training, and clustering.

Author: Yajie Zhu, Alexis Bel
Date: 2024-12-08
"""
import argparse
import os
import sys

# --- Determinism: these MUST be set before TensorFlow is imported (it loads via the
# `src.model.*` imports below). setdefault so an explicit override from the shell still wins. ---
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import pickle
import numpy as np
import matplotlib.pyplot as plt

from loopbin.fn import function
from loopbin.fn import processing
from loopbin.fn.preprocess import run as run_preprocess
from loopbin.plot import plotting
from loopbin.model.ae import AE
from loopbin.model.vade_model import VADE
from sklearn.cluster import KMeans
import tensorflow as tf
import random
tf.keras.backend.set_floatx('float32')
#from sklearn.mixture import GaussianMixture


def _set_seed(seed=None, threads=None):
    """Seed every RNG (python / numpy / tensorflow) for reproducible training.

    The determinism env vars (TF_DETERMINISTIC_OPS, ...) are set at module import, before
    TensorFlow loads. Verified 2026-06-09: with those set + every RNG seeded, two full runs are
    bit-identical even at high thread counts (tested at 32) — so `threads` is purely a speed /
    politeness knob, NOT a determinism switch. Reproducibility holds at any *fixed* thread count;
    just keep it fixed across runs (changing it can flip the last bit of a multithreaded reduction).

    Resolution order for each value: explicit arg > env var (LOOPBIN_SEED / LOOPBIN_THREADS) >
    built-in default (seed 73, threads 16).

    Must be called before the first TF op of the process (TF forbids changing the thread pools once
    the runtime is initialised); each pipeline step is its own process, so the one call at the top of
    pretrain_ae / train_vade is safe.
    """
    if seed is None:
        seed = int(os.environ.get("LOOPBIN_SEED", 73))
    if threads is None:
        threads = int(os.environ.get("LOOPBIN_THREADS", 16))
    print(f"seed = {seed} | threads = {threads} | "
          f"TF_DETERMINISTIC_OPS={os.environ.get('TF_DETERMINISTIC_OPS')}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    return seed


def _loss_weights(d_input):
    """Per-feature weights for the BCE reconstruction loss (Phase 2): make the punctate active
    marks (CTCF/SMC1A/H3K27ac) compete with the broad H3K27me3, so the active class resolves on
    more seeds. Returns None for the default 'uniform' mode -> the models fall back to the
    unmodified BinaryCrossentropy (BYTE-IDENTICAL to upstream; no regression).

    The channel layout is derived from the input size: D = 256 Micro-C + 32 * n_marks, so
    n_marks = (D - 256) // 32. The returned vector w sums to 1 (uniform => 1/D) so it slots into a
    weighted-mean BCE. Selected by env LOOPBIN_LOSS_WEIGHT:
      uniform (default) -> None (no change)
      channel           -> Micro-C and each mark get equal total weight 1/(1+n_marks)
      marks             -> Micro-C 0.5, the marks share 0.5 (gentler; preserves Micro-C weight)
    """
    mode = os.environ.get("LOOPBIN_LOSS_WEIGHT", "uniform").lower()
    if mode == "uniform":
        return None
    n_marks = (d_input - 256) // 32
    if n_marks < 1 or 256 + 32 * n_marks != d_input:
        print(f"[loss-weight] unexpected D={d_input} (not 256+32*n_marks) -> uniform")
        return None
    w = np.zeros(d_input, dtype=np.float32)
    if mode == "channel":
        per = 1.0 / (1 + n_marks)
        w[:256] = per / 256
        for i in range(n_marks):
            w[256 + i * 32: 256 + (i + 1) * 32] = per / 32
    elif mode == "marks":
        w[:256] = 0.5 / 256
        for i in range(n_marks):
            w[256 + i * 32: 256 + (i + 1) * 32] = (0.5 / n_marks) / 32
    else:
        print(f"[loss-weight] unknown LOOPBIN_LOSS_WEIGHT={mode} -> uniform")
        return None
    print(f"[loss-weight] mode={mode} n_marks={n_marks} sum(w)={w.sum():.4f} "
          f"microC={w[:256].sum():.3f} per_mark={w[256:288].sum():.3f}")
    return w


def parse_arguments():
    """Build the subcommand CLI and parse argv."""
    parser = argparse.ArgumentParser(
        prog="loopbin",
        description="LoopBin -- VaDE clustering of chromatin loops (Micro-C + CUT&Tag).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="loopbin 0.1.0")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    def _seed_threads(sp):
        sp.add_argument("-s", "--seed", dest="seed", type=int, default=None,
                        help="random seed (precedence: flag > $LOOPBIN_SEED > 73)")
        sp.add_argument("-t", "--threads", dest="threads", type=int, default=None,
                        help="CPU threads (precedence: flag > $LOOPBIN_THREADS > 16)")
    PROT = "CTCF,H3K27ac,H3K27me3,SMC1A"

    pp = sub.add_parser("preprocess", help="bigWig -> per-chromosome bedgraph")
    pp.add_argument("-b", dest="preprocess", help="path to the bigWig (or 'empty')")
    pp.add_argument("-n", dest="name", help="mark name, e.g. CTCF")
    pp.add_argument("-g", dest="bedgraph_folder", help="output bedgraph folder")
    pp.add_argument("-r", dest="nbr_cpu", default=1, help="number of CPUs")
    pp.add_argument("-res", "--resolution", dest="resolution", type=int, default=10000)
    pp.add_argument("-cs", "--chrom-sizes", dest="chrom_sizes", default=None)
    pp.set_defaults(func=preprocess)

    ps = sub.add_parser("process", help="loops + mcool + bedgraphs -> npy")
    ps.add_argument("-l", dest="list_loop", help="loop .bedpe")
    ps.add_argument("-c", dest="cool_file", help="mcool file")
    ps.add_argument("-g", dest="bedgraph_folder", help="bedgraph folder")
    ps.add_argument("-p", dest="proteins", default=PROT, help="marks (comma-separated)")
    ps.add_argument("-r", dest="nbr_cpu", default=1)
    ps.add_argument("-u", dest="folder", help="output folder")
    ps.add_argument("-res", "--resolution", dest="resolution", type=int, default=10000)
    ps.set_defaults(func=process)

    nm = sub.add_parser("normalize", help="merge + co-normalize conditions")
    nm.add_argument("-e", dest="conditions", help="conditions, e.g. control,degron")
    nm.add_argument("-u", dest="folder", help="the process output folder")
    nm.set_defaults(func=process_all_groups)

    pt = sub.add_parser("pretrain", help="pretrain the autoencoder")
    pt.add_argument("-d", dest="file", help="merged_log_data.npy")
    pt.add_argument("-u", dest="folder", help="output folder")
    _seed_threads(pt)
    pt.set_defaults(func=pretrain_ae)

    tr = sub.add_parser("train", help="train VaDE + cluster")
    tr.add_argument("-num", dest="cluster_number", default=10, help="GMM components")
    tr.add_argument("-d", dest="file", help="merged_log_data.npy")
    tr.add_argument("-if_pre", dest="if_pretrain", default="True")
    tr.add_argument("-pre", dest="pretrained_model", default=None, help="pretrained AE folder")
    tr.add_argument("-ep", dest="epoch_number", default=1000, help="epochs")
    tr.add_argument("-u", dest="folder", help="output folder")
    tr.add_argument("-p", dest="proteins", default=PROT)
    _seed_threads(tr)
    tr.set_defaults(func=train_vade)

    cl = sub.add_parser("cluster", help="predict clusters with a trained model")
    cl.add_argument("-d", dest="file", help="log_data.npy")
    cl.add_argument("-m", dest="model", help="trained model folder")
    cl.add_argument("-u", dest="folder", help="output folder")
    cl.add_argument("-p", dest="proteins", default=PROT)
    cl.set_defaults(func=cluster_data)

    mg = sub.add_parser("merge", help="merge small clusters")
    mg.add_argument("-d", dest="file")
    mg.add_argument("-u", dest="folder")
    mg.add_argument("-k", dest="clusters", help="clusters to merge, e.g. 2,3")
    mg.add_argument("-p", dest="proteins", default=PROT)
    mg.set_defaults(func=merge_small_clusters)

    return parser.parse_args()

def preprocess(args):
    """Preprocessing step: bigWig -> per-chromosome bedgraph.

    Pure-Python (src/fn/preprocess.py); the genome comes from --chrom-sizes (no hardcoded
    mm10), tiled at --resolution. Replaces the old `sh preprocess_local.sh` (mm10-only) call.
    """
    function.verif_preprocess(args)
    # name == "empty" writes a zero track for every chromosome (old empty_preprocess.sh)
    bigwig = "empty" if args.name == "empty" else args.preprocess
    run_preprocess(bigwig, args.name, args.bedgraph_folder,
                   args.resolution, args.chrom_sizes, args.nbr_cpu)
    sys.exit()


def process(args):
    """Processing step"""
    # Verification of the argument use
    cool_file = function.verif_process(args)
    function.verif_folder(args.bedgraph_folder)
    # Process
    processing.process(args.list_loop, cool_file, args.bedgraph_folder, args.proteins,
                       args.nbr_cpu, args.folder, args.resolution)
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
    # Phase 4 speedup: labels/prob/recon are already saved above; skip the diagnostic plotting
    # (cluster plots + t-SNE) for sweeps. Never affects clustering output. Default off => plots as before.
    if os.environ.get("LOOPBIN_NO_PLOTS"):
        return
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
    # Seed python/numpy/tf BEFORE building & fitting the AE. The AE weight init (glorot_uniform)
    # and TF batch shuffle were previously unseeded — the dominant source of run-to-run variation,
    # since `train` fits its GMM off this AE latent space. Uses the same seed/threads as train.
    _set_seed(args.seed, args.threads)
    # shuffle the data
    np.random.shuffle(X)
    # set ae model
    d_input = X.shape[1]
    ae = AE(d_input, loss_weights=_loss_weights(d_input))
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
    # NOTE: removed `function.set_kmeans(z)` + the model_cluster.pkl dump — set_kmeans is undefined
    # in src/fn/function.py (crashes pretrain), and the pickle was unused downstream: `train`
    # refits its own GaussianMixture in vade_model.load_pretrained_weights.

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
    seed = int(os.environ.get("LOOPBIN_SEED", 73))   # env-overridable for the seed sweep; default 73 (code value)
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


def _marginal_entropy_beta():
    """Phase 4: weight beta for the marginal-entropy regularizer (env LOOPBIN_MARGINAL_ENTROPY).
    Default 0.0 => OFF => loss byte-identical to baseline. beta>0 maximizes H(p_bar), the entropy of
    the batch-average GMM responsibility, to stabilize the per-cluster mixing proportions across seeds
    (RIM, Krause et al. NeurIPS 2010; IMSAT, Hu et al. ICML 2017; IIC, Ji et al. ICCV 2019)."""
    try:
        beta = float(os.environ.get("LOOPBIN_MARGINAL_ENTROPY", "0"))
    except ValueError:
        beta = 0.0
    if beta > 0:
        print(f"[marginal-entropy] beta={beta}")
    return beta

def _marginal_kl():
    """Phase 4b KL-to-target. Env LOOPBIN_MARGINAL_KL (beta, default 0=OFF) + LOOPBIN_KL_TARGET
    (comma-sep target size shape). beta>0 anchors the sorted marginal to this fixed UNEQUAL shape
    => proportions reproducible across seeds without being forced uniform."""
    try:
        beta = float(os.environ.get("LOOPBIN_MARGINAL_KL", "0"))
    except ValueError:
        beta = 0.0
    target = None
    tstr = os.environ.get("LOOPBIN_KL_TARGET", "")
    if beta > 0 and tstr:
        target = [float(x) for x in tstr.split(",") if x.strip()]
        s = sum(target); target = [x / s for x in target]
        print(f"[marginal-kl] beta={beta} target={['%.3f' % t for t in target]}")
    return beta, target

def _learning_rate():
    """Train-step base learning rate (env LOOPBIN_LR). Default 0.002 => byte-identical to baseline.
    Feeds the VaDE optimizer AND the scheduler base + floor; floor = lr/10 preserves the baseline
    10:1 ratio so the decay GEOMETRY is unchanged and only the scale moves."""
    try:
        lr = float(os.environ.get("LOOPBIN_LR", "0.002"))
    except ValueError:
        lr = 0.002
    if lr != 0.002:
        print(f"[learning-rate] lr={lr} floor={lr/10}")
    return lr

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
    # Seed everything + deterministic threading (TF_DETERMINISTIC_OPS is set at module top,
    # before TF import). Same seed/threads as pretrain so the whole run is reproducible.
    seed = _set_seed(args.seed, args.threads)
    # load the data
    X = np.load(data_path)
    # no test data for unsupervised learning
    X_train = X
    # set vade model
    d_input = X_train.shape[1]
    _klb, _klt = _marginal_kl()
    vade = VADE(d_input,n_clusters, loss_weights=_loss_weights(d_input), marginal_entropy_beta=_marginal_entropy_beta(), marginal_kl_beta=_klb, marginal_kl_target=_klt)
    vade(np.zeros((10, d_input)))
    #vade.summary()
    # load the pretrain model
    if if_pretrain == 'True':
        vade.load_pretrained_weights(pretrain_model_path,X_train,gmm_name)
    ## Define a learning rate scheduler
    lr0 = _learning_rate()          # env LOOPBIN_LR, default 0.002 == baseline
    decay_nn = 0.9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: max(lr0/10, lr0 * decay_nn ** (epoch//10))  # Apply decay to the learning rate
    )
    # set ae model
    adam_nn= tf.keras.optimizers.Adam(learning_rate=lr0,epsilon=1e-4)
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
    

def cli():
    """Console entry point: parse the subcommand CLI and dispatch."""
    args = parse_arguments()
    args.func(args)


if __name__ == '__main__':
    cli()
