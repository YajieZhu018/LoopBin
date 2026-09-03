"""Module to plot matrix and latent space"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
#import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot_cluster_to_exam(x_image, x_rec,nbr):
    plt.clf()
    x_image = np.split(x_image, 5,axis=2)
    x_rec = np.split(x_rec, 5, axis=2)
    n = 5  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        vmina = np.min([np.min(x_image[i]),np.min(x_rec[i])])
        vmaxa = np.max([np.max(x_image[i]),np.max(x_rec[i])])
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_image[i],cmap="seismic",vmin=vmina,vmax=vmaxa)
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_rec[i],cmap="seismic",vmin=vmina,vmax=vmaxa)
        plt.colorbar()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructio
    plt.savefig(f'../plot/cluster_{nbr}.pdf')
    plt.close()
    #plt.clf()


def plot_score(lat_space, path):
    plt.clf()
    score = []
    sil = []
    for i in range(2,25):
        kmeans = KMeans(n_clusters=i,random_state=0)
        kmeans.fit(lat_space)
        score.append(kmeans.inertia_)
        # subsample the silhouette: the default builds the full n x n distance matrix (O(n^2)
        # time + memory), which made pretrain hang ~43 min on ~44k loops. sample_size keeps the
        # diagnostic faithful while bounding cost. (random_state set for reproducibility.)
        sil.append(silhouette_score(lat_space, kmeans.labels_,
                                    sample_size=min(2000, len(lat_space)), random_state=0))
    x = range(2,25)
    plt.plot(x,score)
    plt.xticks(x, x,)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig(f'{path}/elbow.pdf')
    plt.clf()
    kn = KneeLocator(range(1, len(score) + 1), score, curve='convex', direction='decreasing')
    optimal_num_clusters = kn.knee
    x = range(2,25)
    plt.plot(x,sil)
    plt.xticks(x, x,)
    plt.xlabel("Number of Clusters")
    plt.ylabel("silhouette score")
    plt.savefig(f'{path}/sil.pdf')
    plt.close()
    #plt.clf()

# plot the loss
def plot_loss(history, path):
    """Plot the loss

    Args:
        history (_type_): _description_
    """
    print("loss plotting")
    plt.clf()
    info = history

    first_half = ['loss','reconstruction_loss', 'kl_loss'] #
    second_half = ['val_loss', 'val_reconstruction_loss', 'val_kl_loss']  #

    num_subplots = 3
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots))
    for i, (ori, val) in enumerate(zip(first_half, second_half)):
        ax = axes[i]
        ax.plot(history.history[ori])
        ax.plot(history.history[val])
        if ori == "kl_loss" or ori == "loss":
            plt.yscale("log")
        ax.set_title(f'Model {ori}')
        ax.set_ylabel(ori)
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Validation'], loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}/loss.pdf')
    plt.close()

# plot the loss
def plot_train_loss(history,path):
    """Plot the loss of training

    Args:
        history (_dic_): _description_
    """
    print("loss plotting")
    plt.clf()
    #info = history
    losses = ['loss', 'reconstruction_loss', 'kl_loss']
    num_subplots = 3
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots))
    for i, training in enumerate(losses):
        ax = axes[i]
        if training == 'kl_loss' or training == 'loss':
            ax.set_yscale("log")
        ax.plot(history.history[training])
        ax.set_title(f'Model {training}')
        ax.set_ylabel(training)
        ax.set_xlabel('Epoch')
        ax.legend(['Train'], loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}/loss.pdf')
    plt.close()

def plot_tsne(lat_space, labels, save_dir):
    # Set up directories for saving plots
    os.makedirs(save_dir, exist_ok=True)
    
    # Seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    # Loop over perplexity values for t-SNE
    for nbr in [100]:
        X_embedded = TSNE(
            n_components=2, init='pca', random_state=0,
            learning_rate="auto", perplexity=nbr, n_jobs=3
        ).fit_transform(lat_space)
        
        # Create DataFrame for plotting
        principalDf = pd.DataFrame(data=X_embedded, columns=['component_1', 'component_2'])
        principalDf["label"] = labels.astype(str)
        
        # Save DataFrame to CSV
        principalDf.to_csv(f"{save_dir}/tsne_{nbr}.csv", sep=',', index=False, encoding='utf-8')
        
        # Plot with Matplotlib
        plt.figure(figsize=(8, 6))
        scatter = sns.scatterplot(
            data=principalDf, x="component_1", y="component_2",
            hue="label", palette="viridis", s=60, edgecolor="k", alpha=0.7
        )
        plt.title(f"t-SNE with Perplexity={nbr}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Labels", loc="best")
        
        # Save as PDF
        pdf_path = f"{save_dir}/tsne_perplex{nbr}.pdf"
        plt.savefig(pdf_path, format="pdf")
        plt.close()
        print(f"Saved t-SNE plot with perplexity {nbr} as PDF to {pdf_path}")

def plot_tsne_html(lat_space, labels, save_name_plot):
    if save_name_plot != None:
        if not os.path.exists(f"{save_name_plot}/tsne"):
            os.makedirs(f"{save_name_plot}/tsne")
    else:
        if not os.path.exists("tsne"):
            os.makedirs("tsne")
    for nbr in [50,75,100,125,150]:
        X_embedded = TSNE(n_components=2,  init='pca',random_state=0,learning_rate="auto",  perplexity=nbr, n_jobs=3).fit_transform(lat_space)
        principalDf = pd.DataFrame(data = X_embedded, columns = ['component_1', 'component_2'])
        principalDf["lab"] = labels
        principalDf["lab"] = principalDf["lab"].astype(str)
        if save_name_plot != None:
            principalDf.to_csv(f"{save_name_plot}/tsne/tsne_{nbr}", sep=',', index=False, encoding='utf-8')
        else:
            principalDf.to_csv(f"tsne/tsne_{nbr}", sep=',', index=False, encoding='utf-8')
        figa = px.scatter(principalDf,x='component_1', y='component_2',title=f"perplexity={nbr}",color="lab")
        if save_name_plot != None:
            figa.write_html(f'{save_name_plot}/tsne_{nbr}.html')
        else:
            figa.write_html(f'tsne_{nbr}.html')


def plot_pie( separated_arrays, labels,save_name_plot):
    pourcen = {}
    for i in np.unique(labels):
        pourcen[i] = len(separated_arrays[i])
    plt.pie(pourcen.values(),labels=pourcen.keys(),autopct = lambda x: str(round(x, 1)) + '%',)
    plt.savefig(f'{save_name_plot}/pie.pdf')


def plot_cluster(separated_arrays, labels, separated_reconstruction, save_name_plot, list_epic):
    k = dict()
    rec = dict()
    vmin_loop = []
    vmax_loop = []
    for i in np.unique(labels):
        a = np.mean(separated_arrays[i], axis=0)
        # get channel number
        num_channel = a.shape[2]
        a_split = np.split(a, num_channel, axis=2)
        k[i] = a_split
        b = np.mean(separated_reconstruction[i], axis=0)
        rec_split = np.split(b, num_channel, axis=2)
        rec[i] = rec_split
        vmin_sub = []
        vmax_sub = []
        for j in range(num_channel):
            vmin_sub.append(np.min(a_split[j]))
            vmax_sub.append(np.max(a_split[j]))

        vmin_loop.append(vmin_sub)
        vmax_loop.append(vmax_sub)
    vmin_1 =np.min(vmin_loop,axis=0)
    vmax_1=np.max(vmax_loop,axis=0)
    titles = ["Micro-C"] + list_epic   #, "CTCF", "H3K27ac", "H3K27me3", "SMC1A"]
    for o in np.unique(labels):
        plt.clf()
        plt.figure(figsize=(12, 5))
        for i in range(num_channel):
            ax = plt.subplot(4, num_channel, i + 1)
            plt.imshow(k[o][i], cmap="jet", vmin=vmin_1[i], vmax=vmax_1[i])
            plt.colorbar()
            plt.title(titles[i])  # Utilisation du titre correspondant à l'index i de la liste
            if i == 0:
                ax.set_ylabel('Original data\n general scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, num_channel, i + 1 + num_channel)
            plt.imshow(k[o][i], cmap="jet")
            plt.colorbar()
            if i == 0:
                ax.set_ylabel('Original data\npersonnal scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, num_channel, i + 1 + 2*num_channel)  # Affiche la troisième ligne de la matrice
            plt.imshow(rec[o][i], cmap="jet",vmin=vmin_1[i], vmax=vmax_1[i])
            plt.colorbar()
            if i == 0:
                ax.set_ylabel('Reconstructed \ndata\ngeneral scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, num_channel, i + 1 + 3*num_channel)  # Affiche la troisième ligne de la matrice
            plt.imshow(rec[o][i], cmap="jet")
            plt.colorbar()
            if i == 0:
                ax.set_ylabel('\nReconstructed \ndata\npersonnal scale',rotation=0,labelpad=35)
        plt.tight_layout()  # Ajuste automatiquement l'espacement entre les sous-graphiques
        if save_name_plot != None:
            plt.savefig(f'{save_name_plot}/cluster_{o}.pdf')
        else:
            plt.savefig(f'cluster_{o}.pdf')
        plt.clf()


def _cluster_averages(separated_arrays, labels):
    """
    Per-cluster average image, split into channels.

    Returns (dict_clusters, vmin_shared, vmax_shared) where dict_clusters[cluster] is a
    list of (16,16) arrays (one per channel) and vmin_shared/vmax_shared are per-channel
    limits pooled over every cluster.
    """
    dict_clusters = {}
    vmin_loop = []
    vmax_loop = []
    # loop through each cluster
    for i in np.unique(labels):
        # calcuate the average of the cluster
        average_clusters = np.mean(separated_arrays[i], axis=0)  # shape: (16,16,num_channel)
        # get channel number
        num_channel = average_clusters.shape[2]
        # split the average cluster into each channel, squeezing the trailing singleton
        # axis np.split leaves behind so imshow gets a plain 2-D array
        dict_clusters[i] = [np.squeeze(c, axis=2)
                            for c in np.split(average_clusters, num_channel, axis=2)]
        # get the vmin and vmax of each channel
        vmin_loop.append([np.min(dict_clusters[i][j]) for j in range(num_channel)])
        vmax_loop.append([np.max(dict_clusters[i][j]) for j in range(num_channel)])
    # get final vmin and vmax, shared across clusters so the rows stay comparable
    return dict_clusters, np.min(vmin_loop, axis=0), np.max(vmax_loop, axis=0)


def _label_cbar_ends(cbar, vmin, vmax, fontsize):
    """
    Tick only the two ends of a horizontal colorbar, pushed outward so neighbouring
    columns don't collide, and formatted with %g so a narrow range stays readable --
    %.3f rendered several low-signal panels as "0.002 -> 0.002".
    """
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.3g}', f'{vmax:.3g}'])
    cbar.ax.tick_params(labelsize=fontsize)
    labels = cbar.ax.get_xticklabels()
    if labels:
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')


def _cluster_grid_page(dict_clusters, uniq, channel_names, mode, vmin_shared, vmax_shared):
    """
    One page of the cluster x channel grid.

    mode='shared'    -- every row uses the same per-channel colour limits, so signal
                        levels are comparable across clusters. One colorbar per column.
    mode='per_panel' -- every panel is autoscaled to its own min/max and carries its own
                        colorbar, so faint structure stays visible in low-signal clusters.
                        Levels are NOT comparable across rows on this page.

    A shared linear scale flattens any cluster whose signal is small relative to the
    strongest cluster -- on the epigenetic channels (outer products of the flanking
    signal, so between-cluster ratios get squared) that was most panels. Hence the two
    pages: read them together.
    """
    num_cluster = len(uniq)
    num_channel = len(channel_names)

    if mode == 'shared':
        fig = plt.figure(figsize=(1.6 * num_channel, 1.0 + 1.5 * num_cluster))
        # extra short row at the bottom for the per-column colorbars
        gs = fig.add_gridspec(num_cluster + 1, num_channel,
                              height_ratios=[1] * num_cluster + [0.25],
                              left=0.16, right=0.97, top=0.93, bottom=0.05,
                              hspace=0.15, wspace=0.15)
        for row, o in enumerate(uniq):
            for i in range(num_channel):
                ax = fig.add_subplot(gs[row, i])
                ax.imshow(dict_clusters[o][i], cmap="jet",
                          vmin=vmin_shared[i], vmax=vmax_shared[i])
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(channel_names[i], fontsize=9)
                if i == 0:
                    ax.set_ylabel(f'Cluster {o}', rotation=0, labelpad=35, fontsize=9)
        # one colorbar per column, spanning the whole channel
        for i in range(num_channel):
            cbar_ax = fig.add_subplot(gs[num_cluster, i])
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap="jet",
                                      norm=plt.Normalize(vmin=vmin_shared[i],
                                                         vmax=vmax_shared[i])),
                cax=cbar_ax, orientation='horizontal')
            _label_cbar_ends(cbar, vmin_shared[i], vmax_shared[i], fontsize=7)
        fig.suptitle('Shared colour scale per channel (comparable across clusters)',
                     fontsize=10)
    else:
        fig = plt.figure(figsize=(1.6 * num_channel, 1.0 + 1.8 * num_cluster))
        gs = fig.add_gridspec(num_cluster, num_channel,
                              left=0.16, right=0.97, top=0.93, bottom=0.03,
                              hspace=0.45, wspace=0.25)
        for row, o in enumerate(uniq):
            for i in range(num_channel):
                # split each cell into the image and a thin colorbar underneath it
                inner = gs[row, i].subgridspec(2, 1, height_ratios=[1, 0.12], hspace=0.08)
                ax = fig.add_subplot(inner[0])
                cbar_ax = fig.add_subplot(inner[1])
                panel = dict_clusters[o][i]
                vmin, vmax = float(np.min(panel)), float(np.max(panel))
                image = ax.imshow(panel, cmap="jet", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(channel_names[i], fontsize=9)
                if i == 0:
                    ax.set_ylabel(f'Cluster {o}', rotation=0, labelpad=35, fontsize=9)
                cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
                _label_cbar_ends(cbar, vmin, vmax, fontsize=6)
        fig.suptitle('Per-panel colour scale (structure visible, NOT comparable across clusters)',
                     fontsize=10)
    return fig


def plot_all_clusters(separated_arrays, labels, out_folder, list_epic):
    """
    Average plot of each channel as column and each cluster as row.

    Writes a two-page PDF: page 1 shares one colour scale per channel across clusters
    (so signal levels are comparable), page 2 autoscales every panel (so low-signal
    clusters are still readable). See _cluster_grid_page for why both are needed.
    """
    dict_clusters, vmin_shared, vmax_shared = _cluster_averages(separated_arrays, labels)
    uniq = np.unique(labels)
    channel_names = ["Micro-C"] + list_epic
    with PdfPages(f'{out_folder}/all_clusters.pdf') as pdf:
        for mode in ('shared', 'per_panel'):
            fig = _cluster_grid_page(dict_clusters, uniq, channel_names, mode,
                                     vmin_shared, vmax_shared)
            pdf.savefig(fig)
            plt.close(fig)


def plot_theta_history(theta_history, out_folder, floor_schedule=None):
    """
    Trajectory of the GMM prior weights (theta_p / pi) over training.

    Args:
        theta_history: (n_updates, 1 + n_centroid) array, column 0 the epoch and the
            remaining columns pi per component -- as saved by main.py's EMPriorUpdate.
        out_folder: folder to write theta_history.pdf into.
        floor_schedule: optional callable epoch -> floor, drawn as a dashed reference so
            a component sitting on the floor is distinguishable from one that has
            genuinely shrunk.
    """
    theta_history = np.asarray(theta_history)
    epochs = theta_history[:, 0]
    pis = theta_history[:, 1:]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(pis.shape[1]):
        ax.plot(epochs, pis[:, c], marker='o', markersize=2.5, label=f'component {c}')
    if floor_schedule is not None:
        ax.plot(epochs, [floor_schedule(int(e)) for e in epochs], 'k--', linewidth=1.2,
                label='floor')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\pi_c$')
    ax.set_title('GMM prior mixture weights over training')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(f'{out_folder}/theta_history.pdf')
    plt.close(fig)
