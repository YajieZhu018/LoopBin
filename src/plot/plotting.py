"""Module to plot matrix and latent space"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import os


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
        sil.append(silhouette_score(lat_space, kmeans.labels_))
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
    # plot the loss
    #val_rep1 = [np.mean(i) for i in history.history["val_reconstruction_loss_rep1"] ]
    #val_rep2 = [np.mean(i) for i in history.history["val_reconstruction_loss_rep2"] ]
    #val_recons = [np.mean(i) for i in history.history["val_reconstruction_loss"]  ]
    #val_kl = [np.mean(i) for i in history.history["val_kl_loss"]  ]
    #history.history["val_reconstruction_loss_rep1"] = val_rep1
    #history.history["val_reconstruction_loss_rep2"] = val_rep2
    #history.history["val_kl_loss"] = val_kl
    #history.history["val_reconstruction_loss"] = val_recons
    #first_half = ['loss', 'reconstruction_loss', 'kl_loss', 'reconstruction_loss_rep1', 'reconstruction_loss_rep2']
    #second_half = ['val_loss', 'val_reconstruction_loss', 'val_kl_loss', 'val_reconstruction_loss_rep1', 'val_reconstruction_loss_rep2']


    first_half = ['loss','reconstruction_loss', 'kl_loss'] # 
    second_half = ['val_loss', 'val_reconstruction_loss', 'val_kl_loss']  # 

    num_subplots = 3
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots))
    for i, (ori, val) in enumerate(zip(first_half, second_half)):
        ax = axes[i]
        ax.plot(history.history[ori])
        ax.plot(history.history[val])
        if ori == "kl_loss":
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
        ax.plot(history.history[training])
        ax.set_title(f'Model {training}')
        ax.set_ylabel(training)
        ax.set_xlabel('Epoch')
        ax.legend(['Train'], loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}/loss.pdf')
    plt.close()

# plot loss with cyclical annealing strategy
#def plot_loss_new(history,epochs):
#    # Plot the training and validation loss
#    for key in ['loss', 'reconstruction_loss', 'kl_loss']:
#        plt.plot(range(1, epochs+1), history[key], label='Training Loss')
#        plt.plot(range(1, epochs+1), history['val_' + key], label='Validation Loss')
#        plt.xlabel('Epochs')
#        plt.ylabel(key)
#        plt.title('Training and Validation Loss with Cyclical Annealing Beta')
#        plt.legend()
#        plt.savefig('../plot/' + key + '.pdf')

def plot_tsne(lat_space, labels, save_name_plot):
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


def plot_cluster(separated_arrays, labels, separated_reconstruction, save_name_plot):
    k = dict()
    rec = dict()
    vmin_loop = []
    vmax_loop = []
    for i in np.unique(labels):
        a = np.mean(separated_arrays[i], axis=0)
        a_split = np.split(a, 5, axis=2)
        k[i] = a_split
        b = np.mean(separated_reconstruction[i], axis=0)
        rec_split = np.split(b, 5, axis=2)
        rec[i] = rec_split
        vmin_sub = []
        vmax_sub = []
        for j in range(5):
            vmin_sub.append(np.min(a_split[j]))
            vmax_sub.append(np.max(a_split[j]))

        vmin_loop.append(vmin_sub)
        vmax_loop.append(vmax_sub)
    vmin_1 =np.min(vmin_loop,axis=0)
    vmax_1=np.max(vmax_loop,axis=0)
    titles = ["Micro-C", "CTCF", "H3K27ac", "H3K27me3", "SMC1A"]
    for o in np.unique(labels):
        plt.clf()
        plt.figure(figsize=(10, 5))
        for i in range(5):
            ax = plt.subplot(4, 5, i + 1)
            plt.imshow(k[o][i], cmap="jet", vmin=vmin_1[i], vmax=vmax_1[i])
            plt.colorbar()
            plt.title(titles[i])  # Utilisation du titre correspondant à l'index i de la liste
            if i == 0:
                ax.set_ylabel('Original data\n general scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, 5, i + 1 + 5)
            plt.imshow(k[o][i], cmap="jet")
            plt.colorbar()
            if i == 0:
                ax.set_ylabel('Original data\npersonnal scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, 5, i + 1 + 10)  # Affiche la troisième ligne de la matrice
            plt.imshow(rec[o][i], cmap="jet",vmin=vmin_1[i], vmax=vmax_1[i])
            plt.colorbar()
            if i == 0:
                ax.set_ylabel('Reconstructed \ndata\ngeneral scale',rotation=0,labelpad=35)
            ax = plt.subplot(4, 5, i + 1 + 15)  # Affiche la troisième ligne de la matrice
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


