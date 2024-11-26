### visualize the pretrained latent space to see if it presents a Gaussian mixture pattern
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
## Data importing
# path of data and model
output_folder = '/usr/users/yzhu1/LoopBin/trials/saved_models/'
data_file = '/usr/users/yzhu1/LoopBin/data/02_process/control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/merged_log_data.npy'
pretrained_model_path = f'{output_folder}pretrain_ae_control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/'
# load the data and model
x = np.load(data_file)
# subselect 10000 points
np.random.seed(0)
np.random.shuffle(x)
data = x[:8000]
model = load_model(pretrained_model_path)
# predict latent space
z = model.encoder(data)
# apply GMM
n_clusters = 8
g = GaussianMixture(n_components=n_clusters, covariance_type='diag', init_params='kmeans', n_init=10, random_state=7) # diag or spherical
g.fit(z)
# Extract GMM centers
gmm_centers = g.means_

## tSNE
# Initialize t-SNE and reduce the dimensionality to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, init='pca',learning_rate="auto",  perplexity=50, n_jobs=3)
latent_tsne = tsne.fit_transform(z)
# Transform GMM centers into the t-SNE space
# Use the same t-SNE object initialized with the previous data parameters
tsne_centers = tsne.fit_transform(np.vstack((z, gmm_centers)))[-n_clusters:]  # Extract only the 2-dimensionized GMM centers

# Plot the 2D t-SNE representation
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], cmap='viridis', s=10, alpha=0.6, label='latent space')
plt.scatter(tsne_centers[:, 0], tsne_centers[:, 1], s=100, c='red', marker='X', label='GMM Centers')
# Add a legend
plt.legend(*scatter.legend_elements())
plt.title('t-SNE plot of the latent space')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
pdf = f'{pretrained_model_path}tSNE_latant_GMM_8centers_diag.pdf'
plt.savefig(pdf)
plt.close()

