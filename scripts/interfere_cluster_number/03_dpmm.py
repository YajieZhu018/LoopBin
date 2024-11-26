### visualize the pretrained latent space to see if it presents a Gaussian mixture pattern
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
## Data importing
# path of data and model
data_file = '/usr/users/yzhu1/LoopBin/data/02_process/control_rep1_H3K4me1/merged_log_data.npy'
pretrained_model_path = '/usr/users/yzhu1/LoopBin/trials/saved_models/pretrain_ae_control_rep1_H3K4me1/'
# load the data and model
x = np.load(data_file)
# subselect 10000 points
np.random.seed(0)
np.random.shuffle(x)
data = x[:10000]
model = load_model(pretrained_model_path)
# predict latent space
z = model.encoder(data)

### Fit DPMM
from sklearn.mixture import BayesianGaussianMixture
# Fit Dirichlet Process Gaussian Mixture Model
dpm = BayesianGaussianMixture(n_components=10, weight_concentration_prior_type='dirichlet_process')
dpm.fit(z)
# Predict the cluster assignments
labels = dpm.predict(z)
print (set(list(labels)))
# Plot the results
from sklearn.manifold import TSNE
# Initialize t-SNE and reduce the dimensionality to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, init='pca',learning_rate="auto",  perplexity=50, n_jobs=3)
latent_tsne = tsne.fit_transform(z)
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.title('Clustering with Dirichlet Process Mixture Model')
plt.legend(*scatter.legend_elements())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
pdf = '/usr/users/yzhu1/LoopBin/trials/saved_models/pretrain_ae_control_rep1_H3K4me1/tSNE_latant_dmpp_labeled.pdf'
plt.savefig(pdf)
plt.close()