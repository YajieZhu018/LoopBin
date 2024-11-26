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

### Fit GMM
#from sklearn.mixture import GaussianMixture
#for i in range(2,10):
#    gmm = GaussianMixture(n_components=i)  # Try with different numbers of components
#    gmm.fit(z)
#    print(f'{i} components')
#    print("Log-likelihood:", gmm.score(z))
#    print("AIC:", gmm.aic(z))
#    print("BIC:", gmm.bic(z))

### Fit KDE
#from sklearn.neighbors import KernelDensity
#kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(z)
#log_density = kde.score_samples(z)
#plt.hist(np.exp(log_density), bins=50)
#plt.title('Kernel Density Estimation of Latent Space')
#pdf = '/usr/users/yzhu1/LoopBin/trials/saved_models/pretrain_ae_control_rep1_H3K4me1/kde.pdf'
#plt.savefig(pdf)
#plt.close()

### Henze-Zirkler's test
#import pingouin as pg
#hz_test = pg.multivariate_normality(z)
#print(f'pvalue of hz_test: {hz_test}')