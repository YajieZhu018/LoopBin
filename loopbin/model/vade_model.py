"""
An AE model to pretrain the VaDE model
"""
#from statistics import covariance
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import numpy as np
from tensorflow.keras.layers import GaussianNoise
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import gzip
from six.moves import cPickle
import pickle
import random
import os


# funtion to convert X to the default type float32
def floatX(X):
    return np.asarray(X, dtype=tf.keras.backend.floatx())

# function to load mnist data
def load_data(path, dataset):
    """
    args: path and name of the dataset
    yields: X, Y (inputs and labels of the dataset)

    """

    if dataset == 'mnist':
        file = f'{path}/{dataset}/mnist.pkl.gz'
        with gzip.open(file, 'rb') as f:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding='bytes')
        # normalize to 0-1
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))
    return X,Y

def _gmm_component_variances(g, covariance_type, n_features):
    """
    Per-component, per-dimension variance array (n_components, n_features),
    regardless of sklearn's covariance_type, so cluster collapse can be checked
    uniformly across candidates.
    """
    if covariance_type == 'diag':
        return g.covariances_
    if covariance_type == 'spherical':
        return np.tile(g.covariances_[:, np.newaxis], n_features)
    if covariance_type == 'full':
        return np.stack([np.diag(cov) for cov in g.covariances_])
    if covariance_type == 'tied':
        return np.tile(np.diag(g.covariances_)[np.newaxis, :], (g.n_components, 1))
    raise ValueError(f"Unsupported covariance_type: {covariance_type}")


# Per-dim latent variance at/below this counts a dimension as "dead" (same cutoff as
# scripts/gmm_metric_validation/latent_diagnostics.py).
DEAD_VAR_THRESHOLD = 1e-8


def _fit_gmm_candidates(z, n_centroid, covariance_type='diag', reg_covar=1e-4,
                         random_states=(0, 7, 13, 21, 42, 73, 101, 123, 777, 2024),
                         n_init=1):
    """
    Fit one GaussianMixture per random_state and return the list of candidate dicts
    (random_state, gmm, min_weight, separation, cov_health, assignment_confidence,
    log_likelihood), printing each candidate's diagnostics. Pure fitting + metric
    computation -- no filtering, scoring, or selection (that's select_balanced_gmm's
    job), so this can be reused as-is by anything that wants the same candidate pool,
    e.g. sweeping different metric_weights without refitting.

    n_init is deliberately 1, not raised. Tested on the real production latent
    (n_init in {1, 5, 10}, see run_report.md's 2026-09-03 "n_init in GMM candidate
    fitting" entry): raising it barely moves log-likelihood (the 10 random_states
    already sit within std 0.005 of each other at n_init=1) but collapses the outer
    random_states onto a single shared local optimum -- and that shared optimum is
    *worse* on balance/separation than what a lucky single-init seed finds (best
    min_weight in the pool dropped from 0.0322 to 0.0249, best separation from 1.3651
    to 1.3339). n_init>1 reruns EM from multiple k-means starts and keeps only the
    highest-log-likelihood one, i.e. pure log-likelihood pressure before balance ever
    gets a look -- exactly what select_balanced_gmm exists to counteract. If within-seed
    k-means noise is a concern, add more/different entries to random_states instead;
    that preserves cross-candidate diversity rather than collapsing it.
    """
    n_features = z.shape[1]
    log_k = math.log(n_centroid)

    # cov_health is a ratio of component variances, so dead latent dims (variance ~0 for
    # every component, e.g. under a ReLU bottleneck) drive its numerator to ~0 for every
    # candidate and make the collapse_ratio_threshold reject the whole pool regardless of
    # component quality. Measure it on the alive dims only.
    alive = np.asarray(z).var(axis=0) > DEAD_VAR_THRESHOLD
    if not alive.any():
        alive = np.ones(n_features, dtype=bool)

    candidates = []
    for rs in random_states:
        g = mixture.GaussianMixture(
            n_components=n_centroid, covariance_type=covariance_type,
            init_params='kmeans', n_init=n_init, reg_covar=reg_covar, random_state=rs,
        )
        g.fit(z)

        min_weight = float(g.weights_.min())

        variances = _gmm_component_variances(g, covariance_type, n_features)
        pooled_std = float(np.sqrt(variances.mean()))
        separation = float(pdist(g.means_).min() / (pooled_std + 1e-10))
        variances_alive = variances[:, alive]
        cov_health = float(variances_alive.min() / (np.median(variances_alive) + 1e-10))

        gamma = g.predict_proba(z)
        entropy = -np.sum(gamma * np.log(gamma + 1e-10), axis=1)
        assignment_confidence = float(1.0 - entropy.mean() / log_k)

        log_likelihood = float(g.score(z))

        candidates.append({
            'random_state': rs, 'gmm': g, 'min_weight': min_weight,
            'separation': separation, 'cov_health': cov_health,
            'assignment_confidence': assignment_confidence, 'log_likelihood': log_likelihood,
        })
        print(f"[GMM candidate rs={rs}] min_weight={min_weight:.4f} "
              f"separation={separation:.4f} cov_health={cov_health:.4f} "
              f"assignment_confidence={assignment_confidence:.4f} "
              f"log_likelihood={log_likelihood:.4f}")

    return candidates


def select_balanced_gmm(z, n_centroid, covariance_type='diag', reg_covar=1e-4,
                         random_states=(0, 7, 13, 21, 42, 73, 101, 123, 777, 2024),
                         n_init=1, min_weight_threshold=0.02, collapse_ratio_threshold=0.05,
                         metric_weights=(1.0, 1.0, 1.0)):
    """
    Fit several GMM candidates (fixed random_state list, deterministic) and pick the
    one that best balances cluster spread/weight against fit quality, instead of
    sklearn's single best-by-log-likelihood pick (which tends to favor one tight
    component plus several near-empty ones).

    Each candidate is scored on five metrics (computed by _fit_gmm_candidates):
      - balance: smallest component weight, must exceed min_weight_threshold
        (default 2% of the data). Looser than a "fair share" cutoff so genuinely
        small, scattered clusters aren't discarded, while still rejecting
        essentially-empty components.
      - separation: minimum pairwise distance between cluster centers, normalized
        by the pooled per-dimension std, so it's comparable across candidates
        regardless of the latent space's arbitrary scale.
      - covariance health: minimum component variance relative to the median
        variance across all components/dims. A low ratio flags a component that
        has collapsed onto a handful of points (a degenerate, overfit spike).
      - assignment quality: average soft-assignment confidence
        (1 - normalized entropy of predict_proba), rewarding crisp cluster
        boundaries over ambiguous, overlapping ones.
      - fit quality: average log-likelihood per sample.

    Candidates failing the balance or covariance-health floors are dropped; the
    survivors are ranked by a weighted sum of z-scored separation / assignment
    quality / fit quality (weights via metric_weights). If nothing survives the
    floors, all candidates are scored the same way and the least degenerate one
    (by balance, then covariance health) is picked instead.
    """
    candidates = _fit_gmm_candidates(z, n_centroid, covariance_type=covariance_type,
                                      reg_covar=reg_covar, random_states=random_states,
                                      n_init=n_init)

    survivors = [c for c in candidates if c['min_weight'] >= min_weight_threshold
                 and c['cov_health'] >= collapse_ratio_threshold]
    pool = survivors if survivors else candidates
    if not survivors:
        print(f"WARNING: no GMM candidate met min_weight_threshold={min_weight_threshold:.4f} "
              f"and collapse_ratio_threshold={collapse_ratio_threshold:.4f}; "
              f"falling back to the least degenerate candidate.")

    def _zscore(values):
        values = np.asarray(values, dtype=float)
        std = values.std()
        return np.zeros_like(values) if std < 1e-10 else (values - values.mean()) / std

    w_sep, w_conf, w_ll = metric_weights
    for c, s, cf, ll in zip(pool, _zscore([c['separation'] for c in pool]),
                             _zscore([c['assignment_confidence'] for c in pool]),
                             _zscore([c['log_likelihood'] for c in pool])):
        c['composite_score'] = w_sep * s + w_conf * cf + w_ll * ll

    if survivors:
        best = max(pool, key=lambda c: c['composite_score'])
    else:
        best = max(pool, key=lambda c: (c['min_weight'], c['cov_health'], c['composite_score']))

    print(f"Selected GMM: random_state={best['random_state']}, "
          f"min_weight={best['min_weight']:.4f}, separation={best['separation']:.4f}, "
          f"cov_health={best['cov_health']:.4f}, "
          f"assignment_confidence={best['assignment_confidence']:.4f}, "
          f"log_likelihood={best['log_likelihood']:.4f}, "
          f"composite_score={best['composite_score']:.4f}")
    return best['gmm']

# define sampling function
class Sampling(Layer):
    """
    args: mean and log of variance of Q(z|X)
    reparameterization trick is used to sample z
    yields: sampled latent vector z, shape (batch_size, latent_dim)
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GMM(Layer):
    """
    a GMM model to learn the mean, variance and p(c|z) of the latent space
    """
    def __init__(self, n_centroid, trainable_theta=False, trainable_prior=False, **kwargs):
        super(GMM, self).__init__(**kwargs)
        self.n_centroid = n_centroid
        self.trainable_theta = trainable_theta
        # Diagnostic-only knob (not wired to any CLI flag): lets u_p/lambda_p move under
        # gradient descent instead of staying frozen at their load_pretrained_weights
        # init. See run_report.md's 2026-09-07 entry (trainable u_p/lambda_p) for why
        # this is opt-in, not default: it reproducibly collapses to one cluster.
        self.trainable_prior = trainable_prior

    def build(self, input_shape):
        batch_size, latent_dim = input_shape[0], input_shape[1]
        # theta_p (the prior mixture weights, pi) is frozen by default. When trainable it
        # needs the NonNeg constraint: it is only ever used through tf.math.log(), so an
        # unconstrained gradient step that takes it to <= 0 produces NaN losses.
        self.theta_p = self.add_weight(name='theta_p', shape=(self.n_centroid,),
                                        initializer=initializers.Constant(1 / self.n_centroid),
                                        dtype=tf.float32, trainable=self.trainable_theta,
                                        constraint=(keras.constraints.NonNeg()
                                                    if self.trainable_theta else None))
        self.u_p = self.add_weight(name='u_p', shape=(latent_dim, self.n_centroid),
                                    initializer='zeros', dtype=tf.float32, trainable=self.trainable_prior)
        self.lambda_p = self.add_weight(name='lambda_p', shape=(latent_dim, self.n_centroid),
                                         initializer='ones', dtype=tf.float32, trainable=self.trainable_prior)
        super(GMM, self).build(input_shape)

    def theta(self):
        """theta_p renormalised to a valid distribution and floored away from zero.
        A no-op when theta_p is frozen (it stays at its uniform 1/k initialiser); the
        clip/renormalise only matters once gradients can move it."""
        t = tf.clip_by_value(self.theta_p, 1e-8, 1.0)
        return t / tf.reduce_sum(t)

    def assign_theta(self, pi, floor=0.0):
        """Set theta_p directly (no gradient), floored at `floor` and renormalised.

        This is how the EM arm updates the prior: the closed-form M-step optimum is
        pi_c = mean_n gamma_nc, which main.py's EMPriorUpdate computes and damps before
        handing it here. The floor is what keeps a shrinking component recoverable --
        unlike the trainable-theta path, where NonNeg() clips to exactly 0 and
        clip_by_value's zero gradient outside its bounds makes that absorbing.
        """
        pi = np.asarray(pi, dtype=np.float32)
        pi = np.maximum(pi, floor)
        pi = pi / pi.sum()
        self.theta_p.assign(pi)
        return pi

    def call(self, inputs):
        batch_size, latent_dim = tf.shape(inputs)[0], tf.shape(inputs)[1]
        temp_z = tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.n_centroid])
        temp_u = tf.tile(tf.expand_dims(self.u_p, 0), [batch_size, 1, 1])
        temp_lambda = tf.tile(tf.expand_dims(self.lambda_p, 0), [batch_size, 1, 1])
        temp_theta = tf.tile(tf.expand_dims(tf.expand_dims(self.theta(), 0), 0), [batch_size, latent_dim, 1])
        temp_p_c_z = tf.exp(tf.reduce_sum(
            tf.math.log(temp_theta) - 0.5 * tf.math.log(2 * math.pi * temp_lambda)
            - tf.square(temp_z - temp_u) / (2 * temp_lambda), axis=1)) + 1e-10
        temp_p_c_z = temp_p_c_z / tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)
        return temp_p_c_z


class VADE(keras.Model):
    def __init__(self, original_size, n_centroid, trainable_theta=False, trainable_prior=False,
                loss_weights=None, marginal_entropy_beta=0.0, marginal_kl_beta=0.0,
                marginal_kl_target=None, **kwargs):
        super().__init__(**kwargs)
        self.original_size = original_size
        self.n_centroid = n_centroid
        # Phase 4: weight of the marginal-entropy regularizer (0.0 => OFF => byte-identical baseline)
        self.marginal_entropy_beta = float(marginal_entropy_beta)
        # Phase 4b: KL-to-target on the sorted marginal (0.0 => OFF). target = desired UNEQUAL size shape.
        self.marginal_kl_beta = float(marginal_kl_beta)
        self.marginal_kl_target = (tf.constant(marginal_kl_target, dtype=tf.float32) if marginal_kl_target is not None else None)
        # Phase 2: optional per-feature BCE weights (None => unmodified loss, byte-identical baseline)
        self.feature_weights = None if loss_weights is None else tf.constant(loss_weights, dtype=tf.float32)
        self.gmm = GMM(n_centroid, trainable_theta=trainable_theta, trainable_prior=trainable_prior)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def build_encoder(self):
        input = Input(shape=(self.original_size,))
        x = Dense(500, activation='relu')(input)
        x = Dense(500, activation='relu')(x)
        x = Dense(2000, activation='relu')(x)
        z_mean = Dense(10)(x)
        z_log_var = Dense(10)(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(input, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        input = Input(shape=(10,))
        x = Dense(2000, activation='relu')(input)
        x = Dense(500, activation='relu')(x)
        x = Dense(500, activation='relu')(x)
        output = Dense(self.original_size, activation='sigmoid')(x)
        return Model(input, output, name='decoder')

    def load_pretrained_weights(self, file_path, inputs, gmm_name, output_path,
                                metric_weights=(1.0, 1.0, 1.0), init_theta_from_gmm=False,
                                prior_gmm_path=None):
        """
        Load the pretrained weights from the pretrained model and assign to the encoder and decoder (first two layers)

        metric_weights is forwarded to select_balanced_gmm as (separation,
        assignment_confidence, log_likelihood); the default keeps the equal-weight
        composite every earlier run used.

        init_theta_from_gmm additionally seeds the prior weights theta_p from the fitted
        GMM. This used to never happen -- u_p and lambda_p were taken from the GMM but
        theta_p was left at its uniform 1/k initialiser -- so it stays opt-in to keep the
        default behaviour of earlier runs unchanged.

        prior_gmm_path loads an already-fitted GaussianMixture pickle instead of running
        encoder.predict + select_balanced_gmm, and metric_weights is then unused.
        Re-deriving the prior is not reproducible across machines: the same code, seeds
        and AE weights selected random_state=73 on 2026-08-29 (the CUDA shim failed to
        unpack, so it ran on CPU) and random_state=13 on 2026-08-31 (A100), because the
        two encoder forward passes differ by more than fp32 rounding and the ten
        candidate separation scores all sit within 1.13-1.37 of each other, so the argmax
        moves. Pinning the pickle makes the prior an input to the experiment rather than
        something every job re-rolls. See data/03_prior/relu_rs73/provenance.json.
        """
        saved_model = load_model(file_path)
        self.encoder.layers[1].set_weights(saved_model.encoder.layers[0].get_weights())
        self.encoder.layers[2].set_weights(saved_model.encoder.layers[1].get_weights())
        self.encoder.layers[3].set_weights(saved_model.encoder.layers[2].get_weights())
        self.encoder.layers[4].set_weights(saved_model.encoder.layers[3].get_weights())
        self.decoder.set_weights(saved_model.decoder.get_weights())

        if prior_gmm_path:
            with open(prior_gmm_path, 'rb') as file_pointer:
                g = pickle.load(file_pointer)
            covariance_type = g.covariance_type
            print(f'GMM prior loaded from {prior_gmm_path} '
                  f'(random_state={g.random_state}, covariance_type={covariance_type})')
        else:
            covariance_type = 'diag'
            z = saved_model.encoder.predict(inputs)
            g = select_balanced_gmm(z, self.n_centroid, covariance_type=covariance_type,
                                    metric_weights=metric_weights)

        latent_dim = g.means_.shape[1]
        if g.n_components != self.n_centroid or latent_dim != self.gmm.u_p.shape[0]:
            raise ValueError(
                f'GMM prior shape mismatch: prior is {g.n_components} components x '
                f'{latent_dim} latent dims, model expects {self.n_centroid} x '
                f'{int(self.gmm.u_p.shape[0])}')

        gmm_folder = os.path.join(output_path, 'gmm_models')
        os.makedirs(gmm_folder, exist_ok=True)
        with open(os.path.join(gmm_folder, f'{gmm_name}.pkl'), 'wb') as file_pointer:
            pickle.dump(g, file_pointer)

        if covariance_type == 'spherical':
            covariances = np.tile(g.covariances_[:, np.newaxis], latent_dim)
        else:
            covariances = g.covariances_

        self.gmm.u_p.assign(tf.cast(g.means_.T, tf.float32))
        self.gmm.lambda_p.assign(tf.cast(covariances.T, tf.float32))
        if init_theta_from_gmm:
            self.gmm.theta_p.assign(tf.cast(g.weights_, tf.float32))
            print(f'theta_p initialised from the pretrained GMM: '
                  f'{np.array2string(g.weights_, precision=4)}')
        print('Pretrained weights loaded')

    def calculate_entropy(self, gamma):
        avg_cluster_prob = tf.reduce_mean(gamma, axis=0)
        entropy = -tf.reduce_sum(avg_cluster_prob * tf.math.log(avg_cluster_prob + 1e-10))
        return entropy

    def calculate_kl_loss(self, z, z_mean, z_log_var):
        batch_size, latent_dim = tf.shape(z)[0], tf.shape(z)[1]
        Z = tf.tile(tf.expand_dims(z_mean, 2), [1, 1, self.gmm.n_centroid])
        z_mean_t = tf.tile(tf.expand_dims(z_mean, 2), [1, 1, self.gmm.n_centroid])
        z_log_var_t = tf.tile(tf.expand_dims(z_log_var, 2), [1, 1, self.gmm.n_centroid])
        u_tensor3 = tf.tile(tf.expand_dims(self.gmm.u_p, 0), [batch_size, 1, 1])
        lambda_tensor3 = tf.tile(tf.expand_dims(self.gmm.lambda_p, 0), [batch_size, 1, 1])
        theta_tensor3 = tf.tile(tf.expand_dims(tf.expand_dims(self.gmm.theta(), 0), 0), [batch_size, latent_dim, 1])
        p_c_z = tf.exp(tf.reduce_sum(
            tf.math.log(theta_tensor3) - 0.5 * tf.math.log(2 * math.pi * lambda_tensor3)
            - tf.square(Z - u_tensor3) / (2 * lambda_tensor3), axis=1)) + 1e-10
        gamma = p_c_z / tf.reduce_sum(p_c_z, axis=-1, keepdims=True)
        gamma_t = tf.tile(tf.expand_dims(gamma, 1), [1, latent_dim, 1])
        kl_loss = 0.5 * tf.reduce_sum(
            gamma_t * (tf.cast(latent_dim, tf.float32) * tf.math.log(math.pi * 2)
                       + tf.math.log(lambda_tensor3)
                       + tf.exp(z_log_var_t) / lambda_tensor3
                       + tf.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
            - 0.5 * tf.reduce_sum(z_log_var + 1, axis=-1) \
            - tf.reduce_sum(tf.math.log(tf.tile(tf.expand_dims(self.gmm.theta(), 0), [batch_size, 1])) * gamma, axis=-1) \
            + tf.reduce_sum(tf.math.log(gamma) * gamma, axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)
        # Phase 4: optional marginal-entropy regularizer (RIM Krause NeurIPS'10 / IMSAT Hu ICML'17 /
        # IIC Ji ICCV'19). beta==0 (default) => term skipped => byte-identical baseline. beta>0
        # maximizes H(p_bar) = entropy of the batch-average GMM responsibility, anchoring the marginal
        # cluster distribution so the per-cluster mixing proportions stop wandering across seeds.
        if self.marginal_entropy_beta > 0.0:
            norm_entropy = self.calculate_entropy(gamma) / tf.math.log(tf.cast(self.n_centroid, tf.float32))
            kl_loss = kl_loss - self.marginal_entropy_beta * norm_entropy
        # Phase 4b: KL(sorted(p_bar) || target) — anchor the marginal SHAPE to a fixed UNEQUAL target
        # so proportions become reproducible across seeds WITHOUT being forced uniform. OFF by default.
        if self.marginal_kl_beta > 0.0 and self.marginal_kl_target is not None:
            p_bar = tf.reduce_mean(gamma, axis=0)
            sorted_p = tf.sort(p_bar, direction='DESCENDING')
            tgt = self.marginal_kl_target
            kl_to_target = tf.reduce_sum(sorted_p * (tf.math.log(sorted_p + 1e-10) - tf.math.log(tgt + 1e-10)))
            kl_loss = kl_loss + self.marginal_kl_beta * kl_to_target
        return kl_loss

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.gmm(z), self.decoder(z)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            if self.feature_weights is None:
                reconstruction_loss = keras.losses.BinaryCrossentropy()(data, reconstruction) * self.original_size
            else:
                # weighted recon; (w * original_size) sums to original_size => uniform == the line above
                bce = K.binary_crossentropy(data, reconstruction)          # (batch, D)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce * (self.feature_weights * self.original_size), axis=1))
            kl_loss = self.calculate_kl_loss(z, z_mean, z_log_var)
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        if self.feature_weights is None:
            reconstruction_loss = keras.losses.BinaryCrossentropy()(data, reconstruction) * self.original_size
        else:
            bce = K.binary_crossentropy(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce * (self.feature_weights * self.original_size), axis=1))
        kl_loss = self.calculate_kl_loss(z, z_mean, z_log_var)
        loss = reconstruction_loss + kl_loss
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
        }
