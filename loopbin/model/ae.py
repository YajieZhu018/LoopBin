########################################################################################
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import numpy as np
import math
from sklearn import mixture
from sklearn.cluster import KMeans
tf.keras.backend.set_floatx('float32')

# define class of the AE model
class AE(keras.Model):
    def __init__(self, input_size, loss_weights=None, latent_activation='relu', **kwargs):
        """
        latent_activation: activation on the 10-unit bottleneck. Defaults to 'relu'
        (the original, and what every published trial under trials/ was trained with).
        Pass None for a linear bottleneck -- under investigation because 'relu' has two
        known costs here: (1) a bottleneck unit whose pre-activation goes negative for
        every sample outputs exactly 0 with exactly 0 gradient, permanently, which is
        the "dead latent dim" failure documented in
        scripts/gmm_metric_validation/run_report.md (5-8 of 10 dims dead); and (2) it
        confines the latent to the non-negative orthant with a mass spike at 0, which
        the Gaussian mixture fitted on it (select_balanced_gmm, and VADE's GMM layer)
        is misspecified for. Note VADE's own encoder already uses a *linear* z_mean
        (vade_model.py:build_encoder), so 'relu' also makes the pretrained bottleneck
        inconsistent with the layer its weights get loaded into.
        """
        super().__init__(**kwargs)
        self.latent_activation = latent_activation
        # Phase 2: optional per-feature BCE weights (None => unmodified loss, byte-identical baseline)
        self.feature_weights = None if loss_weights is None else tf.constant(loss_weights, dtype=tf.float32)
        self.encoder = tf.keras.Sequential([
            Input(shape=(input_size,)),
            Dense(500, activation='relu'),
            Dense(500, activation='relu'),
            Dense(2000, activation='relu'),
            Dense(10, activation=latent_activation)
        ])
        self.decoder = tf.keras.Sequential([
            Input(shape=(10,)),
            Dense(2000, activation='relu'),
            Dense(500, activation='relu'),
            Dense(500, activation='relu'),
            Dense(input_size, activation='sigmoid')
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def call(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)
    
    @property
    def metrics(self):
        return [self.total_loss_tracker]
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            if self.feature_weights is None:
                loss = keras.losses.BinaryCrossentropy()(data, reconstruction)
            else:
                # weighted-mean BCE over features (w sums to 1 => same scale as the mean BCE)
                bce = K.binary_crossentropy(data, reconstruction)          # (batch, D)
                loss = tf.reduce_mean(tf.reduce_sum(bce * self.feature_weights, axis=1))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):
        z = self.encoder(data)
        reconstruction = self.decoder(z)
        if self.feature_weights is None:
            loss = keras.losses.BinaryCrossentropy()(data, reconstruction)
        else:
            bce = K.binary_crossentropy(data, reconstruction)
            loss = tf.reduce_mean(tf.reduce_sum(bce * self.feature_weights, axis=1))
        return {"loss": loss}