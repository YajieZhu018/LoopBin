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
    def __init__(self, input_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            Input(shape=(input_size,)),
            Dense(500, activation='relu'),
            Dense(500, activation='relu'),
            Dense(2000, activation='relu'),
            Dense(10, activation='relu')
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
        loss_fn = keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            loss = loss_fn(data, reconstruction)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):
        loss_fn = keras.losses.BinaryCrossentropy()
        z = self.encoder(data)
        reconstruction = self.decoder(z)
        loss = loss_fn(data, reconstruction)
        return {"loss": loss}