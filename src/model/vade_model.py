"""
An AE model to pretrain the VaDE model
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import numpy as np
import math
from sklearn import mixture
from sklearn.cluster import KMeans
import gzip
from six.moves import cPickle
import pickle


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
    
# define GMM class
class GMM(Layer):
    """
    a GMM model to learn the mean, variance and p(c|z) of the latent space
    """
    def __init__(self, n_centroid, **kwargs):
        super(GMM, self).__init__(**kwargs)
        self.n_centroid = n_centroid

    def build(self, input_shape): # create the state of the layer (weights)
        batch_size, latent_dim = input_shape[0], input_shape[1]
        # initialize theta_p with constant 1/n_centroid of dimension (n_centroid,)
        
        self.theta_p = self.add_weight(name='theta_p', shape=(self.n_centroid,), initializer=initializers.Constant(1/self.n_centroid), dtype=tf.float32, trainable=False)
        self.u_p = self.add_weight(name='u_p', shape=(latent_dim, self.n_centroid), initializer='zeros', dtype=tf.float32, trainable=False)
        self.lambda_p = self.add_weight(name='lambda_p', shape=(latent_dim, self.n_centroid), initializer='ones', dtype=tf.float32, trainable=False)
        super(GMM, self).build(input_shape)

    def call(self, inputs):  # calculate the probability of each cluster given the latent variable z (inputs)
        batch_size, latent_dim = tf.shape(inputs)[0], tf.shape(inputs)[1]
        # expand z,u,lambda,theta to shape (batch_size,latent_dim,n_centroid)
        temp_z = tf.tile(tf.expand_dims(inputs,2),[1,1,self.n_centroid])
        temp_u = tf.tile(tf.expand_dims(self.u_p,0),[batch_size,1,1])
        temp_lambda = tf.tile(tf.expand_dims(self.lambda_p,0),[batch_size,1,1])
        temp_theta = tf.tile(tf.expand_dims(tf.expand_dims(self.theta_p,0),0),[batch_size,latent_dim,1])
        # calculate p(c|z)
        temp_p_c_z = tf.exp(tf.reduce_sum((tf.math.log(temp_theta)-0.5*tf.math.log(2*math.pi*temp_lambda)-\
                           tf.square(temp_z-temp_u)/(2*temp_lambda)),axis=1))+1e-10
        # sum over latent dim and normalize it to make the sum of all centroids to 1
        temp_p_c_z = temp_p_c_z/tf.reduce_sum(temp_p_c_z,axis=-1,keepdims=True)
        return temp_p_c_z

# define class of the AE model
class VADE(keras.Model):
    def __init__(self, original_size, n_centroid, **kwargs):
        super().__init__(**kwargs)
        # add class GMM as one layer
        self.original_size = original_size
        self.n_centroid = n_centroid
        self.gmm = GMM(n_centroid)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
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

    def load_pretrained_weights(self, file_path, inputs, gmm_name):
        """
        Load the pretrained weights from the pretrained model and assign to the encoder and decoder (first two layers)
        """
        saved_model = load_model(file_path)
        # set the weights of the encoder
        self.encoder.layers[1].set_weights(saved_model.encoder.layers[0].get_weights())
        self.encoder.layers[2].set_weights(saved_model.encoder.layers[1].get_weights())
        self.encoder.layers[3].set_weights(saved_model.encoder.layers[2].get_weights())
        self.encoder.layers[4].set_weights(saved_model.encoder.layers[3].get_weights())
        # set the weights of the decoder
        self.decoder.set_weights(saved_model.decoder.get_weights())
        ## get kmeans centers
        #kmeans = KMeans(n_clusters=self.n_centroid, init='k-means++', n_init=10, random_state=42)
        #z = saved_model.encoder.predict(inputs)
        #kmeans.fit(z)
        ## save the kmeans model
        #kmeans_folder = '/usr/users/yzhu1/LoopHigh/result/saved_models/kmeans_models/'
        #with open(f'{kmeans_folder}{gmm_name}.pkl','wb') as file_pointer:
        #    pickle.dump(kmeans, file_pointer)
        ## set the center to kmeans center
        #self.gmm.u_p.assign(tf.cast(kmeans.cluster_centers_.T, tf.float32))
        ## get the centers with GMM
        g = mixture.GaussianMixture(n_components=self.n_centroid, covariance_type='diag', random_state=7)
        z = saved_model.encoder.predict(inputs)
        g.fit(z)
        # save the g in a folder
        gmm_folder = '/usr/users/yzhu1/LoopBin/trials/saved_models/gmm_models/'
        with open(f'{gmm_folder}{gmm_name}.pkl','wb') as file_pointer:
            pickle.dump(g, file_pointer)
        # set the weights of the GMM
        self.gmm.u_p.assign(tf.cast(g.means_.T, tf.float32))
        self.gmm.lambda_p.assign(tf.cast(g.covariances_.T, tf.float32))
        print('Pretrained weights loaded')
        
        
    def calculate_kl_loss(self, z, z_mean, z_log_var):  # calculate the KL loss of the latent space
        batch_size, latent_dim = tf.shape(z)[0], tf.shape(z)[1]
        # expand z to shape (batch_size,latent_dim,n_centroid)
        Z = tf.tile(tf.expand_dims(z_mean,2),[1,1,self.gmm.n_centroid])
        z_mean_t = tf.tile(tf.expand_dims(z_mean,2),[1,1,self.gmm.n_centroid])
        z_log_var_t = tf.tile(tf.expand_dims(z_log_var,2),[1,1,self.gmm.n_centroid])
        u_tensor3 = tf.tile(tf.expand_dims(self.gmm.u_p,0),[batch_size,1,1])
        lambda_tensor3 = tf.tile(tf.expand_dims(self.gmm.lambda_p,0),[batch_size,1,1])
        theta_tensor3 = tf.tile(tf.expand_dims(tf.expand_dims(self.gmm.theta_p,0),0),[batch_size,latent_dim,1])
        # calculate p(c|z)
        p_c_z = tf.exp(tf.reduce_sum((tf.math.log(theta_tensor3)-0.5*tf.math.log(2*math.pi*lambda_tensor3)-\
                           tf.square(Z-u_tensor3)/(2*lambda_tensor3)),axis=1))+1e-10
        # sum over latent dim and normalize it to make the sum of all centroids to 1
        gamma = p_c_z/tf.reduce_sum(p_c_z,axis=-1,keepdims=True)
        gamma_t= tf.tile(tf.expand_dims(gamma,1),[1,latent_dim,1])
        # calculate the loss
        loss = 0.5*tf.reduce_sum(gamma_t*(tf.cast(latent_dim, tf.float32)*tf.math.log(math.pi*2)+tf.math.log(lambda_tensor3)+tf.exp(z_log_var_t)/lambda_tensor3+tf.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        -0.5*tf.reduce_sum(z_log_var+1,axis=-1)\
        -tf.reduce_sum(tf.math.log(tf.tile(tf.expand_dims(self.gmm.theta_p,0),[batch_size,1]))*gamma,axis=-1)\
        +tf.reduce_sum(tf.math.log(gamma)*gamma,axis=-1)
        return loss
    

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.gmm(z), self.decoder(z)
    
    @property
    def metrics(self):
        return [self.total_loss_tracker]

    @tf.function
    def train_step(self, data):
        loss_fn = keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = loss_fn(data, reconstruction)
            # calculate vae loss according to the vae loss function
            kl_loss = self.calculate_kl_loss(z, z_mean, z_log_var)
            loss = reconstruction_loss*self.original_size + kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}