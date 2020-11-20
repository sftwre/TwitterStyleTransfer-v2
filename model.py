import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2


class Encoder(keras.Model):
    """
    Unidirectional RNN encoder
    """
    def __init__(self, embedder, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.model = keras.Sequential()
        self.gru = layers.GRU(700, dropout=0.5)
        self.model.add(embedder)
        self.model.add(self.gru)

    def call(self, inputs):
        x = self.model(inputs)
        return x

class Decoder(keras.Model):
    """
    LSTM based generator that re-creates text based on
    latent space and controllable parameters.
    """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.lstm = layers.LSTM(700, dropout=0.5)

    def call(self, inputs):
        x = self.lstm(inputs)
        return x

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }