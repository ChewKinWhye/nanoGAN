import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from utils.data import load_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, compute_metrics_standardized_confident
import random

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 10

encoder_inputs = keras.Input(shape=(479,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(479, activation="sigmoid")(x)
decoder_outputs = layers.Reshape((479,))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        # if isinstance(data, tuple):
        #     data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 479
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


def plot_label_clusters(encoder, decoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    filter_indices = random.sample(range(0, data.shape[0]-1), 3000)
    data_sample = np.take(data, filter_indices, axis=0)
    label_sample = np.take(labels, filter_indices, axis=0)
    z_mean, _, _ = encoder.predict(data_sample)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label_sample, s=0.5)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


args = parse_args()
_, x_train, y_train, x_test, y_test = load_data(args)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128)
plot_label_clusters(encoder, decoder, x_train, y_train)


predictor_input = keras.Input(shape=(latent_dim*2,))
x = layers.Dense(8, activation="relu")(predictor_input)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dense(8, activation="relu")(x)
predictor_output = layers.Dense(1, activation="sigmoid")(x)
predictor = keras.Model(predictor_input, predictor_output, name="predictor")

predictor.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

x_train_mean, x_train_sd, _ = encoder.predict(x_train)
x_train = np.concatenate((x_train_mean, x_train_sd), axis=1)
predictor.fit(x_train, y_train, epochs=30, batch_size=128)

# Test model
x_test_mean, x_test_sd, _ = encoder.predict(x_test)
x_test = np.concatenate((x_test_mean, x_test_sd), axis=1)

pred_out = predictor.predict(x_test)
accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(
    pred_out, y_test)

print(f"\tAccuracy    : {accuracy_val:.3f}")
print(f"\tSensitivity : {sensitivity_val:.3f}")
print(f"\tSpecificity : {specificity_val:.3f}")
print(f"\tPrecision   : {precision_val:.3f}")
print(f"\tAUC         : {au_roc_val:.3f}")
print(f"{cm_val}")

accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized_confident(
    pred_out, y_test)

print(f"\tAccuracy    : {accuracy_val:.3f}")
print(f"\tSensitivity : {sensitivity_val:.3f}")
print(f"\tSpecificity : {specificity_val:.3f}")
print(f"\tPrecision   : {precision_val:.3f}")
print(f"\tAUC         : {au_roc_val:.3f}")
print(f"{cm_val}")
