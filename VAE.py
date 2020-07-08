import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils.data import load_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics, plot_prc
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score, accuracy_score
from sklearn.utils import class_weight

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 20

encoder_inputs = keras.Input(shape=(428,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
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
x = layers.Dense(256, activation="relu")(x)
decoder_outputs = layers.Dense(428)(x)
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
            reconstruction_loss *= 428
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
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=1-labels, s=0.5)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


args = parse_args()
x_train, x_test, y_test, x_val, y_val = load_data(args)
print(list(y_val).count(1))
print(list(y_val).count(0))

normalized_x_test = preprocessing.normalize(x_test)
normalized_x_val = preprocessing.normalize(x_val)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(normalized_x_test, epochs=100, batch_size=128)
plot_label_clusters(encoder, decoder, normalized_x_test, y_test)


predictor_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(predictor_input)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
predictor_output = layers.Dense(1, activation="relu")(x)
predictor = keras.Model(predictor_input, predictor_output, name="predictor")

# weights = class_weight.compute_class_weight('balanced',
#                                             np.unique(y_val),
#                                             y_val)

w = {0: 1.,
     1: 9.}

predictor.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

x_train_predictor = encoder.predict(normalized_x_val)
predictor.fit(x_train_predictor, y_val, epochs=100, batch_size=128, class_weight=w)

# Test model
enc_out = encoder.predict(normalized_x_val)
pred_out = predictor.predict(enc_out)

precision, recall, _ = precision_recall_curve(y_val, pred_out)
fpr, tpr, _ = roc_curve(y_val, pred_out)

plt.plot(recall, precision, 'r-', label="Precision-Recall curve of model")
plt.plot(fpr, tpr, 'b-', label="FPR-TPR curve of model")
random_predictions = np.random.random_sample((len(y_val)))
rand_au_prc, rand_recall, rand_precision, rand_au_roc, rand_fpr, rand_tpr, rand_accuracy, rand_f_measure \
        = compute_metrics(random_predictions, y_val, 0.5)
plt.plot(rand_recall, rand_precision, 'r:', label="Precision-Recall curve of random")
plt.plot(rand_fpr, rand_tpr, 'b:', label="FPR-TPR curve of random")
plt.xlabel('Recall/FPR')
plt.ylabel('Precision/TPR')
plt.axis([0, 1, 0, 1])
plt.legend()
plt.show()
