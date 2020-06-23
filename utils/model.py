from utils.custom_losses import *

import keras.backend as K
from keras import losses
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, Bidirectional, LSTM, Layer
from keras.optimizers import Adam

gamma = K.variable([1])


class Linear(Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

def set_trainability(model, trainable=False):
    # Alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def d_loss(y_true, y_predicted):
    loss_gen = losses.binary_crossentropy(y_true, y_predicted)
    loss = gamma * loss_gen
    return loss


def load_deep_signal_model(args):
    # Building Generator
    G = Sequential()
    G.add(Dense(args.latent_dim, input_dim=args.latent_dim))
    G.add(LeakyReLU(alpha=0.2))
    G.add(BatchNormalization(momentum=0.8))
    G.add(Reshape((args.latent_dim, 1)))
    G.add(LSTM(428))

    # Building Discriminator
    D_in = Input(shape=(428,))
    x = Linear(units=428, input_dim=428)(D_in)
    x = Reshape((428, 1))(x)
    x = LSTM(4)(x)
    x = Dense(4, activation='relu')(x)
    D_out = Dense(1, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    d_opt = Adam(lr=args.d_lr, beta_1=0.5, beta_2=0.999)
    D.compile(loss=d_loss, optimizer=d_opt)

    # Building GAN
    set_trainability(D, False)
    gan_in = Input(shape=(args.latent_dim,))
    g_out = G(gan_in)
    gan_out = D(g_out)
    gan = Model(gan_in, gan_out)

    g_opt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    gan.compile(loss=fence_loss(g_out, args.beta, 2), optimizer=g_opt)
    return G, D, gan
