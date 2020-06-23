from utils.custom_losses import *

import keras.backend as K
from keras import losses
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, Bidirectional, LSTM
from keras.optimizers import Adam

gamma = K.variable([1])


def set_trainability(model, trainable=False):
    # Alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def d_loss(y_true, y_predicted):
    loss_gen = losses.binary_crossentropy(y_true, y_predicted)
    loss = gamma * loss_gen
    return loss


def load_model(args):
    # Building Generator
    G = Sequential()
    G.add(Dense(args.latent_dim, input_dim=args.latent_dim))
    G.add(LeakyReLU(alpha=0.2))
    G.add(BatchNormalization(momentum=0.8))
    G.add(Reshape((args.latent_dim, 1)))
    G.add(LSTM(428))

    # Building Discriminator
    D = Sequential()
    D.add(Reshape((428, 1), input_shape=(428,)))
    D.add(LSTM(4))
    D.add(Dense(4, activation='relu'))
    D.add(Dense(1, activation='sigmoid'))
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
