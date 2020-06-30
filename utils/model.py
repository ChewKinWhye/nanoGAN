from utils.custom_losses import *

import keras.backend as K
from keras import losses
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, LSTM,\
    Layer, Lambda, Permute, Flatten, add, Bidirectional, Concatenate, Conv1D, Conv2D,\
    MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.layers import LeakyReLU

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


def load_deep_signal_model(args):
    # Building Generator
    '''
    g_in = Input(shape=(args.latent_dim,))
    x = Dense(300, activation='relu')(g_in)
    ffnn_out = Dense(400, activation='relu')(x)
    # Top module model
    top_module = Lambda(lambda x: x[:, 0:100])(ffnn_out)
    x = Reshape((100, 1))(top_module)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    top_out = LSTM(68)(x)
    # Bottom module model
    bottom_module = Lambda(lambda x:x[:, 100:])(ffnn_out)
    x = Reshape((1, 300, 1))(bottom_module)
    x = Conv2D(filters=32, kernel_size=(1, 7), strides=1)(x)
    x = AveragePooling2D(pool_size=(1, 3), strides=2)(x)
    x = Reshape((-1, 1))(x)
    x = Bidirectional(LSTM(50))(x)
    bottom_out = Dense(360)(x)
    g_out = Concatenate(axis=1)([top_out, bottom_out])
    G = Model(g_in, g_out)
    '''
    g_in = Input(shape=(args.latent_dim,))
    x = Dense(300, activation='relu')(g_in)
    ffnn_out = Dense(400, activation='relu')(x)

    top_module = Lambda(lambda x: x[:, 0:100])(ffnn_out)
    x = Reshape((100, 1))(top_module)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    top_out = LSTM(68)(x)

    bottom_module = Lambda(lambda x: x[:, 100:])(ffnn_out)
    x = Reshape((1, 300, 1))(bottom_module)
    x = Conv2D(filters=32, kernel_size=(1, 7), strides=1)(x)
    x = AveragePooling2D(pool_size=(1, 7), strides=5)(x)
    x = AveragePooling2D(pool_size=(1, 5), strides=3)(x)
    x = Reshape((-1,))(x)
    bottom_out = Dense(360)(x)
    g_out = Concatenate(axis=1)([top_out, bottom_out])
    G = Model(g_in, g_out)
    G.summary()
    # Building Discriminator
    d_in = Input(shape=(428,))
    # Top module to process 4*17 features using LSTM
    top_module = Lambda(lambda x: x[:, 0:-360])(d_in)
    x = Reshape((68, 1))(top_module)
    x = Bidirectional(LSTM(50))(x)
    x = Reshape((100, 1))(x)
    top_out = Bidirectional(LSTM(50))(x)
    # Bottom model to process 360 signals using CNN
    bottom_module = Lambda(lambda x: x[:, -360:])(d_in)
    x = Reshape((1, 360, 1))(bottom_module)
    x = Conv2D(filters=32, kernel_size=(1, 7), strides=2)(x)
    # Add in inception layers
    x = AveragePooling2D(pool_size=(1, 7), strides=5)(x)
    x = AveragePooling2D(pool_size=(1, 5), strides=3)(x)
    bottom_out = Reshape((-1,))(x)
    # Classification module which combines top and bottom outputs using FFNN
    classification_in = Concatenate(axis=1)([top_out, bottom_out])
    x = Dense(32, activation='relu')(classification_in)
    d_out = Dense(1, activation='sigmoid')(x)
    d_opt = Adam(lr=args.d_lr, beta_1=0.5, beta_2=0.999)
    D = Model(d_in, d_out)
    D.compile(loss=d_loss, optimizer=d_opt)
    D.summary()
    # Building GAN
    set_trainability(D, False)
    gan_in = Input(shape=(args.latent_dim,))
    g_out = G(gan_in)
    gan_out = D(g_out)
    gan = Model(gan_in, gan_out)

    g_opt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    gan.compile(loss=fence_loss(g_out, args.beta, 2), optimizer=g_opt)
    gan.summary()
    return G, D, gan


def load_model(args):
    # Building Generator
    G = Sequential()
    G.add(Dense(args.latent_dim, input_dim=args.latent_dim))
    G.add(LeakyReLU(alpha=0.2))
    G.add(BatchNormalization(momentum=0.8))
    G.add(Reshape((args.latent_dim, 1)))
    G.add(LSTM(4))
    G.add(Dense(428))
    G.summary()
    # Building Discriminator
    D = Sequential()
    D.add(Reshape((428,1), input_shape=(428,)))
    D.add(LSTM(20))
    D.add(Dense(50))
    D.add(LeakyReLU(alpha=0.2))
    D.add(BatchNormalization(momentum=0.8))
    D.add(Dense(4, activation='relu'))
    D.add(Dense(1, activation='sigmoid'))
    d_opt = Adam(lr=args.d_lr, beta_1=0.5, beta_2=0.999)
    D.compile(loss=d_loss, optimizer=d_opt)
    D.summary()
    # Building GAN
    set_trainability(D, False)
    gan_in = Input(shape=(args.latent_dim,))
    g_out = G(gan_in)
    gan_out = D(g_out)
    gan = Model(gan_in, gan_out)

    g_opt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    gan.compile(loss=fence_loss(g_out, args.beta, 2), optimizer=g_opt)
    gan.summary()

    return G, D, gan
