from utils.custom_losses import *

import keras.backend as K
from keras import losses
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, LSTM,\
    Layer, Lambda, Permute, Flatten, add, Bidirectional, Concatenate, Conv1D, Conv2D,\
    MaxPooling1D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Dropout
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


def load_dc_gan_model(args):
    generator = Sequential()
    generator.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(args.latent_dim,)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Reshape((7, 7, 256)))
    assert generator.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    generator.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert generator.output_shape == (None, 7, 7, 128)
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert generator.output_shape == (None, 14, 14, 64)
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert generator.output_shape == (None, 28, 28, 1)
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    generator.add(Flatten())
    generator.add(Dense(479, activation='sigmoid'))
    generator.summary()

    discriminator = Sequential()
    discriminator.add(Dense(28 * 28, use_bias=False, input_shape=(479,)))
    discriminator.add(Reshape((28, 28, 1)))
    discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    d_opt = Adam(lr=args.d_lr, beta_1=0.5, beta_2=0.999)
    discriminator.compile(loss=d_loss, optimizer=d_opt)

    set_trainability(discriminator, False)
    gan_in = Input(shape=(args.latent_dim,))
    g_out = generator(gan_in)
    print(g_out.shape)
    gan_out = discriminator(g_out)
    gan = Model(gan_in, gan_out)
    g_opt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    gan.compile(loss=fence_loss(g_out, args.beta, 2), optimizer=g_opt)
    gan.summary()
    return generator, discriminator, gan

def load_deep_signal_model(args):
    # Building Generator
    g_in = Input(shape=(args.latent_dim,))
    
    x = Dense(64, activation='relu')(g_in)
    ffnn_out = Dense(100, activation='relu')(x)

    top_module = Lambda(lambda x: x[:, 0:50])(ffnn_out)
    x = Reshape((50, 1))(top_module)
    x = Bidirectional(LSTM(8))(x)
    top_out = Dense(68)(x)

    bottom_module = Lambda(lambda x: x[:, 50:])(ffnn_out)
    x = Reshape((1, 50, 1))(bottom_module)
    x = Conv2D(filters=32, kernel_size=(1, 7), strides=2)(x)
    x = AveragePooling2D(pool_size=(1, 7), strides=5)(x)
    x = AveragePooling2D(pool_size=(1, 5), strides=3)(x)
    x = Reshape((-1,))(x)
    bottom_out = Dense(360, activation='tanh')(x)
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
    G.add(Dense(428, activation='tanh'))
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


def load_deep_signal_supervised(args):
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
    d_opt = Adam(lr=0.001, beta_1=0.5, beta_2=0.999)
    D = Model(d_in, d_out)
    D.compile(loss=d_loss, optimizer=d_opt, metrics=['accuracy'])
    D.summary()
    return D
