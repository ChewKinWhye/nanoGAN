from utils.custom_losses import *

import keras.backend as K
from keras import losses
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, LSTM,\
    Layer, Lambda, Permute, Flatten, add
from keras.optimizers import Adam

gamma = K.variable([1])


class SequenceFeature(Model):
    def __init__(self):
        super(SequenceFeature, self).__init__()
        self.f_lstm_1 = LSTM(100)
        self.f_lstm_2 = LSTM(100)
        self.f_lstm_3 = LSTM(100)
        self.b_lstm_1 = LSTM(100)
        self.b_lstm_2 = LSTM(100)
        self.b_lstm_3 = LSTM(100)
        self.reverse = Lambda(lambda x: K.reverse(x, axes=1))
        self.reshape1 = Reshape((4, 17))
        self.reshape2 = Reshape((100, 1))
        self.permute = Permute((2, 1), input_shape=(4, 17))

    def call(self, inputs):
        reshaped = self.permute(self.reshape1(inputs))
        layer_1_out = self.reshape2(self.reverse(self.b_lstm_1(self.reshape2(self.reverse(self.f_lstm_1(reshaped))))))
        layer_2_out = self.reshape2(self.reverse(self.b_lstm_2(self.reshape2(self.reverse(self.f_lstm_2(layer_1_out))))))
        layer_3_out = self.reverse(self.b_lstm_3(self.reshape2(self.reverse(self.f_lstm_3(layer_2_out)))))
        return layer_3_out


class Custom(Layer):
    def __init__(self):
        super(Custom, self).__init__()
        # self.dense2 = Dense(428, activation='relu')
        self.top_layer = Lambda(lambda x: x[:, 0:-360])
        self.bottom_layer = Lambda(lambda x: x[-360:])
        self.sequence_feature = SequenceFeature()
        self.dense = Dense(1, activation='relu')

    def call(self, inputs):
        return self.dense(self.top_layer(inputs))


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
    d_in = Input(shape=(428,))
    # Top module to process 4*17 features using LSTM
    top_module = Lambda(lambda x: x[-360:])(d_in)
    top_out = LSTM(100)(top_module)
    # Bottom model to process 360 signals using CNN
    bottom_module = Lambda(lambda x: x[-360:])(d_in)
    bottom_out = Dense(100)(bottom_module)
    # Classification module which combines top and bottom outputs using FFNN
    classification_in = add([top_out, bottom_out])
    d_out = Dense(1)(classification_in)
    d_opt = Adam(lr=args.d_lr, beta_1=0.5, beta_2=0.999)
    D = Model(d_in, d_out)
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
