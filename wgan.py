# example of a wgan for generating handwritten digits
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot
from utils.arguments import parse_args
from utils.data import load_dna_data_gan
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, LSTM
from utils.evaluate import compute_metrics_standardized
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Reshape, LeakyReLU, Input, Dense, BatchNormalization, LSTM,\
    Layer, Lambda, Permute, Flatten, add, Bidirectional, Concatenate, Conv1D, Conv2D,\
    MaxPooling1D, MaxPooling2D, AveragePooling2D

def set_trainability(model, trainable=False):
    # Alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


# define the standalone critic model
def define_critic():
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
    d_out = Dense(1)(x)
    D = Model(d_in, d_out)
    opt = RMSprop(lr=0.00005)
    D.compile(loss=wasserstein_loss, optimizer=opt)
    D.summary()
    return D


# define the standalone generator model
def define_generator():
    # weight initialization
    g_in = Input(shape=(args.latent_dim,))

    x = Dense(128, activation='relu')(g_in)
    ffnn_out = Dense(100, activation='relu')(x)

    top_module = Lambda(lambda x: x[:, 0:50])(ffnn_out)
    x = Reshape((50, 1))(top_module)
    x = Bidirectional(LSTM(8))(x)
    top_out = Dense(68)(x)

    bottom_module = Lambda(lambda x: x[:, 50:])(ffnn_out)
    x = Reshape((1, 50, 1))(bottom_module)
    x = Conv2D(filters=32, kernel_size=(1, 7), strides=1)(x)
    x = AveragePooling2D(pool_size=(1, 7), strides=5)(x)
    x = AveragePooling2D(pool_size=(1, 5), strides=3)(x)
    x = Reshape((-1,))(x)
    bottom_out = Dense(360)(x)
    g_out = Concatenate(axis=1)([top_out, bottom_out])
    G = Model(g_in, g_out)
    G.summary()
    return G


# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
    # make weights in the critic not trainable
    set_trainability(critic, False)
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the critic
    model.add(critic)
    # compile model
    opt = RMSprop(lr=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# load images
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_dna_data_gan()
    # select all of the examples for a given class
    selected_ix = trainy == 7
    X = trainX[selected_ix]
    # expand to 3d, e.g. add channels
    X = expand_dims(X, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels, -1 for 'real'
    y = -ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = ones((n_samples, 1))
    return X, y


# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    # plot history
    pyplot.plot(d1_hist, label='crit_real')
    pyplot.plot(d2_hist, label='crit_fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.savefig('plot_line_plot_loss.png')
    pyplot.close()


# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = [], [],  []
    best_cm = []
    best_au_roc_val, best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc = 0, 0, 0, 0, 0, 0

    # manually enumerate epochs
    for i in range(n_steps):
        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update critic model weights
            c_loss1 = c_model.train_on_batch(X_real, y_real)
            c1_tmp.append(c_loss1)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update critic model weights
            c_loss2 = c_model.train_on_batch(X_fake, y_fake)
            c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)
        # summarize loss on this batch
        print('>%d, c1=%.8f, c2=%.8f g=%.8f' % (i + 1, c1_hist[-1], c2_hist[-1], g_loss))
        # evaluate the model performance every 'epoch'
        if (i + 1) % bat_per_epo == 0:
            y_predicted = 1 - (np.squeeze(c_model.predict_on_batch(x_val)) + 1) / 2
            accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(y_predicted, y_val)
            if au_roc_val > best_au_roc_val:
                best_au_roc_val = au_roc_val
                # Save the best test results
                y_predicted = 1 - np.squeeze(c_model.predict_on_batch(x_test))
                best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc, best_cm = compute_metrics_standardized(y_predicted, y_test)
            print(f"\tAccuracy    : {accuracy_val:.3f}")
            print(f"\tSensitivity : {sensitivity_val:.3f}")
            print(f"\tSpecificity : {specificity_val:.3f}")
            print(f"\tPrecision   : {precision_val:.3f}")
            print(f"\tAUC         : {au_roc_val:.3f}")
            print(f"{cm_val}")

    print('===== End of Adversarial Training =====')
    print(f"\tBest accuracy    : {best_accuracy:.3f}")
    print(f"\tBest sensitivity : {best_sensitivity:.3f}")
    print(f"\tBest specificity : {best_specificity:.3f}")
    print(f"\tBest precision   : {best_precision:.3f}")
    print(f"\tBest AUC         : {best_au_roc:.3f}")
    print(f"{best_cm}")
    # line plots of loss
    plot_history(c1_hist, c2_hist, g_hist)

args = parse_args()
# size of the latent space
latent_dim = args.latent_dim
# create the critic
critic = define_critic()
# create the generator
generator = define_generator()
# create the gan
gan_model = define_gan(generator, critic)
# load image data
x_train, x_test, y_test, x_val, y_val = load_dna_data_gan(args)
print(x_train.shape)
# train model
train(generator, critic, gan_model, x_train, latent_dim, n_epochs=args.epochs)
