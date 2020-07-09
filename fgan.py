from utils.arguments import parse_args
from utils.data import load_data
from utils.model import load_deep_signal_model, load_model
from utils.train_model import pre_train, train

import numpy as np
import tensorflow as tf

args = parse_args()

np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)

x_train, x_test, y_test, x_val, y_val = load_data(args)
generator, discriminator, GAN = load_model(args)
pre_train(args, generator, discriminator, x_train)
results = train(args, generator, discriminator, GAN, x_train, x_test, y_test, x_val, y_val)