from utils.arguments import parse_args
from utils.data import load_data
from utils.model import load_model
from utils.train_model import pretrain, train
import numpy as np
import tensorflow as tf

args = parse_args()

np.random.seed(args.seed)
tf.set_random_seed(args.seed)

x_train, x_test, y_test, x_val, y_val = load_data(args)
G, D, GAN = load_model(args)
pretrain(args, G, D, x_train)
train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val)
