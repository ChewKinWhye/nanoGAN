from utils.arguments import parse_args
from utils.data import load_data
from utils.model import load_deep_signal_model, load_model
from utils.train_model import pre_train, train
from utils.save import save_results
from utils.evaluate import compute_metrics, plot_prc
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import activations

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

args = parse_args()

np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)

x_train, x_test, y_test, x_val, y_val = load_data(args)
y_test_ohe = to_categorical(y_test, num_classes=2)

y_val_ohe = to_categorical(y_val, num_classes=2)

model = Sequential([
  Dense(64, activation='relu', input_shape=(428,)),
  Dense(64, activation='relu'),
  Dense(2, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.fit(x_test, y_test, epochs=30, batch_size=128)

y_predicted = np.squeeze(model.predict_on_batch(x_val))
print(y_predicted.shape)
print(y_predicted[:])
y_predicted = y_predicted[:, 1]

results = (compute_metrics(y_predicted, y_val, args.threshold))

save_results(args, results, y_val)
