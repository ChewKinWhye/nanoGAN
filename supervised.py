from utils.arguments import parse_args
from utils.data import load_data
from utils.evaluate import compute_metrics_standardized
from utils.model import load_deep_signal_supervised

import numpy as np
import tensorflow as tf

args = parse_args()
np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)

x_train, x_test, y_test, x_val, y_val = load_data(args)
model = load_deep_signal_supervised(args)

model.fit(x_test, y_test, epochs=150, batch_size=512, validation_data=(x_val, y_val))
y_predicted = np.squeeze(model.predict_on_batch(x_val))
results = (compute_metrics_standardized(y_predicted, y_val))

print(results)
