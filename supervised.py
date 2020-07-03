from utils.arguments import parse_args
from utils.data import load_data
from utils.model import load_deep_signal_supervised
from utils.save import save_results
from utils.evaluate import compute_metrics, plot_prc


import numpy as np
import tensorflow as tf

args = parse_args()

np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)

x_train, x_test, y_test, x_val, y_val = load_data(args)
weights = {0: 1,
           1: 9}
model = load_deep_signal_supervised(args)
model.fit(x_test, y_test, epochs=1, batch_size=128, class_weight=weights)

y_predicted = np.squeeze(model.predict_on_batch(x_val))
results = (compute_metrics(y_predicted, y_val, args.threshold))

save_results(args, results, y_val)
