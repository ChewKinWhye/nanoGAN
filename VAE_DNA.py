from tensorflow import keras
import numpy as np
from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, plot_label_clusters_10
from utils.vae_model import VAE_DNA
from utils.save import save_vae_model_dna

latent_dim = 20

args = parse_args()
x_train, y_train, x_test, y_test, x_val, y_val, min_values, max_values = load_dna_data_vae(args)

# Train VAE
vae = VAE_DNA(latent_dim)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=100, batch_size=128)

# Visualize cluster
print(x_train[0:5, 24:44])
print(vae.decoder.predict(vae.encoder.predict(x_train[0:5, :]))[:, 24:44])
predictor = vae.predictor
encoder = vae.encoder
plot_label_clusters(args.output_filename, encoder, x_train, y_train)

# Train predictor
predictor_size = int(len(x_train)/10)
x_train_mean, x_train_sd, _ = encoder.predict(x_train[0:predictor_size])
x_train = np.concatenate((x_train_mean, x_train_sd), axis=1)
predictor.fit(x_train, y_train[0:predictor_size], epochs=30, batch_size=128)

# Test model
x_test_mean, x_test_sd, _ = encoder.predict(x_test)
x_test = np.concatenate((x_test_mean, x_test_sd), axis=1)
pred_out = predictor.predict(x_test)
accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(
    pred_out, y_test)

# Save model
#save_vae_model_dna(encoder, predictor, min_values, max_values)

# Print results
print(f"\tAccuracy    : {accuracy_val:.3f}")
print(f"\tSensitivity : {sensitivity_val:.3f}")
print(f"\tSpecificity : {specificity_val:.3f}")
print(f"\tPrecision   : {precision_val:.3f}")
print(f"\tAUC         : {au_roc_val:.3f}")
print(f"{cm_val}")

test_x_10, test_y_10 = load_multiple_reads_data(args)
plot_label_clusters_10(args.output_filename, encoder, test_x_10, test_y_10)

predictions = []
for x in test_x_10:
    test_x_mean, test_x_sd, _ = encoder.predict(x)
    test_x_predictor = np.concatenate((test_x_mean, test_x_sd), axis=1)
    test_x_prediction = predictor.predict(test_x_predictor)
    predictions.append(np.average(test_x_prediction))

accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(
        np.asarray(predictions), test_y_10)

# Print results
print(f"\tAccuracy    : {accuracy_val:.3f}")
print(f"\tSensitivity : {sensitivity_val:.3f}")
print(f"\tSpecificity : {specificity_val:.3f}")
print(f"\tPrecision   : {precision_val:.3f}")
print(f"\tAUC         : {au_roc_val:.3f}")
print(f"{cm_val}")
