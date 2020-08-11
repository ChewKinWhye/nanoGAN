from tensorflow import keras
import numpy as np
from utils.data import load_rna_data_vae
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters
from utils.vae_model import VAE_RNA
from utils.save import save_vae_model_rna

latent_dim = 10

args = parse_args()
x_train, y_train, x_test, y_test, x_val, y_val = load_rna_data_vae(args)

# Train encoder
vae = VAE_RNA(latent_dim)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128)

# Visualize cluster
encoder = vae.encoder
predictor = vae.predictor
plot_label_clusters(encoder, x_train, y_train)

# Train predictor
x_train_mean, x_train_sd, _ = encoder.predict(x_train[0:5000])
x_train = np.concatenate((x_train_mean, x_train_sd), axis=1)
predictor.fit(x_train, y_train[0:5000], epochs=30, batch_size=128)

# Test model
x_test_mean, x_test_sd, _ = encoder.predict(x_test)
x_test = np.concatenate((x_test_mean, x_test_sd), axis=1)
pred_out = predictor.predict(x_test)
accuracy_val, sensitivity_val, specificity_val, precision_val, au_roc_val, cm_val = compute_metrics_standardized(
    pred_out, y_test)

# Save model
save_vae_model_rna(args, encoder, predictor)

# Print results
print(f"\tAccuracy    : {accuracy_val:.3f}")
print(f"\tSensitivity : {sensitivity_val:.3f}")
print(f"\tSpecificity : {specificity_val:.3f}")
print(f"\tPrecision   : {precision_val:.3f}")
print(f"\tAUC         : {au_roc_val:.3f}")
print(f"{cm_val}")
