from data import load_multiple_reads_data
from arguments import parse_args
from evaluate import compute_metrics_standardized
from gan_model import load_dc_gan_model
import itertools
import csv
import numpy as np
import os


args = parse_args()
x_modified, x_non_modified, y_modified, y_non_modified = load_multiple_reads_data(args)

predictions = []

# Load model
_, model, _ = load_dc_gan_model(args)
model.load_weights('results/fgan/discriminator.h5')
file_path_normal = os.path.join(args.data_path, "pcr.tsv")
file_path_modified = os.path.join(args.data_path, "msssi.tsv")
dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}

with open(file_path_normal, 'r') as f:
    for row in x_modified:
        row_input = []
        for index in row[1:]:
            temp_data = next(itertools.islice(csv.reader(f), index, None))
            row_data = []
            # Append the row data values
            for i in temp_data[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(temp_data[7].split(","))
            row_data.extend(temp_data[8].split(","))
            row_data.extend(temp_data[9].split(","))
            row_data.extend(temp_data[10].split(","))
            row_data_float = [float(i) for i in row_data]

            row_input.append(row_data_float)
        # Row input has 10 rows corresponding to 10 reads for a particular position
        row_input = np.asarray(row_input)
        # Normalize and stuff
        feature_1 = row_input[:, 0:68]
        feature_2 = row_input[:, 68:85]
        feature_3 = row_input[:, 85:102]
        feature_4 = row_input[:, 102:119]
        signals = row_input[:, 119:]
        # Standardize features by block
        total = [feature_1, feature_2, feature_3, feature_4, signals]
        # Find temp_min and temp_max
        temp_min, temp_max = 0, 0
        for i in range(len(total)):
            total[i] = (total[i] - temp_min) / (temp_max - temp_min)
        total = list(np.concatenate((total[0], total[1], total[2], total[3], total[4]), axis=1))
        temp_prediction = 1 - np.squeeze(model.predict_on_batch(total))
        predictions.append(sum(temp_prediction)/len(temp_prediction))

print(len(predictions))
best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc, best_cm = compute_metrics_standardized(predictions, y_modified)

print(f"\tAccuracy    : {best_accuracy:.3f}")
print(f"\tSensitivity : {best_sensitivity:.3f}")
print(f"\tSpecificity : {best_specificity:.3f}")
print(f"\tPrecision   : {best_precision:.3f}")
print(f"\tAUC         : {best_au_roc:.3f}")
print(f"{best_cm}")
