import numpy as np
import csv
import os
import random
from sklearn import preprocessing

def noise_data(n_samples, latent_dim):
    return np.random.normal(0, 1, [n_samples, latent_dim])


def D_data(n_samples, G, mode, x_train, latent_dim):
    # Feeding training data for normal case
    if mode == 'normal':
        sample_list = random.sample(list(range(np.shape(x_train)[0])), n_samples)
        x_normal = x_train[sample_list, ...]
        y1 = np.ones(n_samples)

        return x_normal, y1

    # Feeding training data for generated case
    if mode == 'gen':
        noise = noise_data(n_samples, latent_dim)
        x_gen = G.predict_on_batch(noise)
        y0 = np.zeros(n_samples)

        return x_gen, y0


def load_data(args):
    train_size = int(args.data_size * 0.8)
    test_size = int(args.data_size * 0.1)
    val_size = int(args.data_size * 0.1)

    modification_ratio = 0.5
    dna_lookup = {"A": 0, "T": 1, "G": 2, "C": 3}
    # Global parameters
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")

    test_from_non_modified = int(test_size * (1 - modification_ratio))
    val_from_non_modified = int(val_size * (1 - modification_ratio))
    total_from_non_modified = int(test_from_non_modified + val_from_non_modified + train_size)

    test_from_modified = int(test_size * modification_ratio)
    val_from_modified = int(val_size * modification_ratio)
    total_from_modified = test_from_modified + val_from_modified

    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        non_modified_data = []
        data_count = 0
        for row in read_tsv:
            if data_count == total_from_non_modified:
                break
            # The second last row contains the 360 signal values, separated by commas
            row_data = [dna_lookup[i] for i in row[6]]
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 428 or row[-1] != "0":
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            non_modified_data.append(row_data_float)
            data_count += 1

    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        modified_data = []
        data_count = 0
        for i, row in enumerate(read_tsv):
            # 3000 test data points
            if data_count == total_from_modified:
                break
            # The second last row contains the 360 signal values, separated by commas
            row_data = [dna_lookup[i] for i in row[6]]
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 428 or row[-1] != "1":
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            modified_data.append(row_data_float)
            data_count += 1
    
    random.shuffle(non_modified_data)
    random.shuffle(modified_data)

    # Normalize data
    non_modified_data = list(preprocessing.normalize(np.asarray(non_modified_data)))
    modified_data = list(preprocessing.normalize(np.asarray(modified_data)))

    train_data = np.asarray(non_modified_data[0:train_size])

    test_data = modified_data[0:test_from_modified]
    test_data.extend(non_modified_data[train_size:train_size + test_from_non_modified])
    test_data = np.asarray(test_data)
    test_data_labels = np.append(np.ones(test_from_modified), np.zeros(test_from_non_modified))
    
    val_data = modified_data[test_from_modified:]
    val_data.extend(non_modified_data[train_size + test_from_non_modified:])
    val_data = np.asarray(val_data)
    val_data_labels = np.append(np.ones(val_from_modified), np.zeros(val_from_non_modified))
    val_data_labels.astype(int)
    test_data_labels.astype(int)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data labels shape: {test_data_labels.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation data labels shape: {val_data_labels.shape}")
    return train_data, test_data, test_data_labels, val_data, val_data_labels

