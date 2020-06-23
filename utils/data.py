import numpy as np
import csv
import os
import random


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
        x_gen = G.predict(noise)
        y0 = np.zeros(n_samples)

        return x_gen, y0


def load_data(args):
    train_size = int(args.data_size * 0.8)
    test_size = int(20000 * 0.2)
    modification_ratio = 0.1
    dna_lookup = {"A": 0, "T": 1, "G": 2, "C": 3}
    # Global parameters
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")

    test_from_non_modified = int(test_size * (1 - modification_ratio))
    total_from_non_modified = int(test_from_non_modified + train_size)
    test_from_modified = int(test_size * modification_ratio)
    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        train_data = []
        data_count = 0
        for row in read_tsv:
            if data_count == total_from_non_modified:
                break
            # The second last row contains the 360 signal values, separated by commas
            row_data = row[7].split(",")
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 360 or row[-1] != "0":
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            train_data.append(row_data_float)
            data_count += 1

    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        test_data = []
        data_count = 0
        for i, row in enumerate(read_tsv):
            # 3000 test data points
            if data_count == test_from_modified:
                break
            # The second last row contains the 360 signal values, separated by commas
            row_data = row[7].split(",")
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 360 or row[-1] != "1":
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            test_data.append(row_data_float)
            data_count += 1

    test_data.extend(train_data[-test_from_non_modified:])
    test_data = np.asarray(test_data)
    train_data = np.asarray(train_data[0:-test_from_non_modified])
    test_data_labels = np.append(np.ones(test_from_modified), np.zeros(test_from_non_modified))
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data labels shape: {test_data_labels.shape}")
    return train_data, test_data, test_data_labels, test_data, test_data_labels
