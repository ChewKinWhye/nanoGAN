import numpy as np
import csv
import os
import random
from sklearn import preprocessing
from sklearn.utils import shuffle


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


def load_rna_data_vae(args):
    train_size = int(args.data_size * 0.8)
    test_size = int(args.data_size * 0.1)
    file_path_normal = os.path.join(args.data_path, "epinano_rna_data.csv")
    X = []
    Y = []
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter=",")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            Y.append(int(row[0]))
            row_float = [float(x) for x in row[1:]]
            X.append(row_float[1:])
    # Normalize between 0 and 1
    min_max_scalar = preprocessing.MinMaxScaler()
    X = min_max_scalar.fit_transform(np.asarray(X))
    Y = np.asarray(Y)
    X, Y = shuffle(X, Y, random_state=0)
    x_train = X[0:train_size, :]
    y_train = Y[0:train_size]
    x_test = X[train_size: train_size + test_size, :]
    y_test = Y[train_size: train_size + test_size]

    x_val = X[train_size + test_size:, :]
    y_val = Y[train_size + test_size:]

    return x_train, y_train, x_test, y_test, x_val, y_val


def load_dna_data_vae(args):
    pass


def load_dna_data_gan(args):
    # Global parameters
    train_size = int(args.data_size * 0.8)
    test_size = int(args.data_size * 0.1)
    val_size = int(args.data_size * 0.1)
    modification_ratio = 0.5
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
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
        outlier_counter = 0
        for row in read_tsv:
            if data_count == total_from_non_modified:
                break
            row_data = []
            # Append the row data values
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]

            # Check for data outliers
            if max(signal_float) > 4 or min(signal_float) < -4 or max(len_float) > 150 or max(sd_float) > 1:
                outlier_counter += 1
                continue
            # Check for data errors
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "0":
                continue

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

            row_data = []
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]

            # Check for data outliers
            if max(signal_float) > 4 or min(signal_float) < -4 or max(len_float) > 150 or max(sd_float) > 1:
                outlier_counter += 1
                continue
                # Check for data errors
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "1":
                continue
            modified_data.append(row_data_float)
            data_count += 1
    
    print(f"Number of outliers: {outlier_counter}")

    # Normalize data
    non_modified_data.extend(modified_data)
    total = np.asarray(non_modified_data)
    feature_1 = total[:, 0:68]
    feature_2 = total[:, 68:85]
    feature_3 = total[:, 85:102]
    feature_4 = total[:, 102:119]
    signals = total[:, 119:]
    # Standardize features by block
    total = [feature_1, feature_2, feature_3, feature_4, signals]
    for i in range(len(total)):
        temp_max = np.amax(total[i])
        temp_min = np.amin(total[i])
        print(temp_max)
        print(temp_min)
        total[i] = (total[i] - temp_min) / (temp_max - temp_min)
    total = list(np.concatenate((total[0], total[1], total[2], total[3], total[4]), axis=1))

    non_modified_data = total[0:total_from_non_modified]
    modified_data = total[total_from_non_modified:]
    
    random.shuffle(non_modified_data)
    random.shuffle(modified_data)

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


def load_multiple_reads_data(args):
    test_size = 1000

    modification_ratio = 0.5
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
    # Global parameters
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")
    total_from_non_modified = 5000000
    total_from_modified = 5000000
    non_modified_duplicate = {}
    non_modified_duplicate_10 = []
    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            if row[3] in non_modified_duplicate:
                non_modified_duplicate[row[3]][0] += 1
            else:
                non_modified_duplicate[row[3]] = [1]
            non_modified_duplicate[row[3]].append(index)
            data_count += 1

    for x in non_modified_duplicate:
        if non_modified_duplicate[x][0] >= 10:
            non_modified_duplicate_10.append(non_modified_duplicate[x])

    modified_duplicate = {}
    modified_duplicate_10 = []
    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            if data_count == total_from_modified:
                break
            if row[3] in modified_duplicate:
                modified_duplicate[row[3]][0] += 1
            else:
                modified_duplicate[row[3]] = [1]
            modified_duplicate[row[3]].append(index)
            data_count += 1
    for x in modified_duplicate:
        if modified_duplicate[x][0] >= 10:
            modified_duplicate_10.append(modified_duplicate[x])
    print(len(modified_duplicate_10))
    print(len(non_modified_duplicate_10))
    test_x = modified_duplicate_10[10000:20000]
    test_x.extend(non_modified_duplicate_10[10000:20000])
    test_y = np.append(np.ones(10000), np.zeros(10000))
    return test_x, test_y
