import numpy as np
import csv
import os
import random
from sklearn import preprocessing
from sklearn.utils import shuffle
import itertools


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
    file_path_normal = os.path.join(args.data_path, "ecoli_MSssI_50mil_coverage10_readqual_extracted.tsv")
    file_path_modified = os.path.join(args.data_path, "ecoli_pcr_50mil_coverage10_readqual_extracted.tsv")
    X = []
    Y = []
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            if i == int(args.data_size/2) + 1:
                break
            Y.append(int(0))
            row_float = [float(x) for x in row[3:]]
            X.append(row_float)

    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            if i == int(args.data_size/2) + 1:
                break
            Y.append(int(1))
            row_float = [float(x) for x in row[3:]]
            X.append(row_float)
    # Normalize between 0 and 1
    min_max_scalar = preprocessing.MinMaxScaler()
    X = min_max_scalar.fit_transform(np.asarray(X))
    Y = np.asarray(Y)
    print(X.shape)
    X, Y = shuffle(X, Y, random_state=0)
    x_train = X[0:train_size, :]
    y_train = Y[0:train_size]
    x_test = X[train_size: train_size + test_size, :]
    y_test = Y[train_size: train_size + test_size]

    x_val = X[train_size + test_size:, :]
    y_val = Y[train_size + test_size:]

    return x_train, y_train, x_test, y_test, x_val, y_val


def load_dna_data_vae(args):
    # Global parameters
    train_size = int(args.data_size * 0.8 / 2)
    test_size = int(args.data_size * 0.1 / 2)
    val_size = int(args.data_size * 0.1 / 2)
    total_size = train_size + test_size + val_size

    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")

    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        non_modified_data = []
        data_count = 0
        outlier_counter = 0
        for row in read_tsv:
            if data_count == total_size:
                break
            row_data = []
            # Append the row data values
            if row[6][6:11] != "ATCGA":
                continue
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend([float(i)*10 for i in row[10].split(",")])
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]
            # Check for data outliers
            if max(signal_float) > 8 or min(signal_float) < -8 or max(len_float) > 300 or max(sd_float) > 2:
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
            if data_count == total_size:
                break
            if row[6][6:11] != 'ATCGA':
                continue
            row_data = []
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend([float(i)*10 for i in row[10].split(",")])
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]
            # Check for data outliers
            if max(signal_float) > 8 or min(signal_float) < -8 or max(len_float) > 300 or max(sd_float) > 2:
                outlier_counter += 1
                continue
            # Check for data errors
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "1":
                continue
            modified_data.append(row_data_float)
            data_count += 1

    print(f"Number of outliers: {outlier_counter}")

    random.shuffle(non_modified_data)
    random.shuffle(modified_data)

    train_x = modified_data[0:train_size]
    train_x.extend(non_modified_data[0:train_size])
    train_x = np.asarray(train_x)
    train_y = np.append(np.ones(train_size), np.zeros(train_size))
    train_y.astype(int)
    train_x, train_y = shuffle(train_x, train_y, random_state=0)

    test_x = modified_data[train_size:train_size + test_size]
    test_x.extend(non_modified_data[train_size:train_size + test_size])
    test_x = np.asarray(test_x)
    test_y = np.append(np.ones(test_size), np.zeros(test_size))
    test_y.astype(int)

    val_x = modified_data[train_size + test_size:]
    val_x.extend(non_modified_data[train_size + test_size:])
    val_x = np.asarray(val_x)
    val_y = np.append(np.ones(val_size), np.zeros(val_size))
    val_y.astype(int)

    print(f"Train data shape: {train_x.shape}")
    print(f"Train data labels shape: {train_y.shape}")
    print(f"Test data shape: {test_x.shape}")
    print(f"Test data labels shape: {test_y.shape}")
    print(f"Validation data shape: {val_x.shape}")
    print(f"Validation data labels shape: {val_y.shape}")

    return train_x, train_y, test_x, test_y, val_x, val_y, 0, 0


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
            row_data.extend([float(i)*5 for i in row[10].split(",")])
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
            row_data.extend([float(i)*5 for i in row[10].split(",")])
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
    test_size = 2500
    # Global parameters
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")
    total_size = 2000000
    non_modified_duplicate = {}
    non_modified_duplicate_10 = []
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
    test_x = []
    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            if data_count == total_size:
                break
            if row[6][6:11] != 'ATCGA':
                continue
            if row[3] not in non_modified_duplicate:
                non_modified_duplicate[row[3]] = []
            # Append data instead of index
            row_data = []
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend([float(i)*10 for i in row[10].split(",")]) 
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]
            # Check for data outliers
            if max(signal_float) > 8 or min(signal_float) < -8 or max(len_float) > 300 or max(sd_float) > 2:
                continue
            # Check for data errors
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "0":
                continue
            non_modified_duplicate[row[3]].append(row_data_float)
            data_count += 1
    # Find the ones with more than 10 reads
    for x in non_modified_duplicate:
        if len(non_modified_duplicate[x]) >= 10:
            test_x.append(non_modified_duplicate[x][0:10])
    non_modified_duplicate.clear()
    test_x = test_x[0:test_size]
    modified_duplicate = {}
    modified_duplicate_10 = []
    # Extract data from modified
    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            if data_count == total_size:
                break
            if row[6][6:11] != 'ATCGA':
                continue
            if row[3] not in modified_duplicate:
                modified_duplicate[row[3]] = []
            # Append data instead of index
            row_data = []
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend([float(i)*10 for i in row[10].split(",")]) 
            row_data_float = [float(i) for i in row_data]
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]
            # Check for data outliers
            if max(signal_float) > 8 or min(signal_float) < -8 or max(len_float) > 300 or max(sd_float) > 2:
                continue
            # Check for data errors
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "1":
                continue
            modified_duplicate[row[3]].append(row_data_float)
            data_count += 1

    # Find the ones with more than 10 reads
    for x in modified_duplicate:
        if len(modified_duplicate[x]) >= 10:
            test_x.append(modified_duplicate[x][0:10])
    test_x = test_x[0:2 * test_size]
    test_x = np.asarray(test_x)
    print(test_x.shape)
    test_y = np.append(np.zeros(test_size), np.ones(test_size))
    return test_x, test_y
