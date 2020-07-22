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
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
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
        counter = 0
        for row in read_tsv:
            if data_count == total_from_non_modified:
                break
            # The second last row contains the 360 signal values, separated by commas
            #if row[6][6:11] != 'ATCGA':
            #    continue
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
            if max(signal_float) > 4 or min(signal_float) < -4 or max(len_float) > 150 or max(sd_float) > 1:
                counter += 1
                continue
            
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "0":
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
            #if row[6][6:11] != 'ATCGA':
            #    continue
            row_data = []
            for i in row[6]:
                row_data.extend(dna_lookup[i])
            row_data.extend(row[7].split(","))
            row_data.extend(row[8].split(","))
            row_data.extend(row[9].split(","))
            row_data.extend(row[10].split(","))
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            signal_float = [float(i) for i in row[10].split(",")]
            len_float = [float(i) for i in row[9].split(",")]
            sd_float = [float(i) for i in row[8].split(",")]
            if max(signal_float) > 4 or min(signal_float) < -4 or max(len_float) > 150 or max(sd_float) > 1:
                counter += 1
                continue
            
            if row[5].lower() == 'c' or len(row_data) != 479 or row[-1] != "1":
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            modified_data.append(row_data_float)
            data_count += 1
    
    print(f"COUNTER: {counter}")
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
    '''
    signal_max = np.amax(signals)
    signal_min = np.amin(signals)
    signals = (signals - signal_min) / (signal_max - signal_min)
    total_min_max = list(np.concatenate((features, signals), axis=1))
    '''
    total = list(np.concatenate((total[0], total[1], total[2], total[3], total[4]), axis=1))
    non_modified_data = total[0:total_from_non_modified]
    modified_data = total[total_from_non_modified:]
    
    random.shuffle(non_modified_data)
    random.shuffle(modified_data)
    # Normalize data

    print(non_modified_data[0][24:44])
    print(non_modified_data[1][24:44])
    print(non_modified_data[2][24:44])
    print(modified_data[0][24:44])
    print(modified_data[1][24:44]) 
    print(modified_data[2][24:44]) 
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

