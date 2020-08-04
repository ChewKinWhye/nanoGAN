import numpy as np
import csv
import os
import random
from sklearn import preprocessing
from arguments import parse_args


def load_data(args):
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

