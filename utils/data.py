import numpy as np
from keras.datasets import mnist, cifar10
import csv

def preprocess(x):
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x


def load_data(args):
    ###Dataset: mnist, cifar10, kdd99, custom
    ###Ano_class: 1 actual class label of the dataset
    if args.dataset == 'mnist':
        return get_mnist(args.ano_class)
    elif args.dataset == 'cifar10':
        return get_cifar10(args.ano_class)
    elif args.dataset == "ecoli":
        return get_ecoli(args.ano_class)

def get_ecoli(ano_class):
    train_size = 100000
    test_size = 20000
    modification_ratio = 0.1
    # Global parameters
    file_path_normal = "./data/pcr.tsv"
    file_path_modified = "./data/msssi.tsv"

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
            row_data = row[-2].split(",")
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 360 or row[-1] != "0" or max(row_data_float) > 5 or min(row_data_float) < -5:
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            row_data_float = [float(i) for i in row_data]
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
            row_data = row[-2].split(",")
            row_data_float = [float(i) for i in row_data]
            # Check for data inconsistencies, and to only use the template strand
            if row[5].lower() == 'c' or len(row_data) != 360 or row[-1] != "1" or max(row_data_float) > 5 or min(row_data_float) < -5:
                continue
            # The last row represents the methylation state. We only want to train the model on unmethylated datapoints
            row_data_float = [float(i) for i in row_data]
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


def get_mnist(ano_class):
    
    '''
    There are 2 classes: normal and anomalous
    - Training data: x_train: 80% of data/images from normal classes
    - Validation data: x_val: 5% of data/images from normal classes + 25% of data/images from anomalous classes
    - Testing data: x_test: 15% of data/images from normal classes + 75% of data/images from anomalous classes
    '''
    (x_tr, y_tr), (x_tst, y_tst) = mnist.load_data()
    
    x_total = np.concatenate([x_tr, x_tst])
    y_total = np.concatenate([y_tr, y_tst])
    print(x_total.shape, y_total.shape)
    
    x_total = x_total.reshape(-1, 28, 28, 1)
    x_total = preprocess(x_total)

    delete = []
    for count, i in enumerate(y_total):
        if i == ano_class:
            delete.append(count)
    
    ano_data = x_total[delete,:]
    normal_data = np.delete(x_total,delete,axis=0)
    
    normal_num = normal_data.shape[0] #Number of data/images of normal classes
    ano_num = ano_data.shape[0] #Number of data/images of anomalous classes
    
    del x_total
    
    x_train = normal_data[:int(0.8*normal_num),...]
    
    x_test = np.concatenate((normal_data[int(0.8*normal_num):int(0.95*normal_num),...], ano_data[:ano_num*3//4]))
    y_test = np.concatenate((np.ones(int(0.95*normal_num)-int(0.8*normal_num)), np.zeros(ano_num*3//4)))
    
    x_val = np.concatenate((normal_data[int(0.95*normal_num):,...], ano_data[ano_num*3//4:]))
    y_val = np.concatenate((np.ones(normal_num-int(0.95*normal_num)), np.zeros(ano_num-ano_num*3//4)))
    
    return x_train, x_test, y_test, x_val, y_val

def get_cifar10(ano_class):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train_anomaly = X_train[np.where(y_train == ano_class)[0]]
    X_train_normal = X_train[np.where(y_train != ano_class)[0]]
    X_test_anomaly = X_test[np.where(y_test == ano_class)[0]]
    X_test_normal = X_test[np.where(y_test != ano_class)[0]]

    X_normal = np.concatenate([X_train_normal, X_test_normal])
    X_anomaly = np.concatenate([X_train_anomaly, X_test_anomaly])

    '''
    Pick 20% of X_normal to be in Test set. Place the remaining in Train + Validation set
    '''
    idx = np.random.choice(X_normal.shape[0], int(0.2 * X_normal.shape[0]), replace=False)
    X_test_normal = np.array([X_normal[i] for i in idx])
    X_trainval_normal = np.array([X_normal[i] for i in range(X_normal.shape[0]) if i not in idx])
    
    '''
    Pick 20% of X_anomaly to be in Validation set. Place the remaining in Test Set
    '''
    idx = np.random.choice(X_anomaly.shape[0], int(0.2 * X_anomaly.shape[0]), replace=False) 
    X_val_anomaly = np.array([X_anomaly[i] for i in idx])
    X_test_anomaly = np.array([X_anomaly[i] for i in range(X_anomaly.shape[0]) if i not in idx])
    
    '''
    Compute the ratio of normal data and anomaly data in the Test Set
    '''
    ratio = int(X_val_anomaly.shape[0] * X_test_normal.shape[0] / X_test_anomaly.shape[0]) #Compute Validation Ratio between Normal and Anomaly Samples
    idx = np.random.choice(X_trainval_normal.shape[0], ratio, replace=False)
    X_val_normal = np.array([X_trainval_normal[i] for i in idx])
    
    '''
    Form the Train, Validation and Test Sets
    '''
    X_train = np.array([X_trainval_normal[i] for i in range(X_trainval_normal.shape[0]) if i not in idx])
    X_val = np.concatenate([X_val_normal, X_val_anomaly])
    X_test = np.concatenate([X_test_normal, X_test_anomaly])

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)
    
    '''
    Prepare Labels
    '''
    y_val_anomaly = [0] * X_val_anomaly.shape[0] #anomalies are labeled '0'
    y_val_normal = [1] * X_val_normal.shape[0] #Normal samples are labeled '1'
    y_val = np.asarray(np.concatenate([y_val_normal, y_val_anomaly]))
    
    y_test_anomaly = [0] * X_test_anomaly.shape[0]
    y_test_normal = [1] * X_test_normal.shape[0]
    y_test = np.asarray(np.concatenate([y_test_normal, y_test_anomaly]))
    
    return X_train, X_test, y_test, X_val, y_val

#def get_kdd99():
    
#def get_custom():
