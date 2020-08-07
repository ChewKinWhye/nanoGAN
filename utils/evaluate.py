import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, roc_curve, f1_score, accuracy_score, recall_score, precision_score
import random


def compute_metrics_standardized(y_predicted, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    au_roc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_predicted_binary = [0 if i < optimal_threshold else 1 for i in y_predicted]
    
    accuracy = accuracy_score(y_test, y_predicted_binary)
    sensitivity = recall_score(y_test, y_predicted_binary)
    # Lazy to calculate
    specificity = -1
    precision = precision_score(y_test, y_predicted_binary)
    cm = confusion_matrix(y_test, y_predicted_binary)

    return accuracy, sensitivity, specificity, precision, au_roc, cm


def compute_metrics_standardized_confident(y_predicted, y_test):
    y_predicted_confident = []
    y_test_confident = []
    for i in range(len(y_predicted)):
        # Confident if it predicts very low or very high
        if y_predicted[i] < 0.45 or y_predicted[i] > 0.55:
            y_predicted_confident.append(y_predicted[i])
            y_test_confident.append(y_test[i])

    fpr, tpr, thresholds = roc_curve(y_test_confident, y_predicted_confident)
    au_roc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_predicted_binary = [0 if i < optimal_threshold else 1 for i in y_predicted_confident]

    accuracy = accuracy_score(y_test_confident, y_predicted_binary)
    sensitivity = recall_score(y_test_confident, y_predicted_binary)
    # Lazy to calculate
    specificity = -1
    precision = precision_score(y_test_confident, y_predicted_binary)
    cm = confusion_matrix(y_test_confident, y_predicted_binary)

    return accuracy, sensitivity, specificity, precision, au_roc, cm


def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    filter_indices = random.sample(range(0, data.shape[0]-1), 3000)
    data_sample = np.take(data, filter_indices, axis=0)
    label_sample = np.take(labels, filter_indices, axis=0)
    z_mean, _, _ = encoder.predict(data_sample)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label_sample, s=0.5)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
