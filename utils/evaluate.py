import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def compute_metrics(discriminator, x, y):
    y_predicted = 1 - np.squeeze(discriminator.predict((x)))
    precision, recall, _ = precision_recall_curve(y, y_predicted)
    au_prc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y, y_predicted)
    au_roc = auc(fpr, tpr)

    return au_prc, precision, recall, au_roc, fpr, tpr


def plot_prc(precision, recall):
    # Plot recall-precision curve
    plt.plot(recall, precision, 'r-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    return plt
