import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def compute_metrics(y_predicted, y_test):
    precision, recall, _ = precision_recall_curve(y_test, y_predicted)
    au_prc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    au_roc = auc(fpr, tpr)

    return au_prc, recall, precision, au_roc, fpr, tpr


def plot_prc(results, y_test):
    # Plot recall-precision and fpr-tpr curve
    plt.plot(results[1], results[2], 'r-', label="Precision-Recall curve of model")
    plt.plot(results[4], results[5], 'b-', label="FPR-TPR curve of model")

    # Plot for random predictions
    random_predictions = np.random.random_sample((len(y_test)))
    rand_au_prc, rand_recall, rand_precision, rand_au_roc, rand_fpr, rand_tpr \
        = compute_metrics(random_predictions, y_test)
    plt.plot(rand_recall, rand_precision, 'r:', label="Precision-Recall curve of random")
    plt.plot(rand_fpr, rand_tpr, 'b:', label="FPR-TPR curve of random")

    # Plot for Majority vote
    # majority_predictions = np.ones((len(y_test)))
    # np.random.shuffle(y_test)
    # majority_au_prc, majority_recall, majority_precision, majority_au_roc, majority_fpr, majority_tpr \
    #     = compute_metrics(majority_predictions, y_test)
    # plt.plot(majority_recall, majority_precision, 'r:', label="Precision-Recall curve of majority vote classifier")
    # plt.plot(majority_fpr, majority_tpr, 'b:', label="FPR-TPR curve of majority vote classifier")

    # Axis labels
    plt.xlabel('Recall/FPR')
    plt.ylabel('Precision/TPR')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    return plt
