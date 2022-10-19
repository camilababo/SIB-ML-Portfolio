import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes the accuracy of the model.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The error value between y_true and y_pred.
    """

    # calculates the number of correct predictions by comparing the true labels with the predicted labels
    # negative, true positive, false negative and false positive

    return np.sum(y_true == y_pred) / len(y_true)
