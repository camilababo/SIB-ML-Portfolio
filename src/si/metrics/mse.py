import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

    :param y_true: The true labels of the dataset.
    :param y_pred: The predicted labels of the dataset.
    :return: The mean squared error of the model
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)
