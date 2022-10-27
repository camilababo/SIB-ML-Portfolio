import numpy as np


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.

    :param x: array of samples.
    :return: array of sigmoid values
    """
    return 1 / (1 + np.exp(-x))
