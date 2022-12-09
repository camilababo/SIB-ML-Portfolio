import numpy as np
from src.si.statistics.sigmoid_function import sigmoid_function


class Dense:
    def __init__(self, input_size: int, output_size: int):
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        # weight matriz initialization
        shape = (input_size, output_size)

        self.x = None
        self.weights = np.random.randn(*shape) * 0.01  # 0.01 is a hyperparameter to avoid exploding
        # gradients
        # each layer receives a weight that multiplies by the input that are then summed
        self.bias = np.zeros((1, output_size))  # bias initialization, receives a bias to avoid overfitting

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        :param x: input data value
        :return: Returns the input data multiplied by the weights.
        """
        self.x = x
        # the input_data needs to be a matrix with the same number of columns as the number of features
        # the number os columns of the input_data must be equal to the number of rows of the weights
        return np.dot(x, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Computes the backward pass of the layer
        :param error: error value of the loss function
        :param learning_rate: learning rate
        :return: Returns the error of the previous layer.
        """

        # updates the weights and bias
        self.weights -= learning_rate * np.dot(self.x.T, error)  # x.T is used to multiply the error by the input
        # data due to matrix multiplication rules


class SigmoidActivation:
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        :param input_data: input data
        :return: Returns the input data multiplied by the weights.
        """

        return sigmoid_function(input_data)


class SoftMaxActivation:
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the probability of each class.
        :param input_data: input data
        :return: Returns the probability of each class.
        """

        zi_exp = np.exp(input_data - np.max(input_data))
        formula = zi_exp / np.sum(zi_exp, axis=1, keepdims=True)  # axis=1 means that the sum is done by row
        # if set to True will keep the dimension of the array

        return formula


class ReLUActivation:
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the rectified linear relationship.
        :param input_data: input data
        :return: Returns the rectified linear relationship.
        """

        data_pos = np.maximum(0, input_data)  # maximum between 0 and the input_data, the 0 is to avoid negative values

        return data_pos


class LinearActivation:
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the linear relationship.
        :param input_data: input data
        :return: Returns the linear relationship.
        """

        return input_data
