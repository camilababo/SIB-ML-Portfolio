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

        error_to_propagate = np.dot(error, self.weights.T)

        # updates the weights and bias
        self.weights = self.weights - learning_rate * np.dot(self.x.T, error)  # x.T is used to multiply the error by
        # the input data due to matrix multiplication rules

        self.bias = self.bias - learning_rate * np.sum(error, axis=0)  # sum because the bias has the dimension of
        # nodes and the error has the dimension of samples and nodes (batch size, nodes)

        return error_to_propagate


class SigmoidActivation:
    def __init__(self):
        # attribute
        self.x = None

    @staticmethod
    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.
        :param input_data: input data
        :return: Returns the input data multiplied by the weights.
        """

        return sigmoid_function(input_data)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the layer.
        :return: Returns the error of the previous layer.
        """
        # multiplication of each element by the derivative and not by the entire matrix

        sigmoid_derivative = sigmoid_function(self.x) * (1 - sigmoid_function(self.x))

        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate


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

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the layer.
        :return: Returns the error of the previous layer.
        """

        pass


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

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Computes the backwards pass of the rectified linear relationship.
        :return: Returns the error of the previous layer.
        """

        relu_derivative = np.where(self.x > 0, 1, 0)

        error_to_propagate = error * relu_derivative

        return error_to_propagate


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
