from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


algorithm_type = Literal['static_alpha', 'half_alpha']


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = None

    def gradient_descent(self, dataset: Dataset):
        """
        Computes the gradient descent of the model
        :param dataset: The dataset to compute the gradient descent on.
        :return: The gradient descent of the model
        """
        m, n = dataset.shape()

        # predicted y
        y_pred = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

        # computed the gradient descent and updates with the learning rate
        gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)

        # computing the penalty
        penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

        # updating the model parameters
        self.theta = self.theta - gradient - penalization_term
        self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

    def fit(self, dataset: Dataset, gradient_descent_algorithm: algorithm_type = 'static_alpha') \
            -> 'LogisticRegression':
        """
        Fits the model to the dataset
        :param dataset: The dataset to fit the model on.
        :param gradient_descent_algorithm: The gradient descent algorithm to use. There are two option 'static_alpha'
        where no alterations are applied to the alpha or 'half_alpha' where the value of alpha is set to half everytime
        the cost function value remains the same.
        :return: A Logistic Regression object of the fitted model
        """
        m, n = dataset.shape()

        # Initializes the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # cost history
        self.cost_history = {}

        # check if gradient descent algorithm of choice is valid
        options = get_args(algorithm_type)
        assert gradient_descent_algorithm in options, f"'{gradient_descent_algorithm}' is not in {options}"

        # Gradient descent
        for i in range(self.max_iter):

            self.gradient_descent(dataset)

            # computes the cost function
            self.cost_history[i] = self.cost(dataset)

            # condition to stop gradient descent when cost function value doesn't change
            threshold = 0.0001

            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < threshold:

                if gradient_descent_algorithm == 'half_alpha':
                    # change alpha value to half
                    self.alpha = self.alpha / 2

                if gradient_descent_algorithm == 'static_alpha':
                    break

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dataset and converts them to binary.
        :return: A vector of predictions
        """
        predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

        mask = predictions >= 0.5  # mask for the predictions that are greater than 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset):
        """
        Computes the accuracy of the model
        :param dataset: The dataset to compute the score function on.
        :return: The accuracy of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the cost of the model
        :param dataset: The dataset to compute the cost function on.
        :return: The cost of the model
        """
        sample_n = dataset.shape()[0]

        predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

        cost = (- dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / sample_n
        # regularization term
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * sample_n))

        return cost

    def cost_function_plot(self):
        """
        Plots the cost function history of the model

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        iter = list(self.cost_history.keys())
        val = list(self.cost_history.values())

        plt.plot(iter, val, '-r')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
