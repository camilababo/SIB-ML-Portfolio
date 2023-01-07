from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

alpha_options = Literal['static_alpha', 'half_alpha']


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    cost_history: dict
        The history of the cost function of the model.
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000,
                 alpha_type: alpha_options = 'static_alpha'):
        """
        Initializes the RidgeRegression model

        :param l2_penalty: The L2 regularization parameter
        :param alpha: The learning rate
        :param max_iter: The maximum number of iterations
        :param alpha_type: The gradient descent algorithm to use. There are two options: (1) 'static_alpha': where no
        alterations are applied to the alpha; or (2) 'half_alpha' where the value of alpha is set to half everytime
        the cost function value remains the same.
        """
        # parameters
        self.l2_penalty = l2_penalty  # l2 regularization parameter
        self.alpha = alpha  # learning rate, set as a low value to not jump over the minimum
        self.max_iter = max_iter
        self.alpha_type = alpha_type

        # attributes
        self.theta = None  # model coefficient
        self.theta_zero = None  # f function of a linear model
        self.cost_history = None  # history of the cost function

    def gradient_descent(self, dataset: Dataset):
        """
        Computes the gradient descent of the model
        :param dataset: The dataset to compute the gradient descent on.
        :return: The gradient descent of the model
        """
        m, n = dataset.shape()

        # predicted y
        y_pred = np.dot(dataset.x, self.theta) + self.theta_zero  # corresponds to the classical function of
        # y = mx + b

        # computing and updating the gradient with the learning rate
        gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)  # calculates the
        # gradient of the cost function
        # np.dot sums the colum values of the multiplication arrays
        # learning rate is multiplicated by 1/m to normalize the rate to the dataset size

        # computing the penalty
        penalization_term = self.alpha * (self.l2_penalty / m) * self.theta # calculates the penalty term of the cost
        # function (L2 regularization) and normalizes it to the dataset size

        # updating the model parameters
        self.theta = self.theta - gradient - penalization_term
        self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Fit the model to the dataset
        :param dataset: The dataset to fit the model to
        :return: The fitted model
        """

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)  # vector shape corresponds to number of feature number
        # initialized with zero and as such the weight of each feature is zero
        self.theta_zero = 0

        # cost history
        self.cost_history = {}

        # check if gradient descent algorithm of choice is valid
        options = get_args(alpha_options)
        assert self.alpha_type in options, f"'{self.alpha_type}' is not in {options}"

        # gradient descent
        for i in range(self.max_iter): # iterates over the dataset for the maximum number of iterations

            self.gradient_descent(dataset)

            # computes the cost function
            self.cost_history[i] = self.cost(dataset)

            # condition to stop gradient descent when cost function value doesn't change
            threshold = 1

            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < threshold:

                if self.alpha_type == 'half_alpha':
                    # change alpha value to half
                    self.alpha = self.alpha / 2

                if self.alpha_type == 'static_alpha':
                    break

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        :param dataset: The dataset to predict the output of the dataset
        :return: The predictions of the dataset
        """
        return np.dot(dataset.x, self.theta) + self.theta_zero # predicted values are obtained by calculating the
        # feature values by the parameter coefficients

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        :param dataset: The dataset to compute the MSE on
        :return: The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        :param dataset: The dataset to compute the cost function on
        :return: The cost function of the model
        """
        y_pred = self.predict(dataset)

        cost_function = (np.sum((y_pred - dataset.y) ** 2) +
                         (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

        return cost_function

    # adds a new variable that is different from the data, the penalization term, that avoids the overfitting of the
    # data

    def cost_function_plot(self):
        """
        Plots the cost function history of the model
        :return: None
        """
        import matplotlib.pyplot as plt

        iter = list(self.cost_history.keys())
        val = list(self.cost_history.values())

        plt.plot(iter, val, '-r')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
