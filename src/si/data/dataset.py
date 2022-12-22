import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import isnull


class Dataset:
    """
    Class that represents a dataset.

    Attributes
    ----------
    x : ndarray
    y: ndarray
    features_names: list
    label_name: str

    """

    def __init__(self, x: ndarray, y: ndarray = None, features_names: list = None, label_name: str = None):
        """
        Initializes the dataset.
        :param x: Values of the features.
        :param y: Samples.
        :param features: Names of the features.
        :param label: Name of the label.
        """
        if x is None:
            raise ValueError("x cannot be None")

        if features_names is None:
            features_names = [str(i) for i in range(x.shape[1])]
        else:
            features_names = list(features_names)

        if y is not None and label_name is None:
            label_name = "y"

        self.x = x
        self.y = y
        self.features_names = features_names
        self.label_name = label_name

    def shape(self):
        """
        Returns the shape of the dataset.
        :return: Tuple
        """
        return self.x.shape

    def has_label(self):
        """
        Checks if the dataset has a label.
        :return: Boolean
        """
        if self.y is not None:
            return True

        return False

    def get_classes(self):
        """
        Returns the classes of the dataset.
        :return: ndarray
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")

        return np.unique(self.y)

    def get_mean(self):
        """
        Returns the mean of the dataset for each feature.
        :return: ndarray
        """
        if self.x is None:
            return

        return np.mean(self.x, axis=0)

    def get_variance(self):
        """
        Returns the variance of the dataset for each feature.
        :return: ndarray
        """
        if self.x is None:
            return

        return np.nanvar(self.x, axis=0)

    def get_median(self):
        """
        Returns the median of the dataset for each feature.
        :return: ndarray
        """

        if self.x is None:
            return

        return np.nanmedian(self.x, axis=0)

    def get_min(self):
        """
        Returns the minimum value of the dataset for each feature.
        :return: ndarray
        """

        if self.x is None:
            return

        return np.nanmin(self.x, axis=0)

    def get_max(self):
        """
        Returns the maximum value of the dataset for each feature.
        :return: ndarray
        """

        if self.x is None:
            return

        return np.nanmax(self.x, axis=0)

    def summary(self):
        """
        Prints a summary of the dataset with the mean, variance, median, min and max for each feature.
        :return: DataFrame
        """

        return pd.DataFrame(
            {'mean': self.get_mean(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max(),
             'var': self.get_variance()}
        )

    def remove_nan(self):
        """
        Removes rows with nan values without numpy functions.
        :return: Dataframe
        """

        if self.x is None:
            return

        if self.has_label():  # if the dataset has a label
            self.y = self.y[~pd.isnull(self.x).any(axis=1)]  # remove rows with nan values from the label

        self.x = self.x[~pd.isnull(self.x).any(axis=1)]  # remove rows with nan values from the features

        return Dataset(self.x, self.y, self.features_names, self.label_name)

    def replace_nan(self, value):
        """
        Replaces nan values with a given value.

        :param value: Value that is going to replace the nan values.
        :return: DataFrame
        """

        if self.x is None:
            return

        self.x = np.where(isnull(self.x), value, self.x)  # replace nan values from the features

        return Dataset(self.x, self.y, self.features_names, self.label_name)

    def print_dataframe(self):
        """
        Prints the dataset as a dataframe.
        :return: DataFrame
        """
        if self.x is None:
            return

        return pd.DataFrame(self.x, columns=self.features_names, index=self.y)


if __name__ == '__main__':
    # x = np.array([[1, 2, 3], [1, 2, 3]])
    # y = np.array([1, 2])
    # features = ["A", "B", "C"]
    # label = "y"
    # dataset = Dataset(x=x, y=y, features_names=features, label_name=label)

    # print(dataset.shape())
    # print(dataset.has_label())
    # print(dataset.get_classes())
    # print(dataset.get_mean())
    # print(dataset.get_variance())
    # print(dataset.get_median())
    # print(dataset.get_min())
    # print(dataset.get_max())
    # print(dataset.summary())

    x = np.array([[1, 2, 3], [1, 2, 3], [1, None, 3], [1, 2, 3], [None, 2, 3]])
    y = np.array([1, 2, 4, 4, 5])
    features = ["A", "B", "C"]
    label = "y"
    dataset = Dataset(x=x, y=y, features_names=features, label_name=label)

    print(dataset.print_dataframe())
    # print(dataset.remove_nan().print_dataframe())
    print(dataset.replace_nan(16).print_dataframe())

