import numpy as np
import pandas as pd
from numpy import ndarray


class Dataset:
    def __init__(self, x: ndarray, y: ndarray = None, features: list = None, label: str = None):
        """
        Initializes the dataset.
        :param x: Data.
        :param y: Samples.
        :param features: Names of the features.
        :param label: Name of the label.
        """
        self.x = x
        self.y = y
        self.feaures = features
        self.label = label

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
        :return: List
        """
        if self.y is None:
            return

        return np.unique(self.y)

    def get_mean(self):
        """
        Returns the mean of the dataset for each feature.
        :return: List
        """
        if self.x is None:
            return

        return np.mean(self.x, axis=0)

    def get_variance(self):
        """
        Returns the variance of the dataset for each feature.
        :return: List
        """
        if self.x is None:
            return

        return np.var(self.x, axis=0)

    def get_median(self):
        """
        Returns the median of the dataset for each feature.
        :return: List
        """

        if self.x is None:
            return

        return np.median(self.x, axis=0)

    def get_min(self):
        """
        Returns the minimum value of the dataset for each feature.
        :return: List
        """

        if self.x is None:
            return

        return np.min(self.x, axis=0)

    def get_max(self):
        """
        Returns the maximum value of the dataset for each feature.
        :return: List
        """

        if self.x is None:
            return

        return np.max(self.x, axis=0)

    def summary(self):
        """
        Prints a summary of the dataset with the mean, variance, median, min and max for each feature.
        :return: DataFrame
        """

        return pd.DataFrame(
            {'mean': self.get_mean(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()}
        )

    def remove_nan(self):
        """
        Removes rows with nan values.

        :return: Dataset
        """

        if self.x is None:
            return

        return pd.DataFrame(self.x).dropna(axis=0)

    def replace_nan(self, value):
        """
        Replaces nan values with a given value.

        :param value: Value that is going to replace the nan values.
        :return: Dataset
        """

        if self.x is None:
            return

        return pd.DataFrame(self.x).fillna(value)

    def print_dataframe(self):
        """
        Prints the dataset as a dataframe.
        :return: DataFrame
        """
        if self.x is None:
            return

        return pd.DataFrame(self.x, columns=self.feaures, index=self.y)


if __name__ == '__main__':
    # x = np.array([[1, 2, 3], [1, 2, 3]])
    # y = np.array([1, 2])
    # features = ["A", "B", "C"]
    # label = "y"
    # dataset = Dataset(x=x, y=y, features=features, label=label)

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
    dataset = Dataset(x=x, y=y, features=features, label=label)

    print(dataset.print_dataframe())
    print(dataset.remove_nan())
    print(dataset.replace_nan(16))
