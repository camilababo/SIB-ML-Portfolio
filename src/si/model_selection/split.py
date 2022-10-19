from random import random
from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.3, random_state: int = 0) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets.

    :param dataset: The dataset to split.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The proportion of the dataset to include in the test split.
    :returns: Tuple[Dataset, Dataset]: The training and test splits.
    """
    # Set the random seed
    np.random.seed(random_state)

    # test set size
    split_div = int(dataset.shape()[0] * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(split_div)

    # get the test and train sets
    test_idx = permutations[:split_div]
    train_idx = permutations[split_div:]

    # get the training and testing datasets
    train = Dataset(dataset.x[train_idx], dataset.y[train_idx], features_names=dataset.features[train_idx],
                    label_name=dataset.label_name[train_idx])

    test = Dataset(dataset.x[test_idx], dataset.y[test_idx], features_names=dataset.features[test_idx],
                   label_name=dataset.label_name[test_idx])

    return train, test
