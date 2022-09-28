from typing import Type, Union

import numpy as np
import pandas as pd
from si.data.dataset import Dataset


def read_csv(filename=Type[str], sep: str = ',', features: bool = None, label: Union[None, int] = None):
    """
    Method that reads a csv file.

    :type filename: object
    :param filename: Name or directory of the csv file.
    :param sep: The value that is used to separate the data.
    :param features: Boolean value that indicates if the dataset has a defined header.
    :param label: Boolean value that indicates if the dataset has defined labels.
    :return: Dataset.
    """
    dataset = pd.read_csv(filename, sep)
    data = dataset.values.tolist()
    header_row = list(dataset.columns.values)
    y_lab = header_row[label]

    if features:
        if label is not None:
            del header_row[label]
    else:
        header_row = None

    if label is not None:
        y = dataset.iloc[1:, label].tolist()
        dataset = dataset.drop(dataset.columns[label], axis=1)
        data = dataset.values.tolist()
    else:
        y = None

    return Dataset(data, y, header_row, y_lab)


if __name__ == '__main__':
    file_dir = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
    df = read_csv(file_dir, features=True, label=4)
    # print(df.summary())
