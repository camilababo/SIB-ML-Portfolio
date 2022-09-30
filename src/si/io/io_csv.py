from typing import Type, Union

import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: object = Type[str], sep: str = ',', features: bool = None, label: Union[None, int] = None) \
        -> object:
    """
    Reads csv files.

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


def write_csv(dataset, filename=Type[str], sep: str = ',', features: bool = None, label: Union[None, int] = None):
    """
    Writes a csv file.

    :param dataset: The dataset that is going to be written.
    :param filename: Name or directory of the csv file.
    :param sep: The value that is used to separate the data.
    :param features: Boolean value that indicates if the dataset has a defined header.
    :param label: Boolean value that indicates if the dataset has defined labels.
    :return: A csv file with the dataset.
    """

    if label is not None:
        dataset = pd.concat([dataset.x, dataset.y], axis=1)
    else:
        dataset = dataset.x

    dataset.to_csv(filename, sep=sep, header=features, index=False)


if __name__ == '__main__':
    file_dir = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
    df = read_csv(file_dir, features=True, label=4)
    print(df.print_dataframe())
    # write_csv(df, "data_out.csv", features=True, label=4)
