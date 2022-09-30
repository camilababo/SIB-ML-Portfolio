from typing import Union

import numpy

from si.data.dataset import Dataset


def read_data_file(filename: str, label: Union[None, int] = None, sep: str = ","):
    """
    Reads a data file.
    :param filename: Name or directory of the data file.
    :param label: Boolean value that indicates if the dataset has defined labels or integer value
    that indicates the column.
    :param sep: The value that is used to separate the data.
    :return: Dataset.
    """
    if label is not None:
        data = numpy.genfromtxt(filename, delimiter=sep, usecols=range(label), skip_header=1)
        y = numpy.genfromtxt(filename, delimiter=sep, usecols=label, dtype=None, encoding=None, skip_header=1)

    else:
        data = numpy.genfromtxt(filename, delimiter=sep, usecols=range(label))
        y = None

    return Dataset(data, y)


def write_data_file(dataset: Dataset, filename: str, label: Union[None, int] = None, sep: str = ","):
    """
    Writes a data file.

    :param dataset: The dataset that is going to be written.
    :param filename: Name or directory of the data file that is going to be written.
    :param sep: The value that is used to separate the data.
    :param label: Boolean value that indicates if the dataset has defined labels.
    :return: A data file with the dataset.
    """

    if label is not None:
        dataset = numpy.concatenate((dataset.x, dataset.y), axis=1)
    else:
        dataset = dataset.x

    numpy.savetxt(filename, dataset, delimiter=sep, fmt="%s")


if __name__ == '__main__':
    file_dir = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
    dataset = read_data_file(file_dir, label=4)
    # write_data_file(dataset, "data_file_out.csv", label=4)
    print(dataset.print_dataframe())
