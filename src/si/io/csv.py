import pandas as pd
from si.data.dataset import Dataset


def read_csv(filename, sep: str, features: bool, label: bool):
    df = pd.read_csv(filename, sep=sep)
    if df.



if __name__ == '__main__':
    file_dir = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\iris.csv"
    read_csv(file_dir, sep=",")
