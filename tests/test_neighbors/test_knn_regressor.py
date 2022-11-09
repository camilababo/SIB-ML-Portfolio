import unittest

from si.io.csv import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_regressor import KNNRegressor
from si.statistics.euclidean_distance import euclidean_distance


class testKNNRegressor(unittest.TestCase):
    def setUp(self) -> None:
        self.df_path = r'C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\cpu.csv'
        self.df = read_csv(self.df_path, sep=",", label=True, features=True)
        self.train, self.test = train_test_split(self.df)

    def test_KNNClassifier(self):
        self.selector = KNNRegressor(k=3, distance=euclidean_distance)
        self.selector.fit(self.train)

        self.assertAlmostEqual(self.selector.score(self.test), 1329.6815, places=4)
