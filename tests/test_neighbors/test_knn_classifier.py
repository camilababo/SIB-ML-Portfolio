import unittest

from si.io.csv import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_classifier import KNNClassifier
from si.statistics.euclidean_distance import euclidean_distance


class testKNNClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.df_path = r'C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\iris.csv'
        self.df = read_csv(self.df_path, sep=",", label=True, features=True)
        self.train, self.test = train_test_split(self.df)

    def test_KNNClassifier(self):
        self.selector = KNNClassifier(k=3, distance=euclidean_distance)
        self.selector.fit(self.train)

        self.assertEqual(self.selector.score(self.test), 0.6)
