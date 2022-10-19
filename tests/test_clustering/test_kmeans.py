import unittest

from si.clustering.kmeans import Kmeans
from si.io.csv import read_csv


class TestKMeans(unittest.TestCase):

    def setUp(self) -> None:
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.df = read_csv(self.iris_dataset_path, sep=",", label=True, features=True)

    def test_get_closest_centroids(self):
        selector = Kmeans(k=3)
        selector.fit_transform(self.df)

        self.assertEqual(selector.centroids.shape, (3, 4))
        self.assertEqual(selector.labels.shape, (150,))
        self.assertEqual(selector.predict(self.df).shape, (150,))


