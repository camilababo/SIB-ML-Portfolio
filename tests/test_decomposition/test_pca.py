import unittest

from si.decomposition.pca import PCA
from si.io.csv import read_csv


class testPCA(unittest.TestCase):
    def setUp(self) -> None:
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.df = read_csv(self.iris_dataset_path, sep=",", label=True, features=True)

    def test_pca(self):
        selector = PCA(n_components=2)
        selector.fit_transform(self.df)

        self.assertAlmostEqual(selector.mean[0], 5.843333333333335)
        self.assertAlmostEqual(selector.components[0][0], 0.361589677381449)
        self.assertAlmostEqual(selector.explained_variance[0], 4.224840768320115)
