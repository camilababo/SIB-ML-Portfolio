import unittest

import numpy as np
from si.feature_selection.variance_threshold import VarianceThreshold
from si.io.csv import read_csv


class testVarianceThreshold(unittest.TestCase):

    def setUp(self):
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.df = read_csv(self.iris_dataset_path, sep=",", label=True, features=True)

    def test_variance_threshold(self):
        selector = VarianceThreshold(threshold=0.5)
        selector.fit_transform(self.df)

        self.assertEqual(selector.variance[0], 0.6811222222222223)
        self.assertEqual(selector.transform(self.df).x.shape, (150, 3))
