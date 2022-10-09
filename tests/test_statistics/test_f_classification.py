import unittest

import numpy as np
from si.statistics.f_classification import f_classification
from si.io.csv import read_csv


class TestFClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.df = read_csv(self.iris_dataset_path, sep=",", label=True, features=True)

    def test_f_classification(self):
        f_value, p_value = f_classification(self.df)

        self.assertAlmostEqual(f_value[0], 119.2645, places=4)
        self.assertAlmostEqual(p_value[0], 1.67e-31)
