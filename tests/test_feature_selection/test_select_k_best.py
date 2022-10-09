import unittest

from si.feature_selection.select_k_best import SelectKBest
from si.io.csv import read_csv
from si.statistics.f_classification import f_classification


class testSelectKBest(unittest.TestCase):
    def setUp(self) -> None:
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.df = read_csv(self.iris_dataset_path, sep=",", label=True, features=True)

    def test_select_k_best(self):
        selector = SelectKBest(k=2, score_func=f_classification)
        selector.fit_transform(self.df)

        self.assertAlmostEqual(selector.f_value[0], 119.2645, places=4)
        self.assertAlmostEqual(selector.p_value[0], 1.67e-31)
        self.assertEqual(selector.transform(self.df).x.shape, (150, 2))
