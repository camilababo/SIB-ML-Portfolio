import os
from unittest import TestCase

from si.io.csv import read_csv, write_csv


class TestIOCSV(TestCase):

    def setUp(self) -> None:
        self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"

    def test_read_csv(self):
        dataset = read_csv(self.iris_dataset_path, sep=",", features=True, label=True)

        self.assertTrue(dataset.has_label())
        self.assertAlmostEqual(dataset.get_mean()[0], 5.84333333)
        self.assertAlmostEqual(dataset.get_max()[0], 7.9)
        self.assertAlmostEqual(dataset.get_median()[0], 5.8)
        self.assertEqual(dataset.shape(), (150, 4))
        self.assertEqual(dataset.x[0, 0], 5.1)

    def test_write_csv(self):
        dataset = read_csv(self.iris_dataset_path, sep=",", features=True, label=True)

        write_csv(dataset, "test_write_csv.csv", sep=",", features=True, label=True)

    def tearDown(self) -> None:
        os.remove("test_write_csv.csv")




