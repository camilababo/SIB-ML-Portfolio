import os
from unittest import TestCase

from si.io.data_file import read_data_file, write_data_file


class TestIODataFile(TestCase):

    def setUp(self) -> None:
        # self.iris_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/iris.csv"
        self.breast_bin_dataset_path = "C:/Users/anaca/Documents/GitHub/SIB-ML-Portfolio/datasets/breast-bin.data"

    def test_read_data_file(self):
        dataset = read_data_file(self.breast_bin_dataset_path, sep=",", label=True)

        self.assertTrue(dataset.has_label())
        self.assertAlmostEqual(dataset.get_mean()[0], 4.41773962)
        self.assertAlmostEqual(dataset.get_max()[0], 10.0)
        self.assertAlmostEqual(dataset.get_median()[0], 4.0)
        self.assertEqual(dataset.shape(), (699, 9))
        self.assertEqual(dataset.x[0, 0], 8.0)

    def test_write_data_file(self):
        dataset = read_data_file(self.breast_bin_dataset_path, sep=",", label=True)

        write_data_file(dataset, "data_file_out.csv", sep=",", label=True)

    def tearDown(self) -> None:
        os.remove("data_file_out.csv")
