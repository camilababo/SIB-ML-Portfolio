import unittest

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split


class TestSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data"
        self.dataset = read_data_file(self.path, sep=",", label=True)

    def test_train_test_split(self):
        self.train, self.test = train_test_split(self.dataset, random_state=2020)

        self.assertEqual(self.train.x.shape, (490, 9))
        self.assertEqual(self.train.y.shape, (490,))
        self.assertEqual(self.test.x.shape, (209, 9))
        self.assertEqual(self.test.y.shape, (209,))
