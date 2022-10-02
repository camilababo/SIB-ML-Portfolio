from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset


class TestData(TestCase):

    def setUp(self) -> None:
        self.x = np.array([[1, 2, 3], [1, 2, 3]])
        self.y = np.array([1, 2])
        self.features_names = ['a', 'b', 'c']
        self.label_name = 'y'
        self.dataset = Dataset(self.x, self.y, self.features_names, self.label_name)

        self.x2 = np.array([[1, 2, 3], [1, 2, 3], [1, None, 3], [1, 2, 3], [None, 2, 3]])
        self.y2 = np.array([1, 2, 4, 4, 5])
        self.dataset2 = Dataset(self.x2, self.y2, self.features_names, self.label_name)

    def test_shape(self):
        self.assertEqual(self.dataset.shape(), (2, 3))

    def test_has_label(self):
        self.assertTrue(self.dataset.has_label())

    def test_remove_nan(self):
        self.assertEqual(self.dataset2.remove_nan().shape, (3, 3))

    def test_replace_nan(self):
        self.assertEqual(self.dataset2.replace_nan(0).shape, (5, 3))

