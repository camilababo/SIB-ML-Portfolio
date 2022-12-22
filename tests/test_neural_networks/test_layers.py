import unittest

import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.layers import Dense


class testLayers(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0, 1, 1, 0]])
        self.dataset = Dataset(self.x, self.y)
        # layer1:
        self.w1 = np.array([[20, -20], [20, -20]])
        self.b1 = np.array([[-30, 10]])
        self.l1 = Dense(2, 2)
        self.l1.weights = self.w1
        self.l1.bias = self.b1
        # layer2
        self.w2 = np.array([[20], [20]])
        self.b2 = np.array([[-10]])
        self.l2 = Dense(2, 1)
        self.l2.weights = self.w2
        self.l2.bias = self.b2

    def test_forward(self):
        # forward pass
        x = self.l1.forward(self.x)
        self.assertEqual(x.tolist(), [[-30, 10], [-10, -10], [-10, -10], [10, -30]])
        x = self.l2.forward(x)
        self.assertEqual(x.tolist(), [[-410], [-410], [-410], [-410]])
