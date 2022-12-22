import unittest

import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.layers import Dense, SigmoidActivation
from si.neural_networks.nn import NN, Backpropagation


class testNN(unittest.TestCase):
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
        # model
        self.l1_activation = SigmoidActivation()
        self.l2_activation = SigmoidActivation()
        self.layers = [self.l1, self.l1_activation, self.l2, self.l2_activation]
        self.model = NN(self.layers)

    def test_fit(self):
        # forward pass
        x = self.l1.forward(self.x)
        self.assertEqual(x.tolist(), [[-30, 10], [-10, -10], [-10, -10], [10, -30]])
        x = self.l2.forward(x)
        self.assertEqual(x.tolist(), [[-410], [-410], [-410], [-410]])

    def test_predict(self):
        # predict
        x = self.model.predict(self.dataset)
        self.assertEqual(x.tolist(), [[0.9999545608951235],
                                      [4.548037850511231e-05],
                                      [4.548037850511231e-05],
                                      [0.9999545608951235]])
