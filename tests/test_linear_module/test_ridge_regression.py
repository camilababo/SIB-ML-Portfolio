import unittest

import numpy as np

from si.data.dataset import Dataset
from si.linear_module.ridge_regression import RidgeRegression


class testRidgeRegression(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        self.y = np.dot(self.x, np.array([1, 2])) + 3
        self.dataset = Dataset(x=self.x, y=self.y)

    def test_RidgeRegression(self):
        self.selector = RidgeRegression()
        self.selector.fit(self.dataset)
        self.pred_df = Dataset(x=np.array([[3, 5]]))

        self.assertEqual(self.selector.theta[0], 1.607736193796488)
        self.assertEqual(self.selector.score(self.dataset), 0.1430479549981516)
        self.assertEqual(self.selector.cost(self.dataset), 1.0609874726416768)
        self.assertEqual(self.selector.predict(self.pred_df)[0], 17.106418656725456)
