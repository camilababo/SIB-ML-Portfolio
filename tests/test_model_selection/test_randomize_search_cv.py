import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from si.io.data_file import read_data_file
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.randomize_search_cv import randomized_search_cv
from si.model_selection.split import train_test_split


class TestCrossValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data"
        self.dataset = read_data_file(self.path, sep=",", label=True)
        self.dataset.x = StandardScaler().fit_transform(self.dataset.x)
        self.train, self.test = train_test_split(self.dataset, random_state=2020)

    def test_cross_validate(self):
        self.lg = LogisticRegression(max_iter=2000)
        self.param = {'l2_penalty': np.linspace(1, 10, 10),
                      'alpha': np.linspace(0.001, 0.0001, 100),
                      'max_iter': np.linspace(1000, 2000, 200)}
        self.random_search = randomized_search_cv(self.lg, self.dataset, self.param, cv=3)

        # check if dictionary has three keys
        self.assertEqual(len(self.random_search.keys()), 4)
        # check if key values are lists
        self.assertIsInstance(self.random_search['seed'], list)
        # self.assertEqual(self.cv.keys('train')[0], 1.2)
        # check if list values are the correct instances
        self.assertIsInstance(self.random_search['seed'][0], int)
        self.assertIsInstance(self.random_search['train'][0], list)
        # check if values are the same
        self.assertEqual(self.random_search['train'][0][0], 0.9653061224489796)
