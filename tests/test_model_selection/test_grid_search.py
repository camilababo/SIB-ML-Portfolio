import unittest

from sklearn.preprocessing import StandardScaler

from si.io.data_file import read_data_file
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.grid_search import grid_search_cv
from si.model_selection.split import train_test_split


class TestCrossValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data"
        self.dataset = read_data_file(self.path, sep=",", label=True)
        self.dataset.x = StandardScaler().fit_transform(self.dataset.x)
        self.train, self.test = train_test_split(self.dataset, random_state=2020)

    def test_cross_validate(self):
        self.lg = LogisticRegression(max_iter=2000)
        self.param = {'l2_penalty': [1, 10],
                      'alpha': [0.001, 0.0001],
                      'max_iter': [1000, 2000]}
        self.grid_search = grid_search_cv(self.lg, self.dataset, self.param, cv=3)

        # check if list values are dictionaries
        self.assertIsInstance(self.grid_search[0], dict)
        # check if dictionaries in the list have 3 keys
        self.assertEqual(len(self.grid_search[0].keys()), 4)
        # check if dictionary values are the correct instances
        self.assertIsInstance(self.grid_search[0]['seed'][0], int)
        self.assertIsInstance(self.grid_search[0]['train'][0], float)
        # check if values are the same
        self.assertEqual(self.grid_search[0]['train'][0], 0.9653061224489796)
