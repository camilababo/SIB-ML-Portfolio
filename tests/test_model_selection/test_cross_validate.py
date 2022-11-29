import unittest

from sklearn.preprocessing import StandardScaler

from si.io.data_file import read_data_file
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.cross_validate import cross_validate
from si.model_selection.split import train_test_split


class TestCrossValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data"
        self.dataset = read_data_file(self.path, sep=",", label=True)
        self.dataset.x = StandardScaler().fit_transform(self.dataset.x)
        self.train, self.test = train_test_split(self.dataset, random_state=2020)

    def test_cross_validate(self):
        self.lg = LogisticRegression(max_iter=2000)
        self.cv = cross_validate(self.lg, self.dataset, cv=5)

        # check if dictionary has three keys
        self.assertEqual(len(self.cv.keys()), 4)
        # check if key values are lists
        self.assertIsInstance(self.cv['seed'], list)
        # self.assertEqual(self.cv.keys('train')[0], 1.2)
        # check if list values are the correct instances
        self.assertIsInstance(self.cv['seed'][0], int)
        self.assertIsInstance(self.cv['train'][0], float)
        # check if the training dataset is different in each fold
        self.assertNotEqual(self.cv['train'][0], self.cv['train'][1])
        # check if values are the same
        self.assertEqual(self.cv['train'][0], 0.9673469387755103)


