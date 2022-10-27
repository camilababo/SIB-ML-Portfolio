import unittest

from si.io.data_file import read_data_file
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split


class testLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.df_path = r'C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data'
        self.df = read_data_file(self.df_path, sep=",", label=True)
        self.train, self.test = train_test_split(self.df)

    def test_LogisticRegression(self):
        self.selector = LogisticRegression()
        self.selector.fit(self.train)

        self.assertEqual(self.selector.theta[0], -0.1579690175924113)
        self.assertEqual(self.selector.score(self.test), 0.9090909090909091)
        self.assertEqual(self.selector.cost(self.test), 0.4293555111797384)
        self.assertEqual(self.selector.predict(self.test)[0], 0.0)