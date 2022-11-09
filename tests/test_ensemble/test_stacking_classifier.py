import unittest

from sklearn.preprocessing import StandardScaler

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split
from si.neighbors.knn_classifier import KNNClassifier
from si.statistics.euclidean_distance import euclidean_distance


class TestStackingClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\breast-bin.data"
        self.dataset = read_data_file(self.path, sep=",", label=True)
        self.dataset.x = StandardScaler().fit_transform(self.dataset.x)
        self.train, self.test = train_test_split(self.dataset, random_state=2020)

    def test_stacking_classifier(self):
        self.lg = LogisticRegression(max_iter=2000)
        self.knn = KNNClassifier(k=3, distance=euclidean_distance)
        self.final = KNNClassifier(k=6, distance=euclidean_distance)

        stacking_system = StackingClassifier([self.lg, self.knn], self.final)
        stacking_system.fit(self.train)

        self.assertEqual(stacking_system.score(self.test), 0.9665071770334929)
        self.assertEqual(stacking_system.models[0].predict(self.test)[0], 1.0)
        self.assertEqual(stacking_system.models[1].predict(self.test)[1], 0.0)
        self.assertEqual(stacking_system.final_model.predict(self.test)[0], 1.0)


