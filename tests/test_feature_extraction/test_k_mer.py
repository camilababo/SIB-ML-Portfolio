import unittest

from sklearn.preprocessing import StandardScaler

from si.io.csv import read_csv
from si.linear_module.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split
from si.feature_extraction.k_mer import KMer


class TestKMer(unittest.TestCase):
    def setUp(self) -> None:
        self.dna_path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\tfbs.csv"
        self.pep_path = r"C:\Users\anaca\Documents\GitHub\SIB-ML-Portfolio\datasets\transporters.csv"
        self.dna_dataset = read_csv(self.dna_path, label=True, features=True)
        self.pep_dataset = read_csv(self.pep_path, label=True, features=True)

    def test_k_mer_dna(self):
        kmer_dna = KMer(3, alphabet='dna')
        self.dna_dataset = kmer_dna.fit_transform(self.dna_dataset)
        self.dna_dataset.x = StandardScaler().fit_transform(self.dna_dataset.x)
        self.dna_train, self.dna_test = train_test_split(self.dna_dataset, random_state=2020)
        self.lg_dna = LogisticRegression(max_iter=2000)
        self.lg_dna.fit(self.dna_train)

        self.assertEqual(self.dna_dataset.x.shape, (2000, 64))
        self.assertEqual(self.dna_dataset.x[0, 0], -0.6535005656083097)
        self.assertEqual(self.dna_dataset.x[2, 4], 2.8291375982139857)
        self.assertEqual(self.dna_dataset.x[3, 6], 0.28575478671183796)
        self.assertEqual(self.lg_dna.score(self.dna_test), 0.9566666666666667)

    def test_k_mer_pep(self):
        kmer_pep = KMer(3, alphabet='peptide')
        self.pep_dataset = kmer_pep.fit_transform(self.pep_dataset)
        self.pep_dataset.x = StandardScaler().fit_transform(self.pep_dataset.x)
        self.pep_train, self.pep_test = train_test_split(self.pep_dataset, random_state=2020)
        self.lg_pep = LogisticRegression(max_iter=2000)
        self.lg_pep.fit(self.pep_train)

        self.assertEqual(self.pep_dataset.x.shape, (2011, 8000))
        self.assertEqual(self.pep_dataset.x[0, 0], 0.5679964329913086)
        self.assertEqual(self.pep_dataset.x[2, 4], -0.2934246436397387)
        self.assertEqual(self.pep_dataset.x[3, 6], -0.19667342249400901)
        self.assertEqual(self.lg_pep.score(self.pep_test), 0.7844112769485904)
