import unittest
import lightmotif


class TestMA0045(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.ma0045 = lightmotif.CountMatrix("ACTG", {
            "A": [3, 7, 9, 3, 11, 11, 11, 3, 4, 3, 8, 8, 9, 9, 11, 2],
            "C": [5, 0, 1, 6, 0, 0, 0, 3, 1, 4, 5, 1, 0, 5, 0, 7],
            "T": [2, 4, 3, 1, 0, 1, 1, 6, 1, 1, 0, 1, 3, 0, 0, 5],
            "G": [4, 3, 1, 4, 3, 2, 2, 2, 8, 6, 1, 4, 2, 0, 3, 0],
            "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }).normalize(pseudocount=0.25).log_odds()

    def test_pvalue(self):
        self.assertAlmostEqual(self.ma0045.pvalue(8.7708), 0.00032910, places=5)

    def test_score(self):
        self.assertAlmostEqual(self.ma0045.score(0.00033), 8.756855, places=5)