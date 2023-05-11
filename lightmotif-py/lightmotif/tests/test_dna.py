import gzip
import os
import tempfile
import unittest

import lightmotif

SEQUENCE = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG"
EXPECTED = [
    -23.07094,
    -18.678621,
    -15.219191,
    -17.745737,
    -18.678621,
    -23.07094,
    -17.745737,
    -19.611507,
    -27.463257,
    -29.989803,
    -14.286304,
    -26.53037,
    -15.219191,
    -10.826873,
    -10.826873,
    -22.138054,
    -38.774437,
    -30.922688,
    -5.50167,
    -24.003826,
    -18.678621,
    -15.219191,
    -35.315006,
    -17.745737,
    -10.826873,
    -30.922688,
    -23.07094,
    -6.4345555,
    -31.855574,
    -23.07094,
    -15.219191,
    -31.855574,
    -8.961102,
    -26.53037,
    -27.463257,
    -14.286304,
    -15.219191,
    -26.53037,
    -23.07094,
    -18.678621,
    -14.286304,
    -18.678621,
    -26.53037,
    -16.152077,
    -17.745737,
    -18.678621,
    -17.745737,
    -14.286304,
    -30.922688,
    -18.678621,
]


class TestDNAMotif(unittest.TestCase):
    def test_calculate(self):
        motif = lightmotif.create(["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"])
        frequencies = motif.counts.normalize(0.1)
        pssm = frequencies.log_odds()

        seq = lightmotif.stripe(SEQUENCE)
        scores = pssm.calculate(seq)
        self.assertEqual(len(scores), len(EXPECTED))
        for x, y in zip(scores, EXPECTED):
            self.assertAlmostEqual(x, y, places=5)
