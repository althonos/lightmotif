import gzip
import os
import tempfile
import unittest
import sys

import lightmotif

class TestEncodedSequence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s1 = lightmotif.EncodedSequence("ATGC")
        cls.s2 = lightmotif.EncodedSequence("ATGCTTAGATAC")

    def test_len(self):
        self.assertEqual(len(self.s1), 4)
        self.assertEqual(len(self.s2), 12)

    def test_index(self):
        A, C, T, G, N = range(5)
        self.assertEqual(self.s1[0], A)
        self.assertEqual(self.s1[1], T)
        self.assertEqual(self.s1[2], G)
        self.assertEqual(self.s1[3], C)
        self.assertEqual(self.s2[0], A)
        self.assertEqual(self.s2[1], T)
        self.assertEqual(self.s2[2], G)
        self.assertEqual(self.s2[3], C)
        self.assertEqual(self.s2[4], T)
        self.assertEqual(self.s2[5], T)
        self.assertEqual(self.s2[6], A)

    def test_index_error(self):
        with self.assertRaises(IndexError):
            _ = self.s1[10]

    @unittest.skipIf(sys.implementation.name != "cpython", "buffer protocol unsupported")
    def test_memoryview(self):
        A, C, T, G, N = range(5)
        mem = memoryview(self.s1)
        self.assertEqual(len(mem), 4)
        self.assertEqual(mem.shape[0], 4)
        self.assertEqual(mem[0], A)
        self.assertEqual(mem[1], T)
        self.assertEqual(mem[2], G)
        self.assertEqual(mem[3], C)

    def test_iter(self):
        A, C, T, G, N = range(5)
        l = list(self.s1)
        self.assertEqual(l, [A, T, G, C])


class TestStripedSequence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s1 = lightmotif.stripe("ATGC")
        cls.s2 = lightmotif.stripe("ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG")

    @unittest.skipIf(sys.implementation.name != "cpython", "buffer protocol unsupported")
    def test_memoryview(self):
        A, C, T, G, N = range(5)
        mem = memoryview(self.s1)
        self.assertEqual(mem[0, 0], A)
        self.assertEqual(mem[1, 0], T)
        self.assertEqual(mem[2, 0], G)
        self.assertEqual(mem[3, 0], C)
    