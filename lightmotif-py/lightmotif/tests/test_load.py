import unittest
import io
import os
import textwrap
import tempfile
import pathlib

import lightmotif


class _TestLoad():
    
    def test_load_file(self):
        text = textwrap.dedent(self.text).encode()
        buffer = io.BytesIO(text)
        motifs = list(lightmotif.load(buffer, self.format))
        self.assertEqual(len(motifs), self.length)
        self.assertEqual(motifs[0].name, self.first)

    def test_load_filename(self):
        text = textwrap.dedent(self.text).encode()
        with tempfile.NamedTemporaryFile("r+b") as f:
            f.write(text)
            f.flush()
            motifs = list(lightmotif.load(f.name, self.format))
        self.assertEqual(len(motifs), self.length)
        self.assertEqual(motifs[0].name, self.first)

    def test_load_filename_bytes(self):
        text = textwrap.dedent(self.text).encode()
        with tempfile.NamedTemporaryFile("r+b") as f:
            f.write(text)
            f.flush()
            motifs = list(lightmotif.load(os.fsencode(f.name), self.format))
        self.assertEqual(len(motifs), self.length)
        self.assertEqual(motifs[0].name, self.first)

    def test_load_path(self):
        text = textwrap.dedent(self.text).encode()
        with tempfile.NamedTemporaryFile("r+b") as f:
            f.write(text)
            f.flush()
            path = pathlib.Path(f.name)
            motifs = list(lightmotif.load(path, self.format))
        self.assertEqual(len(motifs), self.length)
        self.assertEqual(motifs[0].name, self.first)


class TestJASPAR(_TestLoad, unittest.TestCase):
    format = "jaspar"
    length = 1
    first = "MA0001.3"
    text = """
    >MA0001.3	AGL3
    0      0     82     40     56     35     65     25     64      0
    92    79      1      4      0      0      1      4      0      0
    0      0      2      3      1      0      4      3     28     92
    3     16     10     48     38     60     25     63      3      3
    """

class TestJASPAR16(_TestLoad, unittest.TestCase):
    format = "jaspar16"
    length = 2
    first = "MA0001.3"
    text = """
    >MA0001.3	AGL3
    A  [     0      0     82     40     56     35     65     25     64      0 ]
    C  [    92     79      1      4      0      0      1      4      0      0 ]
    G  [     0      0      2      3      1      0      4      3     28     92 ]
    T  [     3     16     10     48     38     60     25     63      3      3 ]
    >MA0017.3	NR2F1
    A  [  7266   6333   8496      0      0      0      0  12059   5116   3229   3276   5681 ]
    C  [  1692    387      0     30      0      0  12059      0   3055   2966   2470   2912 ]
    G  [  1153   4869   3791  12059  12059      0      0     91   1618   4395   3886  36863 ]
    T  [  1948    469      0      0      0  12059      0      0   2270   1469   2427   3466 ]
    """

class TestTRANSFAC(_TestLoad, unittest.TestCase):
    format = "transfac"
    length = 1
    first = "M00005"
    text = """
    NA  M00005
    P0      A      C      G      T
    01      3      0      0      2      W
    02      1      1      3      0      G
    03      3      1      1      0      A
    04      2      1      2      0      R
    05      1      2      0      2      Y
    06      0      5      0      0      C
    07      5      0      0      0      A
    08      0      0      5      0      G
    09      0      5      0      0      C
    10      0      0      1      4      T
    11      0      1      4      0      G
    12      0      2      1      2      Y
    13      1      0      3      1      G
    14      0      0      5      0      G
    15      1      1      1      2      N
    16      1      4      0      0      C
    17      2      1      1      1      N
    18      0      0      3      2      K
    //
    """.lstrip("\n")