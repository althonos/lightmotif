# 🎼🧬 `lightmotif` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*A lightweight [platform-accelerated](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) library for [biological motif](https://en.wikipedia.org/wiki/Sequence_motif) scanning using [position weight matrices](https://en.wikipedia.org/wiki/Position_weight_matrix)*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/python.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/lightmotif/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Docs](https://img.shields.io/readthedocs/lightmotif/latest?style=flat-square&maxAge=600)](https://lightmotif.readthedocs.io)
[![Crate](https://img.shields.io/crates/v/lightmotif-py.svg?maxAge=600&style=flat-square)](https://crates.io/crates/lightmotif-py)
[![PyPI](https://img.shields.io/pypi/v/lightmotif.svg?style=flat-square&maxAge=600)](https://pypi.org/project/lightmotif)
[![Wheel](https://img.shields.io/pypi/wheel/lightmotif.svg?style=flat-square&maxAge=2678400)](https://pypi.org/project/lightmotif/#files)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/lightmotif?style=flat-square&maxAge=3600)](https://anaconda.org/bioconda/lightmotif)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightmotif.svg?style=flat-square&maxAge=600)](https://pypi.org/project/lightmotif/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/lightmotif.svg?style=flat-square&maxAge=600)](https://pypi.org/project/lightmotif/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/tree/main/lightmotif-py)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/lightmotif?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/lightmotif)

## 🗺️ Overview

[Motif](https://en.wikipedia.org/wiki/Sequence_motif) scanning with 
[position weight matrices](https://en.wikipedia.org/wiki/Position_weight_matrix)
(also known as position-specific scoring matrices) is a robust method for 
identifying motifs of fixed length inside a 
[biological sequence](https://en.wikipedia.org/wiki/Sequence_(biology)). They can be 
used to identify [transcription factor](https://en.wikipedia.org/wiki/Transcription_factor) 
[binding sites in DNA](https://en.wikipedia.org/wiki/DNA_binding_site), 
or [protease](https://en.wikipedia.org/wiki/Protease) [cleavage](https://en.wikipedia.org/wiki/Proteolysis) site in [polypeptides](https://en.wikipedia.org/wiki/Proteolysis). 
Position weight matrices are often viewed as [sequence logos](https://en.wikipedia.org/wiki/Sequence_logo):

[![MX000274.svg](https://raw.githubusercontent.com/althonos/lightmotif/main/docs/_static/prodoric_logo_mx000274.svg)](https://www.prodoric.de/matrix/MX000274.html)

The `lightmotif` library provides a Python module to run very efficient
searches for a motif encoded in a position weight matrix. The position
scanning combines several techniques to allow high-throughput processing
of sequences:

- Compile-time definition of alphabets and matrix dimensions.
- Sequence symbol encoding for fast table look-ups, as implemented in
  HMMER[\[1\]](#ref1) or MEME[\[2\]](#ref2)
- Striped sequence matrices to process several positions in parallel,
  inspired by Michael Farrar[\[3\]](#ref3).
- Vectorized matrix row look-up using `permute` instructions of [AVX2](https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions).

*This is the Python version, there is a [Rust crate](https://crates.io/crates/lightmotif) available as well.*

## 🔧 Installing

`lightmotif` can be installed directly from [PyPI](https://pypi.org/project/lightmotif/),
which hosts some pre-built wheels for most mainstream platforms, as well as the 
code required to compile from source with Rust:
```console
$ pip install lightmotif
```
<!-- Otherwise, lightmotif is also available as a [Bioconda](https://anaconda.org/bioconda/lightmotif)
package:
```console
$ conda install -c bioconda lightmotif
``` -->

In the event you have to compile the package from source, all the required
Rust libraries are vendored in the source distribution, and a Rust compiler
will be setup automatically if there is none on the host machine.


## 💡 Example

The motif interface should be mostly compatible with the 
[`Bio.motifs`](https://biopython-tutorial.readthedocs.io/en/latest/notebooks/14%20-%20Sequence%20motif%20analysis%20using%20Bio.motifs.html#)
module from [Biopython](https://biopython.org/). The notable difference is that 
the `calculate` method of PSSM objects expects a *striped* sequence instead.

```python
import lightmotif

# Create a count matrix from an iterable of sequences
motif = lightmotif.create(["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"])

# Create a PSSM with 0.1 pseudocounts and uniform background frequencies
pwm = motif.counts.normalize(0.1)
pssm = pwm.log_odds()

# Encode the target sequence into a striped matrix
seq = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG"
striped = lightmotif.stripe(seq)

# Compute scores using the fastest backend implementation for the host machine
scores = pssm.calculate(sseq)
```

## ⏱️ Benchmarks

Benchmarks use the [MX000001](https://www.prodoric.de/matrix/MX000001.html)
motif from [PRODORIC](https://www.prodoric.de/)[\[4\]](#ref4), and the
[complete genome](https://www.ncbi.nlm.nih.gov/nuccore/U00096) of an
*Escherichia coli K12* strain. 
*Benchmarks were run on a [i7-10710U CPU](https://ark.intel.com/content/www/us/en/ark/products/196448/intel-core-i7-10710u-processor-12m-cache-up-to-4-70-ghz.html) running @1.10GHz, compiled with `--target-cpu=native`*.

```console
lightmotif (avx2):      5,479,884 ns/iter    (+/- 3,370,523) = 807.8 MiB/s
Bio.motifs:           334,359,765 ns/iter   (+/- 11,045,456) =  13.2 MiB/s
MOODS.scan:           182,710,624 ns/iter    (+/- 9,459,257) =  24.2 MiB/s
pymemesuite.fimo:     239,694,118 ns/iter    (+/- 7,444,620) =  18.5 MiB/s
```


## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the [GitHub issue
tracker](https://github.com/althonos/lightmotif/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

<!-- ### 🏗️ Contributing

Contributions are more than welcome! See [`CONTRIBUTING.md`](https://github.com/althonos/lightmotif/blob/master/CONTRIBUTING.md) for more details. -->

## 📋 Changelog

This project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html)
and provides a [changelog](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)
in the [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) format.

## ⚖️ License

This library is provided under the open-source
[MIT license](https://choosealicense.com/licenses/mit/).

*This project was developed by [Martin Larralde](https://github.com/althonos/)
during his PhD project at the [European Molecular Biology Laboratory](https://www.embl.de/)
in the [Zeller team](https://github.com/zellerlab).*


## 📚 References

- <a id="ref1">\[1\]</a> Eddy, Sean R. ‘Accelerated Profile HMM Searches’. PLOS Computational Biology 7, no. 10 (20 October 2011): e1002195. [doi:10.1371/journal.pcbi.1002195](https://doi.org/10.1371/journal.pcbi.1002195).
- <a id="ref2">\[2\]</a> Grant, Charles E., Timothy L. Bailey, and William Stafford Noble. ‘FIMO: Scanning for Occurrences of a given Motif’. Bioinformatics 27, no. 7 (1 April 2011): 1017–18. [doi:10.1093/bioinformatics/btr064](https://doi.org/10.1093/bioinformatics/btr064).
- <a id="ref3">\[3\]</a> Farrar, Michael. ‘Striped Smith–Waterman Speeds Database Searches Six Times over Other SIMD Implementations’. Bioinformatics 23, no. 2 (15 January 2007): 156–61. [doi:10.1093/bioinformatics/btl582](https://doi.org/10.1093/bioinformatics/btl582).
- <a id="ref4">\[4\]</a> Dudek, Christian-Alexander, and Dieter Jahn. ‘PRODORIC: State-of-the-Art Database of Prokaryotic Gene Regulation’. Nucleic Acids Research 50, no. D1 (7 January 2022): D295–302. [doi:10.1093/nar/gkab1110](https://doi.org/10.1093/nar/gkab1110).
