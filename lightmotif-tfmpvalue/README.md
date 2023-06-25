# 🎼🧬 `lightmotif-tfmpvalue` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*A Rust port of the [TFMPvalue](https://bioinfo.lifl.fr/TFM/TFMpvalue/) algorithm for the [`lightmotif`](https://crates.io/crates/lightmotif) crate.*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/rust.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/lightmotif/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Crate](https://img.shields.io/crates/v/lightmotif-tfmpvalue.svg?maxAge=600&style=flat-square)](https://crates.io/crates/lightmotif-tfmpvalue)
[![Docs](https://img.shields.io/docsrs/lightmotif-tfmpvalue?maxAge=600&style=flat-square)](https://docs.rs/lightmotif-tfmpvalue)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/tree/main/lightmotif-tfmpvalue)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)

## 🗺️ Overview

**TFMPvalue** is an algorithm proposed by Touzet & Varré[\[1\]](#ref1) for
computing a [*p-value*](https://en.wikipedia.org/wiki/P-value) from a score
obtained with a position weight matrix.
It uses discretization to compute an approximation of the score distribution
for the position weight matrix, iterating with growing levels of accuracy
until convergence is reached. This approach outperforms
[dynamic-programming](https://en.wikipedia.org/wiki/Dynamic_programming)
based methods such as **LazyDistrib** by Beckstette *et al.*[\[2\]](#ref2).

`lightmotif-tfmpvalue` provides an implementation of the **TFMPvalue** algorithm
to use with position weight matrices from the `lightmotif` crate.

## 💡 Example

Use `lightmotif` to create a position specific scoring matrix, and then use
the TFMPvalue algorithm to compute the exact P-value for a given score, or
a score threshold for a given P-value:

```rust
extern crate lightmotif;
extern crate lightmotif_tfmpvalue;

use lightmotif::pwm::CountMatrix;
use lightmotif::abc::Dna;
use lightmotif::seq::EncodedSequence;
use lightmotif_tfmpvalue::TfmPvalue;

// Use a ScoringMatrix from `lightmotif`
let pssm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap()
    .to_freq(0.25)
    .to_scoring(None);

// Initialize the TFMPvalue algorithm for the given PSSM
// (the `pssm` reference must outlive `tfmp`).
let mut tfmp = TfmPvalue::new(&pssm);

// Compute the exact p-value for a given score
let pvalue = tfmp.pvalue(19.3);
assert_eq!(pvalue, 1.4901161193847656e-08);

// Compute the exact score for a given p-value
let score = tfmp.score(pvalue);
assert_eq!(score, 19.3);
```

*Note that in the example above, the computation is not bounded, so for certain
particular matrices the algorithm may require a large amount of memory to
converge. Use the `TfmPvalue::approximate_pvalue` and `TfmPvalue::approximate_score`
methods to obtain an iterator over the algorithm iterations, allowing you to stop at
any given time based on external criterion such as total memory usage.*


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
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
The original TFMPvalue implementation was written by the [BONSAI team](https://bioinfo.univ-lille.fr/)
of [CRISTaL](https://www.cristal.univ-lille.fr/), [Université de Lille](http://www.univ-lille.fr/)
and [is available](https://bioinfo.univ-lille.fr/tfm-pvalue/tfm-pvalue.php)
under the terms of the [GNU General Public License v2.0](https://choosealicense.com/licenses/gpl-2.0/).

*This project is in no way not affiliated, sponsored, or otherwise endorsed
by the [original TFMPvalue authors](https://bioinfo.univ-lille.fr/). It was
developed by [Martin Larralde](https://github.com/althonos/) during his PhD
project at the [European Molecular Biology Laboratory](https://www.embl.de/)
in the [Zeller team](https://github.com/zellerlab).*

## 📚 References

- <a id="ref1">\[1\]</a> Touzet, Hélène and Jean-Stéphane Varré. ‘Efficient and accurate P-value computation for Position Weight Matrices’. Algorithms for Molecular Biology 2, 1–12 (2007). [doi:10.1186/1748-7188-2-15](https://doi.org/10.1186/1748-7188-2-15).
- <a id="ref2">\[2\]</a> Beckstette, Michael, Robert Homann, and Robert Giegerich. ‘Fast index based algorithms and software for matching position specific scoring matrices’. BMC Bioinformatics 7, 389 (2006). [doi:10.1186/1471-2105-7-389](https://doi.org/10.1186/1471-2105-7-389).

