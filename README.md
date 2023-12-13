# 🎼🧬 `lightmotif` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*A lightweight [platform-accelerated](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) library for [biological motif](https://en.wikipedia.org/wiki/Sequence_motif) scanning using [position weight matrices](https://en.wikipedia.org/wiki/Position_weight_matrix)*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/rust.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/lightmotif/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Crate](https://img.shields.io/crates/v/lightmotif.svg?maxAge=600&style=flat-square)](https://crates.io/crates/lightmotif)
[![Docs](https://img.shields.io/docsrs/lightmotif?maxAge=600&style=flat-square)](https://docs.rs/lightmotif)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)

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

The `lightmotif` library provides a Rust crate to run very efficient
searches for a motif encoded in a position weight matrix. The position
scanning combines several techniques to allow high-throughput processing
of sequences:

- Compile-time definition of alphabets and matrix dimensions.
- Sequence symbol encoding for fast table look-ups, as implemented in
  HMMER[\[1\]](#ref1) or MEME[\[2\]](#ref2)
- Striped sequence matrices to process several positions in parallel,
  inspired by Michael Farrar[\[3\]](#ref3).
- Vectorized matrix row look-up using `permute` instructions of [AVX2](https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions).

Other crates from the ecosystem provide additional features if needed:

- [`lightmotif-io`](https://crates.io/crates/lightmotif-io) is a crate with parser implementations for various count matrix, frequency matrix and position-specific scoring matrix formats such as [TRANSFAC](https://en.wikipedia.org/wiki/TRANSFAC) or [JASPAR](https://jaspar.elixir.no/docs/).
- [`lightmotif-tfmpvalue`](https://crates.io/crates/lightmotif-tfmpvalue) is an exact reimplementation of the TFM-PVALUE[\[4\]](#ref4) algorithm for converting between a score and a *p*-value for a given scoring matrix.

*This is the Rust version, there is a [Python package](https://pypi.org/project/lightmotif) available as well.*

## 💡 Example

```rust
use lightmotif::*;
use lightmotif::abc::Nucleotide;
use typenum::U32;

// Create a count matrix from an iterable of motif sequences
let counts = CountMatrix::<Dna>::from_sequences(&[
    EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
    EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
]).unwrap();

// Create a PSSM with 0.1 pseudocounts and uniform background frequencies.
let pssm = counts.to_freq(0.1).to_scoring(None);

/// Create a pipeline to run tasks with platform acceleration
let pli = Pipeline::dispatch();

// Use the pipeline to encode the target sequence into a striped matrix
let seq = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
let encoded = pli.encode(seq).unwrap();
let mut striped = pli.stripe(encoded);

// Use the pipeline to compute scores for every position of the matrix.
striped.configure(&pssm);
let scores = pli.score(&striped, &pssm);

// Scores can be extracted into a Vec<f32>, or indexed directly.
let v = scores.to_vec();
assert_eq!(scores[0], -23.07094);
assert_eq!(v[0], -23.07094);

// The highest scoring position can be searched with a pipeline as well.
let best = pli.argmax(&scores).unwrap();
assert_eq!(best, 18);

```
This example uses a dynamic dispatch pipeline, which selects the best available
backend (AVX2, SSE2, NEON, or a generic implementation) depending on the local 
platform.

## ⏱️ Benchmarks

Both benchmarks use the [MX000001](https://www.prodoric.de/matrix/MX000001.html)
motif from [PRODORIC](https://www.prodoric.de/)[\[5\]](#ref5), and the
[complete genome](https://www.ncbi.nlm.nih.gov/nuccore/U00096) of an
*Escherichia coli K12* strain.
*Benchmarks were run on a [i7-10710U CPU](https://ark.intel.com/content/www/us/en/ark/products/196448/intel-core-i7-10710u-processor-12m-cache-up-to-4-70-ghz.html) running @1.10GHz, compiled with `--target-cpu=native`*.

- Score every position of the genome with the motif weight matrix:
  ```console
  test bench_avx2    ... bench:   4,510,794 ns/iter (+/-     9,570) = 1029 MB/s
  test bench_sse2    ... bench:  26,773,537 ns/iter (+/-    57,891) =  173 MB/s
  test bench_generic ... bench: 317,731,004 ns/iter (+/- 2,567,370) =   14 MB/s
  ```

- Find the highest-scoring position for a motif in a 10kb sequence
  (compared to the PSSM algorithm implemented in
  [`bio::pattern_matching::pssm`](https://docs.rs/bio/1.1.0/bio/pattern_matching/pssm/index.html)):
  ```console
  test bench_avx2    ... bench:      12,797 ns/iter (+/-   380) = 781 MB/s
  test bench_sse2    ... bench:      62,597 ns/iter (+/-    43) = 159 MB/s
  test bench_generic ... bench:     671,900 ns/iter (+/- 1,150) =  14 MB/s
  test bench_bio     ... bench:   1,193,911 ns/iter (+/- 2,519) =   8 MB/s
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
- <a id="ref4">\[4\]</a> Touzet, Hélène, and Jean-Stéphane Varré. ‘Efficient and Accurate P-Value Computation for Position Weight Matrices’. Algorithms for Molecular Biology 2, no. 1 (2007): 1–12. [doi:10.1186/1748-7188-2-15](https://doi.org/10.1186/1748-7188-2-15).
- <a id="ref5">\[5\]</a> Dudek, Christian-Alexander, and Dieter Jahn. ‘PRODORIC: State-of-the-Art Database of Prokaryotic Gene Regulation’. Nucleic Acids Research 50, no. D1 (7 January 2022): D295–302. [doi:10.1093/nar/gkab1110](https://doi.org/10.1093/nar/gkab1110).
