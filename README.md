# 🎼🧬 `lightmotif` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*A lightweight [platform-accelerated](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) library for [biological motif](https://en.wikipedia.org/wiki/Sequence_motif) scanning using [position weight matrices](https://en.wikipedia.org/wiki/Position_weight_matrix)*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/test.yml?branch=master&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)

## 🗺️ Overview

Motif scanning with position weight matrices (also known as position-specific 
scoring matrices) is a robust method for identifying motifs of fixed length
inside a biological sequence. They can be used to identiy

The `lightmotif` library provides a Rust crate to run very efficient 
searches for a motif encoded in a position weight matrix. The position 
scanning combines several techniques to allow high-throughput processing 
of sequences: 

- Compile-time definition of alphabets and matrix dimensions.
- Sequence symbol encoding for fast easy table look-ups, as implemented in 
  HMMER[\[1\]](#ref1) or MEME[\[2\]](#ref2)
- Striped sequence matrices to process several positions in parallel, 
  inspired by Farrar[\[3\]](#ref3).
- High-throughput matrix row look-up using `permute` instructions of [AVX2](https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions).


## 💡 Example

```rust
use lightmotif::*;

// Create a position weight matrix from a collection of motif sequences
let cm = CountMatrix::from_sequences(&[
    EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
    EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
]).unwrap();
let pbm = cm.to_probability(0.1);
let pwm = pbm.to_weight(Background::uniform());

// Encode the target sequence into a striped matrix
let seq = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
let encoded = EncodedSequence::<DnaAlphabet>::from_text(seq).unwrap();
let mut striped = encoded.to_striped::<32>();

// Create a pipeline and compute scores for every position of the matrix
striped.configure(&pwm);
let pli = Pipeline::<_, f32>::new();
let scores = pli.score(&striped, &pwm);

// Scores can be extracted into a Vec<f32>
let v = scores.to_vec();
```

To use the AVX2 implementation, simply create a `Pipeline<_, __m256>` instead
of the `Pipeline<_, f32>`. This is only supported when the library is compiled
with the `avx2` target feature, but it can be easily configured with Rust's
`#[cfg]` attribute.

## ⏱️ Benchmarks

*Benchmarks were run on a [i7-10710U CPU](https://ark.intel.com/content/www/us/en/ark/products/196448/intel-core-i7-10710u-processor-12m-cache-up-to-4-70-ghz.html) running @1.10GHz, compiled with `--target-cpu=native`*.

Both benchmarks use the [MX000001](https://www.prodoric.de/matrix/MX000001.html) 
motif from [PRODORIC](https://www.prodoric.de/), and the 
[complete genome](https://www.ncbi.nlm.nih.gov/nuccore/U00096) of an
*Escherichia coli K12* strain.

- Score every position of the genome with the motif weight matrix:
  ```console
  running 3 tests
  test bench_avx2    ... bench:  13,053,752 ns/iter (+/- 45,411) = 355 MB/s
  test bench_ssse3   ... bench:  37,203,277 ns/iter (+/- 2,416,572) = 124 MB/s
  test bench_generic ... bench: 314,682,807 ns/iter (+/- 1,072,174) = 14 MB/s
  ```

- Find the highest-scoring position for a motif in a 10kb sequence 
  (compared to the PSSM algorithm implemented in 
  [`bio::pattern_matching::pssm`](https://docs.rs/bio/1.1.0/bio/pattern_matching/pssm/index.html)):
  ```console
  test bench_avx2    ... bench:      46,390 ns/iter (+/- 115) = 215 MB/s
  test bench_ssse3   ... bench:      97,691 ns/iter (+/- 2,720) = 102 MB/s
  test bench_generic ... bench:     740,305 ns/iter (+/- 2,527) = 13 MB/s
  test bench_bio     ... bench:   1,575,504 ns/iter (+/- 2,799) = 6 MB/s
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
