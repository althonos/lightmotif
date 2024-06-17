# 🎼🧬 `lightmotif-io` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*Parser implementations of several formats for the [`lightmotif`](https://crates.io/crates/lightmotif) crate.*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/rust.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/lightmotif/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Crate](https://img.shields.io/crates/v/lightmotif-io.svg?maxAge=600&style=flat-square)](https://crates.io/crates/lightmotif-io)
[![Docs](https://img.shields.io/docsrs/lightmotif-io?maxAge=600&style=flat-square)](https://docs.rs/lightmotif-io)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/tree/main/lightmotif-io)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)

## 🗺️ Overview

Position-specific scoring matrices are relatively small and easy to exchange
as count matrices, or sometimes directly as log-odds, but different formats
have been developed independently to accomodate the associated metadata by 
different databases. This crate provides convenience parsers to load 
[`lightmotif`](https://crates.io/crates/lightmotif) matrices from several 
PSSM file formats, including:

- [x] [TRANSFAC](https://en.wikipedia.org/wiki/TRANSFAC)-formatted records,
  with associated metadata. 
- [x] [JASPAR](https://jaspar.elixir.no/docs/) count matrices
  (in JASPAR 2016 bracketed format or raw format) with their record header.
- [x] [UniPROBE](http://the_brain.bwh.harvard.edu/uniprobe/index.php) 
  frequency matrices with their record header.
- [ ] [MEME minimal](https://meme-suite.org/meme/doc/meme-format.html) records
  with their metadata, background probabilities, and frequency matrices.


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