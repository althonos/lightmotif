# 🎼🧬 `lightmotif-transfac` [![Star me](https://img.shields.io/github/stars/althonos/lightmotif.svg?style=social&label=Star&maxAge=3600)](https://github.com/althonos/lightmotif/stargazers)

*A TRANSFAC parser implementation for the [`lightmotif`](https://crates.io/crates/lightmotif) crate.*.

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/lightmotif/rust.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/lightmotif/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/lightmotif?logo=codecov&style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/lightmotif/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![Crate](https://img.shields.io/crates/v/lightmotif-transfac.svg?maxAge=600&style=flat-square)](https://crates.io/crates/lightmotif-transfac)
[![Docs](https://img.shields.io/docsrs/lightmotif-transfac?maxAge=600&style=flat-square)](https://docs.rs/lightmotif-transfac)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/tree/main/lightmotif-transfac)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/lightmotif/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/lightmotif.svg?style=flat-square&maxAge=600)](https://github.com/althonos/lightmotif/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/lightmotif/blob/master/CHANGELOG.md)

## 🗺️ Overview

The [TRANSFAC](https://en.wikipedia.org/wiki/TRANSFAC) database is a collection
of transcription factors with their binding sites. It provides
[position-specific scoring matrices](https://en.wikipedia.org/wiki/Position_weight_matrix)
for individual transcription factors or closesly related groups.

This crate provides a parser to load TRANSFAC binding sites into `lightmotif`
matrices to be used with the accelerated search pipeline. See the 
[`lightmotif`](https://crates.io/crates/lightmotif) crate for more information.

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