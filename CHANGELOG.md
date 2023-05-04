# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/lightmotif/compare/v0.1.0...HEAD


## [v0.1.1] - 2023-05-04
[v0.1.1]: https://github.com/althonos/lightmotif/compare/v0.1.0...v0.1.1

### Added
- Helper crate to detect CPU features support at runtime.

### Fixed
- AVX2 code being imported on x86-64 platforms without checking for OS support.
- AVX2-enabled extension always being compiled even on platforms with no AVX2 support.

### Removed
- `built` and `pyo3-built` build dependencies (causing issues with workspaces).


## [v0.1.0] - 2023-05-04
[v0.1.0]: https://github.com/althonos/lightmotif/compare/4ccf9596b7...v0.1.0

Initial release.
