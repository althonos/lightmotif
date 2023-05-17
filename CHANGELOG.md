# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/lightmotif/compare/v0.2.0...HEAD


## [v0.2.0] - 2023-05-15
[v0.2.0]: https://github.com/althonos/lightmotif/compare/v0.1.1...v0.2.0

### Changed
- Crate structure to avoid cluttering the main `lightmotif` module namespace.
- Swizzling used in the SSSE3 implementation to make it require SSE2 only.
- TRANSFAC parser to support parsing sections in arbitrary order and accept additional metadata.
- Use `memchr` to parse lines faster in TRANSFAC parser.
- Use `typenum` and `generic-array` to handle constant matrix dimensions.
- Make `Score` trait generic over the number of columns in the striped sequence.

### Added
- Accessors for some of the attributes of `lightmotif_transfac::Matrix`.
- Pipeline method to extract the best position from a `StripedScore` matrix.
- Child trait for alphabet complementation.
- Methods for reverse-complementing all matrices from `lightmotif::pwm`.
- `Alphabet::symbols` method to get all the symbols of an alphabet.
- `StripedSequence::encode` constructor to encode and stripe a text sequence without allocating an extra `EncodedSequence`.
- Iterator methods and helper struct for `StripedScores`.
- `Protein` alphabet to `lightmotif::abc`.
- `DenseMatrix::uninitialized` constructor to allocate a dense matrix without filling its contents.
- `lightmotif::num` module with `typenum` re-exports and additional `StrictlyPositive` marker trait.
- Arm NEON implementation of the position scoring algorithm.
- `Display` implementation for `EncodedSequence` instead of an arbitrary `ToString` implementation.


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
