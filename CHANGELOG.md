# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
[Unreleased]: https://github.com/althonos/lightmotif/compare/v0.8.0...HEAD


## [v0.8.0] - 2024-06-28
[v0.8.0]: https://github.com/althonos/lightmotif/compare/v0.7.3...v0.8.0

### Added

#### `lightmotif`
- `lightmotif::pwm::DiscreteMatrix` to store a discretized `ScoringMatrix`.
- `MatrixElement` marker trait for `Default + Copy` types that can be used as a `DenseMatrix` element.
- `lightmotif::scores::Scores` wrapper of `Vec<T>` with shortcut `max`, `threshold` and `argmax` methods.
- AVX2 and NEON implementations of `Score<u8, Dna>` pipeline operation.

### Fixed

#### `lightmotif`
- Remove requirement for `T: Debug` in `Clone` implementation of `DenseMatrix<T>`.
- SSE2 and AVX2 implementations of `Argmax` not returning the last index in case of ties.

#### `lightmotif-tfmpvalue`
- Cache internal buffer for Q-values to avoid reallocation between iterations.

### Changed

#### `lightmotif`
- Cache the `Pipeline` used in `lightmotif::scan::Scanner`.
- Make `lightmotif::scores::StripedScores` generic over the score type.
- Make `Maximum` and `Threshold` generic over the score type.
- Use a discrete matrix in `Scanner` to quickly eliminate blocks without candidate position above given threshold.
- Compile crate without the `rand` and `rand-distr` features by default.

#### `lightmotif-py`
- Update `pyo3` dependency to `v0.22.0`.


## [v0.7.3] - 2024-06-17
[v0.7.3]: https://github.com/althonos/lightmotif/compare/v0.7.2...v0.7.3

### Added

#### `lightmotif-io`
- `lightmotif_io::jaspar` to read matrices in raw JASPAR format.
- Implementation of `Clone` and `Error` for the `lightmotif_io::error::Error` type.

### Changed

#### `lightmotif`
- Use logarithmic subtraction in `FrequencyMatrix::to_scoring`.
- Use `_mm256_permutevara8x32_ps` instead of `_mm256_permutevar_ps` in AVX2 implementation to avoid special handling of unknown matric character `N`.

### Fixed

#### `lightmotif`
- Striping of empty sequences panicking on AVX2 implementation.
- Scoring of empty range of rows panicking.
- Invalid alignment of data in AVX2 code.
- `Clone` implementation of `DenseMatrix` not preserving memory alignment.


## [v0.7.2] - 2024-06-14
[v0.7.2]: https://github.com/althonos/lightmotif/compare/v0.7.1...v0.7.2

### Added

#### `lightmotif-py`
- Add `__len__` implementation for all matrix clases.
- Allow changing the logarithm base in `WeightMatrix.log_odds` method.
- Implement `__getitem__` for row access for all matrix classes.


## [v0.7.1] - 2024-06-14
[v0.7.1]: https://github.com/althonos/lightmotif/compare/v0.7.0...v0.7.1

### Fixed

#### `lightmotif`
- Resizing of `StripedScores` output in NEON implementation of `Score`.


## [v0.7.0] - 2024-06-14
[v0.7.0]: https://github.com/althonos/lightmotif/compare/v0.6.0...v0.7.0

### Added

#### `lightmotif`
- Implement indexing of `StripedSequence` by sequence index.
- `matrix` accessor to all matrix types in `lightmotif::pwm` and `lightmotif::seq`.
- `entropy` and `information_content` methods to `CountMatrix`.
- `SymbolCount` trait for counting the number of occurrences of a symbol in an iterable.
- Several `Background` constructors for counting occurences in one or more sequences.
- `FromIterator<A::Symbol>` constructor for `EncodedSequence<A>`.
- `MultipleOf<N>` trait to simplify typenums in platform code signatures.
- Sampling of random sequences using the `rand` dependency under a feature flag.
- `ScoringMatrix.score_into` method to re-use a `StripedScores` buffer.
- `ScoringMatrix.score_position` method to score a single sequence position.
- Indexing by `MatrixCoordinates` in `DenseMatrix`.
- Support for chanding logarithm base when building a `ScoringMatrix` from a `WeightMatrix`.
- Scanning algorithm for finding hits in a sequence with an iterator without allocating `StripedScores` for each sequence position.

### `lightmotif-py`
- Support for optional TFMPvalue interface in Python bindings under GPLv3+ code.
- Constructor for `ScoringMatrix` class.
- `ScoringMatrix.reverse_complement` to compute the reverse-complement of a scoring matrix.

### Changed

#### `lightmotif`
- Make `EncodedSequence.stripe` use a dispatching `Pipeline` internally.
- Require power-of-two alignment in `DenseMatrix` implementations.
- Update `generic-array` dependency to `v1.0`.
- Change order of parameters in `ScoringMatrix.score`.
- Reorganize scoring trait and implement row-slice scoring for AVX2 and SSE2.
- Rewrite `Pipeline::threshold` to return matrix coordinates instead of a sequence index.
- Rewrite `Pipeline::argmax` to return matrix coordinates instead of a sequence index.

#### `lightmotif-py`
- Streamline the use of pipelined functions in Python bindings.

### Fixed

#### `lightmotif`
- Handling of unknown residues in `permute` implementation of `Score` on AVX2.
- `PartialEq` for `DenseMatrix` to ignore alignment padding in each row. 

### Removed

#### `lightmotif`
- Platform-specific code for thresholding a `StripedScores` matrix.
- Direct attribute access in `StripedSequence`.

#### `lightmotif-transfac`
- Remove crate from repository, superseded by the `lightmotif-io` crate.


## [v0.6.0] - 2023-12-13
[v0.6.0]: https://github.com/althonos/lightmotif/compare/v0.5.1...v0.6.0

### Added

#### `lightmotif`
- Validating constructor `::pwm::FrequencyMatrix::new` testing for frequencies on each ranks.
- Getter to the raw data matrix of a `::pwm::FrequencyMatrix`.

#### `lightmotif-io`
New crate with JASPAR, TRANSFAC and UNIPROBE format parsers.

### Changed

#### `lightmotif`
- Make `max_score` and `min_score` columns of `::pwm::ScoringMatrix` ignore the wildcard symbol column.

#### `lightmotif-tfmpvalue`
- Invert decay in `TfmPvalue` to reduce rounding errors when computing granularity.
- Use a fast integer hashing algorithm for `i64` keys in maps used for recording *Q*-values.
- Compute the optimal column permutation to accelerate the computation of score distributions.

### Removed

#### `lightmotif-transfac`
Deprecate crate in favour of `lightmotif-io`.


## [v0.5.1] - 2023-08-31
[v0.5.1]: https://github.com/althonos/lightmotif/compare/v0.5.0...v0.5.1

### Fixed

#### `lightmotif`
- Compilation for Arm NEON platforms.


## [v0.5.0] - 2023-08-31
[v0.5.0]: https://github.com/althonos/lightmotif/compare/v0.4.0...v0.5.0

### Added

#### `lightmotif`
- Arm NEON implementation of `Threshold` and `Encode`.
- `Alphabet::as_str` method to get the symbols of an alphabet as a string.
- `DenseMatrix::fill` method to fill a dense matrix with a constant value.
- Convenience getters and conversion traits to `StripedSequence`.
- `Stripe` trait to implement striping of an encoded sequence with SIMD.
- Dynamic dispatched pipeline selecting the best implementation as runtime.
- `PartialEq` implementation for `EncodedSequence`.

#### `lightmotif-py`
- Buffer and copy protocols for `StripedSequence` and `EncodedSequence`.
- Indexing support to `EncodedSequence`.

#### `lightmotif-tfmpvalue`
- Convenience methods to access the wrapped `ScoringMatrix` reference in `TfmPvalue`.

### Changed

#### `lightmotif`
- `Encode::encode` now returns an `EncodedSequence` instead of a raw `Vec`.
- Performance improvements for `Encode` for AVX2 and NEON by removing non-const function calls in loop.
- Performance improvements for `Threshold` by skipping the index buffer initialization.
- Avoid buffer initialization when allocating a new buffer in `EncodedSequence::encode`.
- Require `Symbol` implementors to implement `Eq`.

#### `lightmotif-py`
- Use the dynamic dispatch pipeline to run vectorized operations.

#### `lightmotif-tfmpvalue`
- Make `TfmPvalue` generic over the type of reference to the wrapped `ScoringMatrix`.

### Fixed

#### `lightmotif`
- `Debug` implementation of `DenseMatrix` crashing when attempting to render the padding bytes.


## [v0.4.0] - 2023-08-10
[v0.4.0]: https://github.com/althonos/lightmotif/compare/v0.3.0...v0.4.0

### Changed

#### `lightmotif`
- Improve `DenseMatrix::resize` performance when downsizing.
- Explicitly panic when sequence is too long to be processed with `u32` indices in AVX2 and SSE2.
- Reorganize `DenseMatrix` column alignment and index storage.
- Improve `Score` performance by using pointers instead of slices in SIMD loops.
- Rename `DenseMatrix::columns_effective` to `DenseMatrix::stride`.
- Use streaming intrinsics to store data in AVX2 and SSE2 implementations.
- Rename `BestPosition` trait to `Maximum`.

#### `lightmotif-py`
- Avoid error on missing symbols in `CountMatrix.__init__`

### Added

#### `lightmotif`
- `Threshold` trait to find all position above a certain score in a `StripedScores` matrix.
- `Encode` trait to encode an ASCII sequence into a `Vec<Symbol>`.
- AVX2 implementation of `Score` using `gather` instead of `permute` for protein alphabets.
- `From<Vec<_>>` and `Default` trait implementations to `EncodedSequence`.
- `Alphabet` methods to operate on ASCII strings.
- `StripedScores.is_empty` method to check if a `StripedScores` contains any score.

#### `lightmotif-py`
- `StripedScores.threshold` method wrapping the `Threshold` pipeline operation.
- `StripedScores.max` and `StripedScores.argmax` methods to get the best score and best position.

### Fixed

#### `lightmotif`
- `Score` causing an overflow when given a sequence shorter than the PSSM.
- `Maximum` trait returns the smallest position on equal maxima.


## [v0.3.0] - 2023-06-25
[v0.3.0]: https://github.com/althonos/lightmotif/compare/v0.2.0...v0.3.0

### Changed
- Rewrite the SSE2 maximum search implementation using a generic number of columns.
- Refactor `lightmotif::pwm` to avoid infinite odds-ratio for columns with zero background frequencies.

### Added
- `lightmotif-tfmpvalue` crate implementing the TFMPvalue for computing p-values for a `ScoringMatrix`.
- `DenseMatrix::from_rows` method to create a dense matrix from an iterable of rows.
- `PartialEq` implementation for matrices in `lightmotif`.
- Methods to compute the minimum and maximum scores of a `ScoringMatrix`.


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
