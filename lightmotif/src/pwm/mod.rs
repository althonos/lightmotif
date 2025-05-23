//! Storage types for the different stages of a PSSM construction.
//!
//! A position-specific scoring matrix (PSSM) can be used to compute the
//! *log-likelihood ration* of a motif occurence in a sequence, relative
//! to a background sequence model.
//!
//! ## 📚 References
//!
//! - <a id="ref1">\[1\]</a> Bailey TL. *Discovering sequence motifs.*
//!   Methods Mol Biol. 2008;452:231-51.
//!   [PMID:18566768](https://pubmed.ncbi.nlm.nih.gov/18566768/).
//!   [doi:10.1007/978-1-60327-159-2_12](https://doi.org/10.1007/978-1-60327-159-2_12)

use std::ops::Index;

use crate::abc::Alphabet;
use crate::abc::Background;
use crate::abc::ComplementableAlphabet;
use crate::abc::Pseudocounts;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::err::InvalidData;
use crate::num::ArrayLength;
use crate::num::StrictlyPositive;
use crate::num::Unsigned;
use crate::pli::dispatch::Dispatch;
use crate::pli::Pipeline;
use crate::pli::Score;
use crate::scores::StripedScores;
use crate::seq::EncodedSequence;
use crate::seq::StripedSequence;

pub mod dist;

use self::dist::ScoreDistribution;

macro_rules! matrix_traits {
    ($mx:ident, $t:ty) => {
        impl<A: Alphabet> $mx<A> {
            /// The raw data storage for the matrix.
            #[inline]
            pub fn matrix(&self) -> &DenseMatrix<$t, A::K> {
                &self.data
            }

            /// Get the length of the motif encoded in this matrix.
            #[inline]
            pub fn len(&self) -> usize {
                self.data.rows()
            }

            /// Check whether the matrix is empty.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.data.rows() == 0
            }
        }

        impl<A: Alphabet> AsRef<$mx<A>> for $mx<A> {
            #[inline]
            fn as_ref(&self) -> &Self {
                self
            }
        }

        impl<A: Alphabet> AsRef<DenseMatrix<$t, A::K>> for $mx<A> {
            #[inline]
            fn as_ref(&self) -> &DenseMatrix<$t, A::K> {
                &self.data
            }
        }

        impl<A: Alphabet> Index<usize> for $mx<A> {
            type Output = <DenseMatrix<$t, A::K> as Index<usize>>::Output;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                self.data.index(index)
            }
        }

        impl<A: Alphabet> Correlation for $mx<A> {
            #[inline]
            fn num_rows(&self) -> usize {
                self.matrix().rows()
            }

            #[inline]
            fn dot(&self, other: &Self, i: usize, j: usize) -> f32 {
                self.data[i]
                    .iter()
                    .zip(&other.data[j])
                    .map(|(&x, &y)| (x as f32) * (y as f32))
                    .sum()
            }
        }
    };
}

// --- Correlation -------------------------------------------------------------

/// A trait for computing matrix correlations.
pub trait Correlation {
    /// The number of rows in the matrix.
    fn num_rows(&self) -> usize;

    /// Compute the dot product between row `i` of `self` and row `j` of `other`.
    fn dot(&self, other: &Self, i: usize, j: usize) -> f32;

    /// Compute the Euclidean norm of row `i`.
    #[inline]
    fn norm(&self, i: usize) -> f32 {
        self.dot(self, i, i).sqrt()
    }

    /// Compute the auto-correlation with the given delay.
    fn auto_correlation(&self, delay: usize) -> f32 {
        if delay >= self.num_rows() {
            return 0.0;
        }

        let norms = (0..self.num_rows())
            .map(|i| self.norm(i))
            .collect::<Vec<_>>();

        let mut c = 0.0;
        for (i, j) in (delay..self.num_rows()).enumerate() {
            let dot = self.dot(self, i, j);
            c += dot / (norms[i] * norms[j]);
        }

        c / (self.num_rows() - delay) as f32
    }

    /// Compute the cross-correlation between two matrices.
    fn cross_correlation(&self, other: &Self) -> f32 {
        let rows = self.num_rows().min(other.num_rows());
        let mut c = 0.0;
        for i in 0..rows {
            let dot = self.dot(other, i, i);
            c += dot / (self.norm(i) * other.norm(i));
        }
        c / (rows as f32)
    }
}

// --- CountMatrix -------------------------------------------------------------

/// A matrix storing symbol occurrences at each position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountMatrix<A: Alphabet> {
    /// The alphabet of the count matrix.
    alphabet: std::marker::PhantomData<A>,
    /// The actual counts for each position of the motif.
    data: DenseMatrix<u32, A::K>,
    /// The number of sequences from which this count matrix was obtained.
    #[allow(unused)]
    n: usize,
}

impl<A: Alphabet> CountMatrix<A> {
    /// Create a new count matrix without checking the contents.
    #[doc(hidden)]
    pub fn new_unchecked(data: DenseMatrix<u32, A::K>, n: usize) -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            n,
            data,
        }
    }

    /// Create a new count matrix from the given data.
    ///
    /// The matrix must contain count data, for sequences of the same
    /// length, i.e. rows should all sum to the same value.
    pub fn new(data: DenseMatrix<u32, A::K>) -> Result<Self, InvalidData> {
        // Empty matrices contain valid data.
        if data.rows() == 0 {
            return Ok(Self::new_unchecked(data, 0));
        }
        // Check row sums.
        let n = data
            .iter()
            .map(|row| row.iter().map(|i| *i as usize).sum())
            .max()
            .unwrap();
        // if data.iter().any(|row| row.iter().sum::<u32>() != n) {
        // Err(InvalidData)
        // } else {
        Ok(Self::new_unchecked(data, n))
        // }
    }

    /// Create a new count matrix from the given sequences.
    ///
    /// # Errors
    /// This function returns `Err(InvalidData)` when the sequences do not
    /// all have the same length:
    /// ```rust
    /// # use lightmotif::seq::EncodedSequence;
    /// # use lightmotif::pwm::CountMatrix;
    /// # use lightmotif::abc::Dna;
    /// # use lightmotif::abc::Nucleotide::*;
    /// let result = CountMatrix::<Dna>::from_sequences([
    ///     EncodedSequence::new(vec![T, T, A, T]),
    ///     EncodedSequence::new(vec![T, C, A]),
    /// ]);
    /// assert!(result.is_err());
    /// ```
    pub fn from_sequences<I>(sequences: I) -> Result<Self, InvalidData>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: AsRef<EncodedSequence<A>>,
    {
        let mut n = 0;
        let mut data = None;
        for seq in sequences {
            let seq = seq.as_ref();
            let d = match data.as_mut() {
                Some(d) => d,
                None => {
                    data = Some(DenseMatrix::new(seq.len()));
                    data.as_mut().unwrap()
                }
            };
            if seq.len() != d.rows() {
                return Err(InvalidData);
            }
            for (i, x) in seq.into_iter().enumerate() {
                d[i][x.as_index()] += 1;
            }
            n += 1;
        }
        match data {
            None => Ok(Self::new_unchecked(DenseMatrix::new(0), n)),
            Some(matrix) => Ok(Self::new_unchecked(matrix, n)),
        }
    }

    /// Build a probability matrix from this count matrix using pseudo-counts.
    pub fn to_freq<P>(&self, pseudo: P) -> FrequencyMatrix<A>
    where
        P: Into<Pseudocounts<A>>,
    {
        let p = pseudo.into();
        let mut probas = DenseMatrix::new(self.data.rows());
        for i in 0..self.data.rows() {
            let src = &self.data[i];
            let dst = &mut probas[i];
            for (j, &x) in src.iter().enumerate() {
                dst[j] = x as f32 + p.counts()[j];
            }
            let s: f32 = dst.iter().sum();
            for x in dst.iter_mut() {
                *x /= s;
            }
        }
        FrequencyMatrix::new_unchecked(probas)
    }

    /// Get the number of sequences used to build the matrix.
    pub fn sequence_count(&self) -> usize {
        self.n
    }

    /// Compute the Shannon entropy of a single row.
    fn row_entropy(row: &[u32]) -> f32 {
        let sum = row.iter().sum::<u32>();
        -row.iter()
            .map(|&n| n as f32 / sum as f32)
            .map(|p| if p > 0.0 { p * p.log2() } else { 0.0 })
            .sum::<f32>()
    }

    /// Compute the Shannon entropy of each row of the matrix.
    ///
    /// The entropy of a row, sometimes refered to as "uncertainty", is
    /// computed by treating each motif position as a random variable taking
    /// values in alphabet `A`.
    pub fn entropy(&self) -> Vec<f32> {
        self.matrix()
            .iter()
            .map(|row| Self::row_entropy(row))
            .collect()
    }

    /// Build the consensus sequence for the matrix.
    ///
    /// For each row of the matrix, the symbol with the highest occurence
    /// is extracted. Symbols as position with entropy higher than 1.0 are
    /// shown in lowercase.
    pub fn consensus(&self) -> String {
        let mut consensus = String::with_capacity(self.matrix().rows());
        self.matrix().iter().for_each(|row| {
            let entropy = Self::row_entropy(row);
            let symbol = row
                .iter()
                .zip(A::symbols())
                .max_by_key(|(count, _)| **count)
                .unwrap()
                .1;
            if entropy >= 1.0 {
                consensus.push(symbol.as_char().to_ascii_lowercase());
            } else {
                consensus.push(symbol.as_char().to_ascii_uppercase());
            }
        });
        consensus
    }
}

impl<A: ComplementableAlphabet> CountMatrix<A> {
    /// Get the reverse-complement of this count matrix.
    pub fn reverse_complement(&self) -> Self {
        let mut data = DenseMatrix::new(self.data.rows());
        for (i, row) in self.data.iter().rev().enumerate() {
            for &s in A::symbols() {
                data[i][s.as_index()] = row[A::complement(s).as_index()];
            }
        }
        Self::new_unchecked(data, self.n)
    }
}

impl<A: Alphabet> FromIterator<EncodedSequence<A>> for Result<CountMatrix<A>, InvalidData> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = EncodedSequence<A>>,
    {
        CountMatrix::from_sequences(iter)
    }
}

matrix_traits!(CountMatrix, u32);

// --- FrequencyMatrix ---------------------------------------------------------

/// A matrix storing symbol frequencies at each position.
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyMatrix<A: Alphabet> {
    alphabet: std::marker::PhantomData<A>,
    data: DenseMatrix<f32, A::K>,
}

impl<A: Alphabet> FrequencyMatrix<A> {
    /// Create a new frequency matrix without checking the contents.
    #[doc(hidden)]
    pub fn new_unchecked(data: DenseMatrix<f32, A::K>) -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            data,
        }
    }

    /// Create a new frequency matrix.
    ///
    /// The matrix must contain frequency data, i.e. rows should all sum to 1
    /// (with a tolerance of 0.01).
    pub fn new(data: DenseMatrix<f32, A::K>) -> Result<Self, InvalidData> {
        if data
            .iter()
            .all(|row| (row.iter().sum::<f32>() - 1.0).abs() < 0.01)
        {
            Ok(Self::new_unchecked(data))
        } else {
            Err(InvalidData)
        }
    }

    /// Convert to a weight matrix using the given background frequencies.
    ///
    /// # Note
    /// By convention, columns with null background frequencies receive an
    /// odds-ratio of zero, which will then translate into a log-odds-ratio
    /// of [`f32::NEG_INFINITY`](https://doc.rust-lang.org/std/primitive.f32.html#associatedconstant.NEG_INFINITY)
    /// in the resulting scoring matrix.
    pub fn to_weight<B>(&self, background: B) -> WeightMatrix<A>
    where
        B: Into<Option<Background<A>>>,
    {
        let bg = background.into().unwrap_or_default();
        let mut weight = DenseMatrix::new(self.data.rows());
        for (src, dst) in self.data.iter().zip(weight.iter_mut()) {
            for (j, (&x, &f)) in src.iter().zip(bg.frequencies()).enumerate() {
                if f == 0.0 {
                    dst[j] = 0.0;
                } else {
                    dst[j] = x / f;
                }
            }
        }
        WeightMatrix::new_unchecked(bg, weight)
    }

    /// Convert into a scoring matrix using the given background frequencies.
    ///
    /// This uses the base-2 logarithm.
    ///
    /// # Note
    /// By convention, columns with null background frequencies receive a
    /// log-odds-ratio of [`f32::NEG_INFINITY`](https://doc.rust-lang.org/std/primitive.f32.html#associatedconstant.NEG_INFINITY).
    pub fn to_scoring<B>(&self, background: B) -> ScoringMatrix<A>
    where
        B: Into<Option<Background<A>>>,
    {
        self.clone().into_scoring(background)
    }

    /// Convert into a scoring matrix using the given background frequencies.
    ///
    /// This uses the base-2 logarithm.
    ///
    /// # Note
    /// By convention, columns with null background frequencies receive a
    /// log-odds-ratio of [`f32::NEG_INFINITY`](https://doc.rust-lang.org/std/primitive.f32.html#associatedconstant.NEG_INFINITY).
    pub fn into_scoring<B>(mut self, background: B) -> ScoringMatrix<A>
    where
        B: Into<Option<Background<A>>>,
    {
        let bg = background.into().unwrap_or_default();
        for src in self.data.iter_mut() {
            for (x, &f) in src.iter_mut().zip(bg.frequencies()) {
                if f == 0.0 {
                    *x = f32::NEG_INFINITY;
                } else {
                    *x = (*x / f).log2();
                }
            }
        }
        ScoringMatrix::new(bg, self.data)
    }
}

impl<A: ComplementableAlphabet> FrequencyMatrix<A> {
    /// Get the reverse-complement of this frequency matrix.
    pub fn reverse_complement(&self) -> Self {
        let mut data = DenseMatrix::new(self.data.rows());
        for (i, row) in self.data.iter().rev().enumerate() {
            for &s in A::symbols() {
                data[i][s.as_index()] = row[A::complement(s).as_index()];
            }
        }
        Self::new_unchecked(data)
    }
}

matrix_traits!(FrequencyMatrix, f32);

// --- WeightMatrix ------------------------------------------------------------

/// A matrix storing odds ratio of symbol occurrences at each position.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightMatrix<A: Alphabet> {
    background: Background<A>,
    data: DenseMatrix<f32, A::K>,
}

impl<A: Alphabet> WeightMatrix<A> {
    /// Create a new weight matrix without checking the contents.
    #[doc(hidden)]
    pub fn new_unchecked(background: Background<A>, data: DenseMatrix<f32, A::K>) -> Self {
        Self { background, data }
    }

    /// The background frequencies of the position weight matrix.
    #[inline]
    pub fn background(&self) -> &Background<A> {
        &self.background
    }

    /// Rescale this weight matrix with a different background.
    pub fn rescale<B>(&self, background: B) -> Self
    where
        B: Into<Option<Background<A>>>,
    {
        let b = background.into().unwrap_or_default();
        if b.frequencies() != self.background.frequencies() {
            let old_freqs = self.background.frequencies();
            let new_freqs = b.frequencies();
            let mut data = self.data.clone();
            for row in data.iter_mut() {
                for j in 0..A::K::USIZE {
                    row[j] *= old_freqs[j] / new_freqs[j];
                }
            }
            Self {
                data,
                background: b,
            }
        } else {
            self.clone()
        }
    }

    /// Compute the information content of the matrix.
    pub fn information_content(&self) -> f32 {
        self.matrix()
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.background.frequencies())
                    .map(|(x, b)| if *b == 0.0 { 0.0 } else { x * (x / b).log2() })
                    .sum::<f32>()
            })
            .sum()
    }

    /// Get a position-specific scoring matrix from this position weight matrix.
    pub fn to_scoring(&self) -> ScoringMatrix<A> {
        self.to_scoring_with_base(2.0)
    }

    /// Get a position-specific scoring matrix from this position weight matrix.
    pub fn to_scoring_with_base(&self, base: f32) -> ScoringMatrix<A> {
        let background = self.background.clone();
        let mut data = self.data.clone();
        for row in data.iter_mut() {
            for item in row.iter_mut() {
                *item = match base {
                    2.0 => item.log2(),
                    10.0 => item.log10(),
                    _ => item.log(base),
                };
            }
        }
        ScoringMatrix::new(background, data)
    }
}

impl<A: ComplementableAlphabet> WeightMatrix<A> {
    /// Get the reverse-complement of this weight matrix.
    pub fn reverse_complement(&self) -> Self {
        let mut data = DenseMatrix::new(self.data.rows());
        for (i, row) in self.data.iter().rev().enumerate() {
            for &s in A::symbols() {
                data[i][s.as_index()] = row[A::complement(s).as_index()];
            }
        }
        Self::new_unchecked(self.background.clone(), data)
    }
}

impl<A: Alphabet> From<ScoringMatrix<A>> for WeightMatrix<A> {
    fn from(pwm: ScoringMatrix<A>) -> Self {
        let background = pwm.background;
        let mut data = pwm.data;
        for row in data.iter_mut() {
            for item in row.iter_mut() {
                *item = 2f32.powf(*item);
            }
        }
        WeightMatrix { background, data }
    }
}

matrix_traits!(WeightMatrix, f32);

// --- ScoringMatrix -----------------------------------------------------------

/// A matrix storing log-odds ratio of symbol occurrences at each position.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoringMatrix<A: Alphabet> {
    background: Background<A>,
    data: DenseMatrix<f32, A::K>,
}

impl<A: ComplementableAlphabet> ScoringMatrix<A> {
    /// Get the reverse-complement of this scoring matrix.
    pub fn reverse_complement(&self) -> Self {
        let mut data = DenseMatrix::new(self.data.rows());
        for (i, row) in self.data.iter().rev().enumerate() {
            for &s in A::symbols() {
                data[i][s.as_index()] = row[A::complement(s).as_index()];
            }
        }
        Self::new(self.background.clone(), data)
    }
}

impl<A: Alphabet> ScoringMatrix<A> {
    /// Create a new scoring matrix from the given log-odds matrix.
    pub fn new(background: Background<A>, data: DenseMatrix<f32, A::K>) -> Self {
        Self { background, data }
    }

    /// The background frequencies of the position weight matrix.
    #[inline]
    pub fn background(&self) -> &Background<A> {
        &self.background
    }

    /// The lowest score that can be obtained with this scoring matrix.
    pub fn min_score(&self) -> f32 {
        self.data
            .iter()
            .map(|row| {
                row[..A::K::USIZE - 1]
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .sum()
    }

    /// The highest score that can be obtained with this scoring matrix.
    pub fn max_score(&self) -> f32 {
        self.data
            .iter()
            .map(|row| {
                row[..A::K::USIZE - 1]
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .sum()
    }

    /// Compute the information content of the matrix.
    pub fn information_content(&self) -> f32 {
        self.matrix()
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.background.frequencies())
                    .map(|(x, b)| {
                        if *b == 0.0 || *x == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (2f32.powf(*x) * b) * x
                        }
                    })
                    .sum::<f32>()
            })
            .sum()
    }

    /// Compute the PSSM scores for every position of the given sequence.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn score<S, C>(&self, seq: S) -> StripedScores<f32, C>
    where
        C: StrictlyPositive + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
        Pipeline<A, Dispatch>: Score<f32, A, C>,
    {
        let pli = Pipeline::dispatch();
        pli.score(self, seq)
    }

    /// Compute the score for a single sequence position.
    pub fn score_position<S, C>(&self, seq: S, pos: usize) -> f32
    where
        C: StrictlyPositive + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
    {
        let mut score = 0.0;
        let s = seq.as_ref();
        for (j, row) in self.data.iter().enumerate() {
            score += row[s[pos + j].as_index()]
        }
        score
    }

    /// Get a discrete matrix from this position-specific scoring matrix.
    pub fn to_discrete(&self) -> DiscreteMatrix<A> {
        let max_score = self.max_score();
        let offsets = self
            .matrix()
            .iter()
            .map(|row| {
                row[..A::K::USIZE - 1]
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap()
            })
            .cloned()
            .collect::<Vec<f32>>();
        let offset = offsets.iter().sum::<f32>();
        let factor = (max_score - offset) / (u8::MAX as f32);
        let pssm = self.matrix();
        let mut data = DenseMatrix::new(self.len());
        for i in 0..data.rows() {
            for j in 0..data.columns() {
                data[i][j] = ((pssm[i][j] - offsets[i]) / factor).ceil() as u8;
            }
        }
        DiscreteMatrix {
            data,
            factor,
            offsets,
            offset,
        }
    }

    /// Get the score distribution for this position-specific scoring matrix.
    ///
    /// See [`crate::pwm::dist`] module for more information.
    ///
    /// # Note
    /// Generating a [`ScoreDistribution`] takes time, so this method should be
    /// called only once per [`ScoringMatrix`] and the result cached for
    /// optimal performance.
    #[must_use]
    pub fn to_score_distribution(&self) -> ScoreDistribution<A> {
        ScoreDistribution::from(self)
    }
}

impl<A: Alphabet> From<WeightMatrix<A>> for ScoringMatrix<A> {
    fn from(pwm: WeightMatrix<A>) -> Self {
        pwm.to_scoring()
    }
}

matrix_traits!(ScoringMatrix, f32);

// --- DiscreteMatrix ----------------------------------------------------------

/// A position-specific scoring matrix discretized over `u8::MIN..u8::MAX`.
///
/// # Note
/// The discretization is done by rounding the error *up*, so that the scores
/// computed through a discrete matrix are over an *over*-estimation of the
/// actual scores. This allows for the fast scanning for candidate positions,
/// which have then to be scanned again with the full PSSM to compute the real
/// scores.
///
/// # Example
/// ```
/// # use lightmotif::*;
/// # let counts = CountMatrix::<Dna>::from_sequences(
/// #  ["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"]
/// #        .into_iter()
/// #        .map(|s| EncodedSequence::encode(s).unwrap()),
/// # )
/// # .unwrap();
/// # let pssm = counts.to_freq(0.1).to_scoring(None);
/// # let seq = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
/// # let mut striped = EncodedSequence::encode(seq).unwrap().to_striped();
/// // Create a `DiscreteMatrix` from a `ScoringMatrix`
/// let discrete = pssm.to_discrete();
///
/// // The discrete scores are always higher than the real scores.
/// for j in 0..seq.len() - pssm.len() + 1 {
///     let score_f32 = pssm.score_position(&striped, 1);
///     let score_u8 = discrete.unscale(discrete.score_position(&striped, 1));
///     assert!(score_u8 >= score_f32);
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteMatrix<A: Alphabet> {
    data: DenseMatrix<u8, A::K>,
    factor: f32,
    offsets: Vec<f32>,
    offset: f32,
}

impl<A: Alphabet> DiscreteMatrix<A> {
    /// Compute the score for a single sequence position.
    pub fn score_position<S, C>(&self, seq: S, pos: usize) -> u8
    where
        C: StrictlyPositive + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
    {
        let mut score = 0;
        let s = seq.as_ref();
        for (j, row) in self.data.iter().enumerate() {
            score += row[s[pos + j].as_index()]
        }
        score
    }

    /// Scale the given score to an integer score using the matrix scale.
    ///
    /// # Note
    /// This function rounds down the final score, and is suitable to translate
    /// an `f32` score threshold to a `u8` score threshold.
    #[inline]
    pub fn scale(&self, score: f32) -> u8 {
        ((score - self.offset) / self.factor).floor() as u8
    }

    /// Unscale the given integer score into a score using the matrix scale.
    #[inline]
    pub fn unscale(&self, score: u8) -> f32 {
        (score as f32) * self.factor + self.offset
    }
}

impl<A: Alphabet> From<ScoringMatrix<A>> for DiscreteMatrix<A> {
    fn from(value: ScoringMatrix<A>) -> Self {
        Self::from(&value)
    }
}

impl<A: Alphabet> From<&ScoringMatrix<A>> for DiscreteMatrix<A> {
    fn from(s: &ScoringMatrix<A>) -> Self {
        s.to_discrete()
    }
}

matrix_traits!(DiscreteMatrix, u8);

#[cfg(test)]
mod test {
    use super::*;
    use crate::abc::Dna;

    #[test]
    fn auto_correlation() {
        let counts = DenseMatrix::<u32, <Dna as Alphabet>::K>::from_rows([
            [1, 3, 3, 1, 0],
            [8, 0, 0, 0, 0],
            [1, 7, 0, 0, 0],
            [3, 2, 2, 1, 0],
            [1, 3, 3, 1, 0],
            [8, 0, 0, 0, 0],
            [1, 7, 0, 0, 0],
            [3, 2, 2, 1, 0],
        ]);
        let counts = CountMatrix::<Dna>::new(counts).unwrap();
        assert_eq!(counts.auto_correlation(0), 1.0);
        assert_eq!(counts.auto_correlation(4), 1.0);
    }
}
