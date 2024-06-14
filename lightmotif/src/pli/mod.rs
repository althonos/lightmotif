//! Concrete implementations of the sequence scoring pipeline.

use std::ops::Range;

use crate::abc::Alphabet;
use crate::abc::Dna;
use crate::abc::Protein;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::dense::MatrixCoordinates;
use crate::err::InvalidSymbol;
use crate::err::UnsupportedBackend;
use crate::num::MultipleOf;
use crate::num::StrictlyPositive;
use crate::num::U16;
use crate::pwm::ScoringMatrix;
use crate::scores::StripedScores;
use crate::seq::EncodedSequence;
use crate::seq::StripedSequence;

use self::dispatch::Dispatch;
use self::platform::Avx2;
use self::platform::Backend;
use self::platform::Generic;
use self::platform::Neon;
use self::platform::Sse2;

pub mod dispatch;
pub mod platform;

// --- Score -------------------------------------------------------------------

/// Used for encoding a sequence into rank-based encoding.
pub trait Encode<A: Alphabet> {
    /// Encode the given sequence into a vector of symbols.
    fn encode_raw<S: AsRef<[u8]>>(&self, seq: S) -> Result<Vec<A::Symbol>, InvalidSymbol> {
        let s = seq.as_ref();
        let mut buffer = Vec::with_capacity(s.len());
        unsafe { buffer.set_len(s.len()) };
        match self.encode_into(s, &mut buffer) {
            Ok(_) => Ok(buffer),
            Err(e) => Err(e),
        }
    }

    /// Encode the given sequence into an `EncodedSequence`.
    fn encode<S: AsRef<[u8]>>(&self, seq: S) -> Result<EncodedSequence<A>, InvalidSymbol> {
        self.encode_raw(seq).map(EncodedSequence::new)
    }

    /// Encode the given sequence into a buffer of symbols.
    ///
    /// The destination buffer is expected to be large enough to store the
    /// entire sequence.
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        assert_eq!(seq.as_ref().len(), dst.len());
        for (i, c) in seq.as_ref().iter().enumerate() {
            dst[i] = A::Symbol::from_ascii(*c)?;
        }
        Ok(())
    }
}

/// Used computing sequence scores with a PSSM.
pub trait Score<A: Alphabet, C: StrictlyPositive> {
    /// Compute the PSSM scores into the given striped score matrix.
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<C>,
    ) where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.len() < pssm.len() {
            scores.resize(0, 0);
            return;
        }

        // FIXME?
        scores.resize(rows.len(), seq.len() - pssm.len() + 1);

        let result = scores.matrix_mut();
        let matrix = pssm.matrix();

        for (res_row, seq_row) in rows.enumerate() {
            for col in 0..C::USIZE {
                let mut score = 0.0;
                for (j, pssm_row) in matrix.iter().enumerate() {
                    let symbol = seq.matrix()[seq_row + j][col];
                    score += pssm_row[symbol.as_index()];
                }
                result[res_row][col] = score;
            }
        }
    }

    /// Compute the PSSM scores into the given striped score matrix.
    fn score_into<S, M>(&self, pssm: M, seq: S, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let s = seq.as_ref();
        let m = pssm.as_ref();
        let rows = s.matrix().rows() - s.wrap();
        Self::score_rows_into(&self, m, s, 0..rows, scores)
    }

    /// Compute the PSSM scores for every sequence positions.
    fn score<S, M>(&self, pssm: M, seq: S) -> StripedScores<C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        let mut scores = StripedScores::empty();
        self.score_into(pssm, seq, &mut scores);
        scores
    }
}

/// Used for finding the highest scoring site in a striped score matrix.
pub trait Maximum<C: StrictlyPositive> {
    /// Find the matrix coordinates with the highest score.
    fn argmax(&self, scores: &StripedScores<C>) -> Option<MatrixCoordinates> {
        if scores.is_empty() {
            return None;
        }

        let mut best_row = 0;
        let mut best_col = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, row) in scores.matrix().iter().enumerate() {
            for j in 0..C::USIZE {
                if row[j] >= best_score {
                    best_row = i;
                    best_col = j;
                    best_score = row[j];
                }
            }
        }

        Some(MatrixCoordinates::new(best_row, best_col))
    }

    /// Find the highest score.
    fn max(&self, scores: &StripedScores<C>) -> Option<f32> {
        self.argmax(scores).map(|c| scores.matrix()[c])
    }
}

/// Used for converting an encoded sequence into a striped sequence.
pub trait Stripe<A: Alphabet, C: StrictlyPositive> {
    /// Stripe a sequence into a striped, column-major order matrix.
    fn stripe<S: AsRef<[A::Symbol]>>(&self, seq: S) -> StripedSequence<A, C> {
        let s = seq.as_ref();
        let length = s.len();
        let rows = (length / C::USIZE) + ((length % C::USIZE > 0) as usize);
        let mut striped = StripedSequence::new(DenseMatrix::new(rows), length).unwrap();
        self.stripe_into(s, &mut striped);
        striped
    }

    /// Stripe a sequence into the given striped matrix.
    fn stripe_into<S: AsRef<[A::Symbol]>>(&self, seq: S, striped: &mut StripedSequence<A, C>) {
        // compute length of striped matrix
        let s = seq.as_ref();
        let length = s.len();
        let rows = (length + (C::USIZE - 1)) / C::USIZE;

        // get the data out of the given buffer
        let mut data = std::mem::take(striped).into_matrix();
        data.resize(rows);

        // stripe the sequence
        for (i, &x) in s.iter().enumerate() {
            data[i % rows][i / rows] = x;
        }
        for i in s.len()..data.rows() * data.columns() {
            data[i % rows][i / rows] = A::default_symbol();
        }

        // replace the original matrix with a new one
        *striped = StripedSequence::new(data, length).unwrap();
    }
}

/// Used for finding positions above a score threshold in a striped score matrix.
pub trait Threshold<C: StrictlyPositive> {
    /// Return the coordinates of positions with score equal to or greater than the threshold.
    ///
    /// # Note
    /// The indices are not be sorted, and the actual order depends on the
    /// implementation.
    fn threshold(&self, scores: &StripedScores<C>, threshold: f32) -> Vec<MatrixCoordinates> {
        let mut positions = Vec::new();
        for (i, row) in scores.matrix().iter().enumerate() {
            for col in 0..C::USIZE {
                assert!(!row[col].is_nan());
                if row[col] >= threshold {
                    positions.push(MatrixCoordinates::new(i, col));
                }
            }
        }
        positions
    }
}

// --- Pipeline ----------------------------------------------------------------

/// Wrapper implementing score computation for different platforms.
#[derive(Debug, Default, Clone)]
pub struct Pipeline<A: Alphabet, B: Backend> {
    alphabet: std::marker::PhantomData<A>,
    backend: B,
}

// --- Generic pipeline --------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Generic> {
    /// Create a new generic pipeline.
    pub const fn generic() -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            backend: Generic,
        }
    }
}

impl<A: Alphabet> Encode<A> for Pipeline<A, Generic> {}

impl<A: Alphabet, C: StrictlyPositive> Score<A, C> for Pipeline<A, Generic> {}

impl<A: Alphabet, C: StrictlyPositive> Maximum<C> for Pipeline<A, Generic> {}

impl<A: Alphabet, C: StrictlyPositive> Stripe<A, C> for Pipeline<A, Generic> {}

impl<A: Alphabet, C: StrictlyPositive> Threshold<C> for Pipeline<A, Generic> {}

// --- Dynamic dispatch --------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Dispatch> {
    /// Create a new dynamic dispatch pipeline.
    #[allow(unreachable_code)]
    pub fn dispatch() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            return Self {
                backend: Dispatch::Avx2,
                alphabet: std::marker::PhantomData,
            };
        }
        #[cfg(any(target_arch = "x86"))]
        if std::is_x86_feature_detected!("sse2") {
            return Self {
                backend: Dispatch::Sse2,
                alphabet: std::marker::PhantomData,
            };
        }
        #[cfg(any(target_arch = "x86_64"))]
        return Self {
            backend: Dispatch::Sse2,
            alphabet: std::marker::PhantomData,
        };
        #[cfg(target_arch = "arm")]
        if std::arch::is_arm_feature_detected!("neon") {
            return Self {
                backend: Dispatch::Neon,
                alphabet: std::marker::PhantomData,
            };
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Self {
                backend: Dispatch::Neon,
                alphabet: std::marker::PhantomData,
            };
        }
        Self {
            backend: Dispatch::Generic,
            alphabet: std::marker::PhantomData,
        }
    }
}

// --- SSE2 pipeline -----------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Sse2> {
    /// Attempt to create a new SSE2-accelerated pipeline.
    pub fn sse2() -> Result<Self, UnsupportedBackend> {
        #[cfg(target_arch = "x86")]
        if std::is_x86_feature_detected!("sse2") {
            return Ok(Self::default());
        }
        #[cfg(target_arch = "x86_64")]
        return Ok(Self::default());
        #[allow(unreachable_code)]
        Err(UnsupportedBackend)
    }
}

impl<A: Alphabet> Encode<A> for Pipeline<A, Sse2> {}

impl<A, C> Score<A, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<C>,
    ) where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        Sse2::score_rows_into(pssm, seq, rows, scores)
    }
}

impl<A, C> Maximum<C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
    fn argmax(&self, scores: &StripedScores<C>) -> Option<MatrixCoordinates> {
        Sse2::argmax(scores)
    }
}

impl<A: Alphabet, C: StrictlyPositive> Threshold<C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

// --- AVX2 pipeline -----------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Avx2> {
    /// Attempt to create a new AVX2-accelerated pipeline.
    pub fn avx2() -> Result<Self, UnsupportedBackend> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            return Ok(Self::default());
        }
        Err(UnsupportedBackend)
    }
}

impl<A: Alphabet> Encode<A> for Pipeline<A, Avx2> {
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        Avx2::encode_into::<A>(seq.as_ref(), dst)
    }
}

impl Score<Dna, <Avx2 as Backend>::LANES> for Pipeline<Dna, Avx2> {
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Avx2 as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Dna>>,
    {
        Avx2::score_rows_into_permute(pssm, seq, rows, scores)
    }
}

impl Score<Protein, <Avx2 as Backend>::LANES> for Pipeline<Protein, Avx2> {
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Protein, <Avx2 as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Protein>>,
    {
        Avx2::score_rows_into_gather(pssm, seq, rows, scores)
    }
}

impl<A: Alphabet> Stripe<A, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    /// Stripe a sequence into the given striped matrix.
    fn stripe_into<S: AsRef<[A::Symbol]>>(
        &self,
        seq: S,
        matrix: &mut StripedSequence<A, <Avx2 as Backend>::LANES>,
    ) {
        Avx2::stripe_into(seq, matrix)
    }
}

impl<A: Alphabet> Maximum<<Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    fn argmax(
        &self,
        scores: &StripedScores<<Avx2 as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        Avx2::argmax(scores)
    }
}

impl<A: Alphabet> Threshold<<Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {}

// --- NEON pipeline -----------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Neon> {
    /// Attempt to create a new AVX2-accelerated pipeline.
    pub fn neon() -> Result<Self, UnsupportedBackend> {
        #[cfg(target_arch = "arm")]
        if std::arch::is_arm_feature_detected!("neon") {
            return Ok(Self::default());
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Ok(Self::default());
        }
        Err(UnsupportedBackend)
    }
}

impl<A: Alphabet> Encode<A> for Pipeline<A, Neon> {
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        Neon::encode_into::<A>(seq.as_ref(), dst)
    }
}

impl<A, C> Score<A, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Maximum<C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Threshold<C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}
