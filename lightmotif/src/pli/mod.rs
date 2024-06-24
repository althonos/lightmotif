//! Concrete implementations of the sequence scoring pipeline.

use std::ops::AddAssign;
use std::ops::Range;

use generic_array::ArrayLength;

use crate::abc::Alphabet;
use crate::abc::Dna;
use crate::abc::Protein;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::dense::MatrixCoordinates;
use crate::dense::MatrixElement;
use crate::err::InvalidSymbol;
use crate::err::UnsupportedBackend;
use crate::num::MultipleOf;
use crate::num::StrictlyPositive;
use crate::num::U16;
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
pub trait Score<T: MatrixElement + AddAssign, A: Alphabet, C: StrictlyPositive> {
    /// Compute the PSSM scores into the given striped score matrix.
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<T, C>,
    ) where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<T, A::K>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.len() < pssm.rows() || rows.len() == 0 {
            scores.resize(0, 0);
            return;
        }

        // FIXME?
        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));

        let result = scores.matrix_mut();
        let matrix = pssm;

        for (res_row, seq_row) in rows.enumerate() {
            for col in 0..C::USIZE {
                let mut score = T::default();
                for (j, pssm_row) in matrix.iter().enumerate() {
                    let symbol = seq.matrix()[seq_row + j][col];
                    score += pssm_row[symbol.as_index()];
                }
                result[res_row][col] = score;
            }
        }
    }

    /// Compute the PSSM scores into the given striped score matrix.
    fn score_into<S, M>(&self, pssm: M, seq: S, scores: &mut StripedScores<T, C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<T, A::K>>,
    {
        let s = seq.as_ref();
        let rows = s.matrix().rows() - s.wrap();
        Self::score_rows_into(&self, pssm, s, 0..rows, scores)
    }

    /// Compute the PSSM scores for every sequence positions.
    fn score<S, M>(&self, pssm: M, seq: S) -> StripedScores<T, C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<T, A::K>>,
    {
        let seq = seq.as_ref();
        let mut scores = StripedScores::empty();
        self.score_into(pssm, seq, &mut scores);
        scores
    }
}

/// Used for finding the highest scoring site in a striped score matrix.
pub trait Maximum<T: MatrixElement + PartialOrd, C: StrictlyPositive> {
    /// Find the matrix coordinates with the highest score.
    fn argmax(&self, scores: &StripedScores<T, C>) -> Option<MatrixCoordinates> {
        if scores.is_empty() {
            return None;
        }

        let mut best_row = 0;
        let mut best_col = 0;
        let mut best_score = scores[0];

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
    fn max(&self, scores: &StripedScores<T, C>) -> Option<T> {
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
pub trait Threshold<T: MatrixElement + PartialOrd, C: StrictlyPositive> {
    /// Return the coordinates of positions with score equal to or greater than the threshold.
    ///
    /// # Note
    /// The indices are not be sorted, and the actual order depends on the
    /// implementation.
    fn threshold(&self, scores: &StripedScores<T, C>, threshold: T) -> Vec<MatrixCoordinates> {
        let mut positions = Vec::new();
        for (i, row) in scores.matrix().iter().enumerate() {
            for col in 0..C::USIZE {
                // assert!(!row[col].is_nan());
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

impl<T: MatrixElement + AddAssign, A: Alphabet, C: StrictlyPositive> Score<T, A, C>
    for Pipeline<A, Generic>
{
}

impl<T: MatrixElement + PartialOrd, A: Alphabet, C: StrictlyPositive> Maximum<T, C>
    for Pipeline<A, Generic>
{
}

impl<A: Alphabet, C: StrictlyPositive> Stripe<A, C> for Pipeline<A, Generic> {}

impl<T: MatrixElement + PartialOrd, A: Alphabet, C: StrictlyPositive> Threshold<T, C>
    for Pipeline<A, Generic>
{
}

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
    #[inline]
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

impl<A: Alphabet> Encode<A> for Pipeline<A, Sse2> {
    #[inline]
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        Sse2::encode_into::<A>(seq.as_ref(), dst)
    }
}

impl<A, C> Score<f32, A, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, C>,
    ) where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<f32, A::K>>,
    {
        Sse2::score_rows_into(pssm, seq, rows, scores)
    }
}

impl<A, C> Score<u8, A, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Maximum<f32, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16> + ArrayLength,
{
    #[inline]
    fn argmax(&self, scores: &StripedScores<f32, C>) -> Option<MatrixCoordinates> {
        Sse2::argmax(scores)
    }
}

impl<A, C> Maximum<u8, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Threshold<f32, C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Threshold<u8, C> for Pipeline<A, Sse2>
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
    #[inline]
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        Avx2::encode_into::<A>(seq.as_ref(), dst)
    }
}

impl Score<f32, Dna, <Avx2 as Backend>::LANES> for Pipeline<Dna, Avx2> {
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, <Dna as Alphabet>::K>>,
    {
        Avx2::score_f32_rows_into_permute(pssm, seq, rows, scores)
    }
}

impl Score<f32, Protein, <Avx2 as Backend>::LANES> for Pipeline<Protein, Avx2> {
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Protein, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, <Protein as Alphabet>::K>>,
    {
        Avx2::score_f32_rows_into_gather(pssm, seq, rows, scores)
    }
}

impl Score<u8, Dna, <Avx2 as Backend>::LANES> for Pipeline<Dna, Avx2> {
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<u8, <Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<u8, <Dna as Alphabet>::K>>,
    {
        Avx2::score_u8_rows_into_shuffle(pssm, seq, rows, scores)
    }
}

impl<A: Alphabet> Stripe<A, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    /// Stripe a sequence into the given striped matrix.
    #[inline]
    fn stripe_into<S: AsRef<[A::Symbol]>>(
        &self,
        seq: S,
        matrix: &mut StripedSequence<A, <Avx2 as Backend>::LANES>,
    ) {
        Avx2::stripe_into(seq, matrix)
    }
}

impl<A: Alphabet> Maximum<f32, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    fn argmax(
        &self,
        scores: &StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        Avx2::argmax_f32(scores)
    }

    fn max(&self, scores: &StripedScores<f32, <Avx2 as Backend>::LANES>) -> Option<f32> {
        Avx2::max_f32(scores)
    }
}

impl<A: Alphabet> Maximum<u8, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    fn argmax(
        &self,
        scores: &StripedScores<u8, <Avx2 as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        Avx2::argmax_u8(scores)
    }

    fn max(&self, scores: &StripedScores<u8, <Avx2 as Backend>::LANES>) -> Option<u8> {
        Avx2::max_u8(scores)
    }
}

impl<A: Alphabet> Threshold<f32, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {}

impl<A: Alphabet> Threshold<u8, <Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {}

// --- NEON pipeline -----------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Neon> {
    /// Attempt to create a new AVX2-accelerated pipeline.
    #[inline]
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
    #[inline]
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        Neon::encode_into::<A>(seq.as_ref(), dst)
    }
}

impl<A, C> Score<f32, A, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, C>,
    ) where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<f32, <A as Alphabet>::K>>,
    {
        Neon::score_f32_rows_into(pssm, seq, rows, scores);
    }
}

impl<C> Score<u8, Dna, C> for Pipeline<Dna, Neon>
where
    C: StrictlyPositive + MultipleOf<U16>,
{
    #[inline]
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<u8, C>,
    ) where
        S: AsRef<StripedSequence<Dna, C>>,
        M: AsRef<DenseMatrix<u8, <Dna as Alphabet>::K>>,
    {
        Neon::score_u8_rows_into(pssm, seq, rows, scores);
    }
}

impl<A, C> Maximum<f32, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Maximum<u8, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Threshold<f32, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

impl<A, C> Threshold<u8, C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + MultipleOf<U16>,
{
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod test {
    use std::str::FromStr;
    use typenum::consts::U4;

    use super::*;

    use crate::abc::Dna;
    use crate::pwm::CountMatrix;

    #[test]
    fn score_rows_into_empty() {
        let pli = Pipeline::generic();

        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let mut striped = <Pipeline<_, _> as Stripe<Dna, U4>>::stripe(&pli, seq);

        let cm = CountMatrix::<Dna>::from_sequences(
            ["ATTA", "ATTC"]
                .iter()
                .map(|x| EncodedSequence::encode(x).unwrap()),
        )
        .unwrap();
        let pbm = cm.to_freq(0.1);
        let pwm = pbm.to_weight(None);
        let pssm = pwm.to_scoring();

        striped.configure(&pssm);
        let mut scores = StripedScores::empty();
        pli.score_rows_into(pssm, striped, 1..1, &mut scores);
    }
}
