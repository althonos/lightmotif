//! Concrete implementations of the sequence scoring pipeline.

use std::ops::Div;
use std::ops::Rem;

pub use self::scores::StripedScores;

use self::dispatch::Dispatch;
use self::platform::Avx2;
use self::platform::Backend;
use self::platform::Generic;
use self::platform::Neon;
use self::platform::Sse2;
use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Protein;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::err::InvalidSymbol;
use super::err::UnsupportedBackend;
use super::num::StrictlyPositive;
use super::pwm::ScoringMatrix;
use super::seq::EncodedSequence;
use super::seq::StripedSequence;

use typenum::consts::U16;
use typenum::marker_traits::Unsigned;
use typenum::marker_traits::Zero;

pub mod dispatch;
pub mod platform;
mod scores;

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
    fn score_into<S, M>(&self, seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.length < pssm.len() {
            scores.resize(0, 0);
            return;
        }

        let seq_rows = seq.data.rows() - seq.wrap;
        scores.resize(seq.length - pssm.len() + 1, seq_rows);

        let result = scores.matrix_mut();
        for i in 0..seq.length - pssm.len() + 1 {
            let mut score = 0.0;
            for j in 0..pssm.len() {
                let offset = i + j;
                let col = offset / seq_rows;
                let row = offset % seq_rows;
                score += pssm.weights()[j][seq.data[row][col].as_index()];
            }
            let col = i / result.rows();
            let row = i % result.rows();
            result[row][col] = score;
        }
    }

    /// Compute the PSSM scores for every sequence positions.
    fn score<S, M>(&self, seq: S, pssm: M) -> StripedScores<C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        let data = unsafe { DenseMatrix::uninitialized(seq.data.rows() - seq.wrap) };
        let length = seq.length - pssm.len() + 1;
        let mut scores = StripedScores::new(length, data);
        self.score_into(seq, pssm, &mut scores);
        scores
    }
}

/// Used for finding the highest scoring site in a striped score matrix.
pub trait Maximum<C: StrictlyPositive> {
    /// Find the sequence position with the highest score.
    fn argmax(&self, scores: &StripedScores<C>) -> Option<usize> {
        if scores.len() == 0 {
            return None;
        }

        let mut best_pos = 0;
        let mut best_score = scores[0];
        for i in 1..scores.len() {
            if scores[i] > best_score {
                best_score = scores[i];
                best_pos = i;
            }
        }

        Some(best_pos)
    }

    /// Find the highest score.
    fn max(&self, scores: &StripedScores<C>) -> Option<f32> {
        self.argmax(scores).map(|i| scores[i])
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
    fn stripe_into<S: AsRef<[A::Symbol]>>(&self, seq: S, matrix: &mut StripedSequence<A, C>) {
        let s = seq.as_ref();
        let length = s.len();
        let rows = (length + (C::USIZE - 1)) / C::USIZE;
        matrix.data.resize(rows);
        matrix.length = length;
        matrix.wrap = 0;
        let data = &mut matrix.data;
        for (i, &x) in s.iter().enumerate() {
            data[i % rows][i / rows] = x;
        }
        for i in s.len()..data.rows() * data.columns() {
            data[i % rows][i / rows] = A::default_symbol();
        }
    }
}

/// Used for finding positions above a score threshold in a striped score matrix.
pub trait Threshold<C: StrictlyPositive> {
    /// Return the indices of positions with score equal to or greater than the threshold.
    ///
    /// # Note
    ///
    /// The indices may or may not be sorted, depending on the implementation.
    fn threshold(&self, scores: &StripedScores<C>, threshold: f32) -> Vec<usize> {
        let mut positions = Vec::new();
        for i in 0..scores.len() {
            if scores[i] >= threshold {
                positions.push(i);
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
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    fn score_into<S, M>(&self, seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        Sse2::score_into(seq, pssm, scores)
    }
}

impl<A, C> Maximum<C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    fn argmax(&self, scores: &StripedScores<C>) -> Option<usize> {
        Sse2::argmax(scores)
    }
}

impl<A: Alphabet, C: StrictlyPositive> Threshold<C> for Pipeline<A, Sse2>
where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    fn threshold(&self, scores: &StripedScores<C>, threshold: f32) -> Vec<usize> {
        Sse2::threshold(scores, threshold)
    }
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
    fn score_into<S, M>(
        &self,
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Avx2 as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Dna>>,
    {
        Avx2::score_into_permute(seq, pssm, scores)
    }
}

impl Score<Protein, <Avx2 as Backend>::LANES> for Pipeline<Protein, Avx2> {
    fn score_into<S, M>(
        &self,
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Protein, <Avx2 as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Protein>>,
    {
        Avx2::score_into_gather(seq, pssm, scores)
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
    fn argmax(&self, scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
        Avx2::argmax(scores)
    }
}

impl<A: Alphabet> Threshold<<Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    fn threshold(
        &self,
        scores: &StripedScores<<Avx2 as Backend>::LANES>,
        threshold: f32,
    ) -> Vec<usize> {
        Avx2::threshold(scores, threshold)
    }
}

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
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    fn score_into<S, M>(&self, seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        Neon::score_into(seq, pssm, scores)
    }
}

impl<A, C> Maximum<C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
}

impl<A, C> Threshold<C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    fn threshold(&self, scores: &StripedScores<C>, threshold: f32) -> Vec<usize> {
        Neon::threshold(scores, threshold)
    }
}
