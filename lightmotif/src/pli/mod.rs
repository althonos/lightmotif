//! Concrete implementations of the sequence scoring pipeline.

use std::ops::Div;
use std::ops::Rem;

pub use self::scores::StripedScores;

use self::platform::Avx2;
use self::platform::Backend;
use self::platform::Generic;
use self::platform::Neon;
use self::platform::Sse2;
use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::err::UnsupportedBackend;
use super::num::StrictlyPositive;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;

use typenum::consts::U16;
use typenum::marker_traits::Unsigned;
use typenum::marker_traits::Zero;

pub mod platform;
mod scores;

// --- Score -------------------------------------------------------------------

/// Generic trait for computing sequence scores with a PSSM.
pub trait Score<A: Alphabet, C: StrictlyPositive> {
    /// Compute the PSSM scores into the given striped score matrix.
    fn score_into<S, M>(&self, seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

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

/// Generic trait for finding the highest scoring site in a striped score matrix.
pub trait BestPosition<C: StrictlyPositive> {
    /// Find the sequence position with the highest score.
    fn best_position(&self, scores: &StripedScores<C>) -> Option<usize> {
        if scores.len() == 0 {
            return None;
        }

        let data = scores.matrix();
        let mut best_pos = 0;
        let mut best_score = data[0][0];
        for i in 0..scores.len() {
            let col = i / data.rows();
            let row = i % data.rows();
            if data[row][col] > best_score {
                best_score = data[row][col];
                best_pos = i;
            }
        }

        Some(best_pos)
    }
}

// --- Pipeline ----------------------------------------------------------------

/// Wrapper implementing score computation for different platforms.
#[derive(Debug, Default, Clone)]
pub struct Pipeline<A: Alphabet, B: Backend> {
    alphabet: std::marker::PhantomData<A>,
    backend: std::marker::PhantomData<B>,
}

// --- Generic pipeline --------------------------------------------------------

impl<A: Alphabet> Pipeline<A, Generic> {
    /// Create a new generic pipeline.
    pub const fn generic() -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            backend: std::marker::PhantomData,
        }
    }
}

impl<A: Alphabet, C: StrictlyPositive> Score<A, C> for Pipeline<A, Generic> {}

impl<A: Alphabet, C: StrictlyPositive> BestPosition<C> for Pipeline<A, Generic> {}

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

impl<A: Alphabet> BestPosition<<Sse2 as Backend>::LANES> for Pipeline<A, Sse2> {
    fn best_position(&self, scores: &StripedScores<<Sse2 as Backend>::LANES>) -> Option<usize> {
        Sse2::best_position(scores)
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
        Avx2::score_into(seq, pssm, scores)
    }
}

impl<A: Alphabet> BestPosition<<Avx2 as Backend>::LANES> for Pipeline<A, Avx2> {
    fn best_position(&self, scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
        Avx2::best_position(scores)
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

impl<A, C> BestPosition<C> for Pipeline<A, Neon>
where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
}
