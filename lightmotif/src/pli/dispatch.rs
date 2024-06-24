#![allow(unused)]

use std::ops::Range;

use super::platform::Avx2;
use super::platform::Generic;
use super::platform::Neon;
use super::platform::Sse2;
use super::Backend;
use super::Encode;
use super::Maximum;
use super::Pipeline;
use super::Score;
use super::Stripe;
use super::Threshold;
use crate::abc::Alphabet;
use crate::abc::Dna;
use crate::abc::Protein;
use crate::dense::DenseMatrix;
use crate::dense::MatrixCoordinates;
use crate::dense::MatrixElement;
use crate::err::InvalidSymbol;
#[allow(unused)]
use crate::num::U1;
use crate::pwm::ScoringMatrix;
use crate::scores::StripedScores;
use crate::seq::StripedSequence;

/// A dynamic dispatcher that selects the best available pipeline backend at runtime.
#[derive(Clone, Debug)]
pub enum Dispatch {
    Generic,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon,
}

impl Backend for Dispatch {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    type LANES = <Avx2 as Backend>::LANES;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    type LANES = <Neon as Backend>::LANES;
    #[cfg(not(any(
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    type LANES = U1;
}

impl Dispatch {
    pub fn new(&self) -> Self {
        Self::Generic
    }
}

impl<A: Alphabet> Encode<A> for Pipeline<A, Dispatch> {
    fn encode_into<S: AsRef<[u8]>>(
        &self,
        seq: S,
        dst: &mut [A::Symbol],
    ) -> Result<(), InvalidSymbol> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::encode_into::<A>(seq.as_ref(), dst),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::encode_into::<A>(seq.as_ref(), dst),
            _ => <Generic as Encode<A>>::encode_into::<&[u8]>(&Generic, seq.as_ref(), dst),
        }
    }
}

impl Score<f32, Dna, <Dispatch as Backend>::LANES> for Pipeline<Dna, Dispatch> {
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Dispatch as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Dispatch as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, <Dna as Alphabet>::K>>,
    {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::score_f32_rows_into_permute(pssm, seq.as_ref(), rows, scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::score_rows_into(pssm, seq.as_ref(), rows, scores),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::score_f32_rows_into(pssm, seq.as_ref(), rows, scores),
            _ => <Generic as Score<f32, Dna, <Dispatch as Backend>::LANES>>::score_rows_into(
                &Generic,
                pssm,
                seq.as_ref(),
                rows,
                scores,
            ),
        }
    }
}

impl Score<f32, Protein, <Dispatch as Backend>::LANES> for Pipeline<Protein, Dispatch> {
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Dispatch as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Protein, <Dispatch as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, <Protein as Alphabet>::K>>,
    {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::score_f32_rows_into_gather(pssm, seq.as_ref(), rows, scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::score_rows_into(pssm, seq.as_ref(), rows, scores),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::score_f32_rows_into(pssm, seq.as_ref(), rows, scores),
            _ => <Generic as Score<f32, Protein, <Dispatch as Backend>::LANES>>::score_rows_into(
                &Generic,
                pssm,
                seq.as_ref(),
                rows,
                scores,
            ),
        }
    }
}

impl Score<u8, Dna, <Dispatch as Backend>::LANES> for Pipeline<Dna, Dispatch> {
    fn score_rows_into<S, M>(
        &self,
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<u8, <Dispatch as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Dispatch as Backend>::LANES>>,
        M: AsRef<DenseMatrix<u8, <Dna as Alphabet>::K>>,
    {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::score_u8_rows_into_shuffle(pssm, seq.as_ref(), rows, scores),
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // Dispatch::Sse2 => Sse2::score_rows_into(pssm, seq.as_ref(), rows, scores),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::score_u8_rows_into(pssm, seq.as_ref(), rows, scores),
            _ => <Generic as Score<u8, Dna, <Dispatch as Backend>::LANES>>::score_rows_into(
                &Generic,
                pssm,
                seq.as_ref(),
                rows,
                scores,
            ),
        }
    }
}

impl<A: Alphabet> Stripe<A, <Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {
    fn stripe_into<S: AsRef<[A::Symbol]>>(
        &self,
        seq: S,
        matrix: &mut StripedSequence<A, <Dispatch as Backend>::LANES>,
    ) {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::stripe_into(seq, matrix),
            _ => <Generic as Stripe<A, <Dispatch as Backend>::LANES>>::stripe_into(
                &Generic, seq, matrix,
            ),
        }
    }
}

impl<A: Alphabet> Maximum<f32, <Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {
    fn argmax(
        &self,
        scores: &StripedScores<f32, <Dispatch as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::argmax_f32(scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::argmax(scores),
            _ => <Generic as Maximum<f32, <Dispatch as Backend>::LANES>>::argmax(&Generic, scores),
        }
    }

    fn max(&self, scores: &StripedScores<f32, <Dispatch as Backend>::LANES>) -> Option<f32> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::max_f32(scores),
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // Dispatch::Sse2 => Sse2::argmax(scores),
            _ => <Generic as Maximum<f32, <Dispatch as Backend>::LANES>>::max(&Generic, scores),
        }
    }
}

impl<A: Alphabet> Maximum<u8, <Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {
    fn argmax(
        &self,
        scores: &StripedScores<u8, <Dispatch as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::argmax_u8(scores),
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // Dispatch::Sse2 => Sse2::argmax(scores),
            _ => <Generic as Maximum<u8, <Dispatch as Backend>::LANES>>::argmax(&Generic, scores),
        }
    }

    fn max(&self, scores: &StripedScores<u8, <Dispatch as Backend>::LANES>) -> Option<u8> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::max_u8(scores),
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // Dispatch::Sse2 => Sse2::argmax(scores),
            _ => <Generic as Maximum<u8, <Dispatch as Backend>::LANES>>::max(&Generic, scores),
        }
    }
}

impl<A: Alphabet> Threshold<f32, <Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {}

impl<A: Alphabet> Threshold<u8, <Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {}
