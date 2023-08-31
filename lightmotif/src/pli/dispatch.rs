#![allow(unused)]

use super::platform::Avx2;
use super::platform::Generic;
use super::platform::Neon;
use super::platform::Sse2;
use super::scores::StripedScores;
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
use crate::err::InvalidSymbol;
#[allow(unused)]
use crate::num::U1;
use crate::pwm::ScoringMatrix;
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

impl Score<Dna, <Dispatch as Backend>::LANES> for Pipeline<Dna, Dispatch> {
    fn score_into<S, M>(
        &self,
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Dispatch as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Dna, <Dispatch as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Dna>>,
    {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::score_into_permute(seq.as_ref(), pssm, scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::score_into(seq.as_ref(), pssm, scores),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::score_into(seq.as_ref(), pssm, scores),
            _ => <Generic as Score<Dna, <Dispatch as Backend>::LANES>>::score_into(
                &Generic,
                seq.as_ref(),
                pssm,
                scores,
            ),
        }
    }
}

impl Score<Protein, <Dispatch as Backend>::LANES> for Pipeline<Protein, Dispatch> {
    fn score_into<S, M>(
        &self,
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Dispatch as Backend>::LANES>,
    ) where
        S: AsRef<StripedSequence<Protein, <Dispatch as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Protein>>,
    {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::score_into_gather(seq.as_ref(), pssm, scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::score_into(seq.as_ref(), pssm, scores),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::score_into(seq.as_ref(), pssm, scores),
            _ => <Generic as Score<Protein, <Dispatch as Backend>::LANES>>::score_into(
                &Generic,
                seq.as_ref(),
                pssm,
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

impl<A: Alphabet> Maximum<<Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {
    fn argmax(&self, scores: &StripedScores<<Dispatch as Backend>::LANES>) -> Option<usize> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::argmax(scores),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::argmax(scores),
            _ => <Generic as Maximum<<Dispatch as Backend>::LANES>>::argmax(&Generic, scores),
        }
    }
}

impl<A: Alphabet> Threshold<<Dispatch as Backend>::LANES> for Pipeline<A, Dispatch> {
    fn threshold(
        &self,
        scores: &StripedScores<<Dispatch as Backend>::LANES>,
        threshold: f32,
    ) -> Vec<usize> {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Avx2 => Avx2::threshold(scores, threshold),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Dispatch::Sse2 => Sse2::threshold(scores, threshold),
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            Dispatch::Neon => Neon::threshold(scores, threshold),
            _ => <Generic as Threshold<<Dispatch as Backend>::LANES>>::threshold(
                &Generic, scores, threshold,
            ),
        }
    }
}
