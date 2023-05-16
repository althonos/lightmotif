use typenum::consts::U1;
use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

use super::Backend;
use crate::abc::Alphabet;
use crate::abc::Dna;
use crate::abc::Nucleotide;
use crate::abc::Symbol;
use crate::pli::scores::StripedScores;
use crate::pli::BestPosition;
use crate::pli::Pipeline;
use crate::pli::Score;
use crate::pli::Vector;
use crate::pwm::ScoringMatrix;
use crate::seq::StripedSequence;

/// A marker type for the generic implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Generic;

impl Backend for Generic {
    type LANES = U1;
}

impl<A: Alphabet, C: NonZero + Unsigned> Score<A, C> for Generic {}

impl<C: NonZero + Unsigned> BestPosition<C> for Generic {}
