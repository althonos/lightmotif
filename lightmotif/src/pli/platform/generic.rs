use typenum::consts::U1;
use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

use super::Backend;
use crate::abc::Alphabet;

use crate::pli::BestPosition;

use crate::pli::Score;

/// A marker type for the generic implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Generic;

impl Backend for Generic {
    type LANES = U1;
}

impl<A: Alphabet, C: NonZero + Unsigned> Score<A, C> for Generic {}

impl<C: NonZero + Unsigned> BestPosition<C> for Generic {}
