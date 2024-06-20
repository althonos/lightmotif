use typenum::consts::U1;

use super::Backend;
use crate::abc::Alphabet;
use crate::dense::MatrixElement;
use crate::num::StrictlyPositive;

use crate::pli::Encode;
use crate::pli::Maximum;
use crate::pli::Score;
use crate::pli::Stripe;
use crate::pli::Threshold;

/// A marker type for the generic implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Generic;

impl Backend for Generic {
    type LANES = U1;
}

impl<A: Alphabet> Encode<A> for Generic {}

impl<A: Alphabet, C: StrictlyPositive> Score<A, C> for Generic {}

impl<T: MatrixElement + PartialOrd, C: StrictlyPositive> Maximum<T, C> for Generic {}

impl<A: Alphabet, C: StrictlyPositive> Stripe<A, C> for Generic {}

impl<T: MatrixElement + PartialOrd, C: StrictlyPositive> Threshold<T, C> for Generic {}
