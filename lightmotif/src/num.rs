//! Re-exports for the `typenum` crate, with extra utilities.

use std::ops::Div;
use std::ops::Rem;

#[doc(no_inline)]
pub use typenum::*;

/// A marker trait for type numbers that are strictly positive.
pub trait StrictlyPositive: Unsigned + NonZero {}

impl<N: Unsigned + NonZero> StrictlyPositive for N {}

/// A marker trait for type numbers that are multiple of another number.
pub trait MultipleOf<N: StrictlyPositive>: StrictlyPositive + Div<N> + Rem<N> {
    type Quotient: Unsigned;
}

impl<M, N> MultipleOf<N> for M
where
    M: StrictlyPositive + Rem<N> + Div<N>,
    N: StrictlyPositive,
    <M as Rem<N>>::Output: Zero,
    <M as Div<N>>::Output: Unsigned,
{
    type Quotient = <Self as Div<N>>::Output;
}
