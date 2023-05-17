//! Re-exports for the `typenum` crate, with extra utilities.

#[doc(no_inline)]
pub use typenum::*;

/// A marker trait for type numbers that are strictly positive.
pub trait StrictlyPositive: Unsigned + NonZero {}

impl<N: Unsigned + NonZero> StrictlyPositive for N {}
