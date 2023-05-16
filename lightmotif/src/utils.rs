use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

/// A marker trait for type numbers that are strictly positive.
pub trait StrictlyPositive: Unsigned + NonZero {}

impl<N: Unsigned + NonZero> StrictlyPositive for N {}
