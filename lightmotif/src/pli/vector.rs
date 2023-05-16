use typenum::consts::U1;
use typenum::consts::U16;
use typenum::consts::U32;
use typenum::consts::U4;
use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

/// Trait for concrete vector implementations.
///
/// The trait is defined for the loading vector type, which has `LANES`
/// lanes of `Item` values.
pub trait Vector {
    type Item;
    type LANES: Unsigned + NonZero;
}

impl Vector for u8 {
    type Item = u8;
    type LANES = U1;
}

impl Vector for f32 {
    type Item = f32;
    type LANES = U1;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Vector for std::arch::x86_64::__m128i {
    type Item = u8;
    type LANES = U16;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Vector for std::arch::x86_64::__m256i {
    type Item = u8;
    type LANES = U32;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Vector for std::arch::x86_64::__m128 {
    type Item = f32;
    type LANES = U4;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Vector for std::arch::x86_64::__m256 {
    type Item = f32;
    type LANES = U4;
}
