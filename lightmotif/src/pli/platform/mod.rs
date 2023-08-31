//! Platform-specific code for the scoring pipeline.
#![allow(unused)]

mod avx2;
mod generic;
mod neon;
mod sse2;

pub use self::avx2::Avx2;
pub use self::generic::Generic;
pub use self::neon::Neon;
pub use self::sse2::Sse2;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

/// A marker trait for backends.
pub trait Backend {
    type LANES: Unsigned + NonZero;
}
