mod avx2;
mod generic;
mod sse2;

pub use self::avx2::Avx2;
pub use self::generic::Generic;
pub use self::sse2::Sse2;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

/// A marker trait for backends.
pub trait Backend {
    type LANES: Unsigned + NonZero;
}

/// An error marker when a pipeline backend is unsupported on the host platform.
#[derive(Debug, Clone)]
pub struct UnsupportedBackend;
