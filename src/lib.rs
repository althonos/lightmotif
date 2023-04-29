//! Fast position-weight matrices using sequence striping and SIMD.

extern crate nom;

pub mod io;

mod abc;
mod dense;
mod pli;
mod pwm;
mod seq;

pub use abc::Alphabet;
pub use abc::DnaAlphabet;
pub use abc::DnaSymbol;
pub use abc::Symbol;
pub use dense::DenseMatrix;
pub use pli::Pipeline;
pub use pwm::Background;
pub use pwm::CountMatrix;
pub use pwm::ProbabilityMatrix;
pub use pwm::WeightMatrix;
pub use pwm::StripedScores;
pub use seq::EncodedSequence;
pub use seq::StripedSequence;
