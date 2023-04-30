#![doc = include_str!("../README.md")]

mod abc;
mod dense;
mod pli;
mod pwm;
mod seq;

pub use abc::Alphabet;
pub use abc::Dna;
pub use abc::Nucleotide;
pub use abc::Symbol;
pub use dense::DenseMatrix;
pub use pli::Pipeline;
pub use pli::StripedScores;
pub use pwm::Background;
pub use pwm::CountMatrix;
pub use pwm::ProbabilityMatrix;
pub use pwm::WeightMatrix;
pub use seq::EncodedSequence;
pub use seq::StripedSequence;
