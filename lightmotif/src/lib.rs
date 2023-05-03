#![doc = include_str!("../README.md")]

mod abc;
mod dense;
mod err;
mod pli;
mod pwm;
mod seq;

pub use abc::Alphabet;
pub use abc::Background;
pub use abc::Dna;
pub use abc::Nucleotide;
pub use abc::Pseudocounts;
pub use abc::Symbol;
pub use dense::DenseMatrix;
pub use err::InvalidSymbol;
pub use pli::Pipeline;
pub use pli::Score;
pub use pli::StripedScores;
pub use pwm::CountMatrix;
pub use pwm::FrequencyMatrix;
pub use pwm::ScoringMatrix;
pub use pwm::WeightMatrix;
pub use seq::EncodedSequence;
pub use seq::StripedSequence;
