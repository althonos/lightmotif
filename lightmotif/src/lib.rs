#![doc = include_str!("../README.md")]

extern crate generic_array;
extern crate typenum;

pub mod abc;
pub mod dense;
pub mod err;
pub mod pli;
pub mod pwm;
pub mod seq;

pub use abc::Alphabet;
pub use abc::AminoAcid;
pub use abc::Background;
pub use abc::ComplementableAlphabet;
pub use abc::ComplementableSymbol;
pub use abc::Dna;
pub use abc::Nucleotide;
pub use abc::Protein;
pub use abc::Pseudocounts;
pub use abc::Symbol;
pub use dense::DenseMatrix;
pub use err::InvalidSymbol;
pub use pli::BestPosition;
pub use pli::Pipeline;
pub use pli::Score;
pub use pli::StripedScores;
pub use pwm::CountMatrix;
pub use pwm::FrequencyMatrix;
pub use pwm::ScoringMatrix;
pub use pwm::WeightMatrix;
pub use seq::EncodedSequence;
pub use seq::StripedSequence;
