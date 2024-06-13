#![doc = include_str!("../README.md")]

extern crate generic_array;
extern crate typenum;

pub mod abc;
pub mod dense;
pub mod err;
pub mod num;
pub mod pli;
pub mod pwm;
pub mod scan;
pub mod scores;
pub mod seq;

pub use abc::Alphabet;
pub use abc::Dna;
pub use abc::Protein;
pub use pwm::CountMatrix;
pub use pwm::FrequencyMatrix;
pub use pwm::ScoringMatrix;
pub use pwm::WeightMatrix;
pub use seq::EncodedSequence;
pub use seq::StripedSequence;
