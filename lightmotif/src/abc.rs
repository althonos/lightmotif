use std::convert::TryFrom;
use std::fmt::Debug;

#[derive(Debug)]
pub struct InvalidSymbol(char);

/// Common traits for a biological symbol.
pub trait Symbol: Default + Sized + Copy + TryFrom<char> {
    fn as_index(&self) -> usize;
}

/// Common traits for a biological alphabet.
pub trait Alphabet: Debug + Copy + Default + 'static {
    type Symbol: Symbol;
    const K: usize;

    /// Get the default symbol for this alphabet.
    fn default_symbol() -> Self::Symbol {
        Default::default()
    }
}

/// A nucleobase
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum Nucleotide {
    /// Adenine.
    A = 0,
    /// Cytosine.
    C = 1,
    /// Thymine.
    T = 2,
    /// Guanine.
    G = 3,
    /// Unknown base.
    N = 4,
}

impl Symbol for Nucleotide {
    fn as_index(&self) -> usize {
        *self as usize
    }
}

impl TryFrom<char> for Nucleotide {
    type Error = InvalidSymbol;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'A' => Ok(Nucleotide::A),
            'C' => Ok(Nucleotide::C),
            'T' => Ok(Nucleotide::T),
            'G' => Ok(Nucleotide::G),
            'N' => Ok(Nucleotide::N),
            _ => Err(InvalidSymbol(c)),
        }
    }
}

impl Default for Nucleotide {
    fn default() -> Nucleotide {
        Nucleotide::N
    }
}

/// The standard DNA alphabet composed of 4 nucleobases and a wildcard.
#[derive(Default, Debug, Clone, Copy)]
pub struct Dna;

impl Alphabet for Dna {
    type Symbol = Nucleotide;
    const K: usize = 5;
}
