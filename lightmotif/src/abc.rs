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

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum DnaSymbol {
    A = 0,
    C = 1,
    T = 2,
    G = 3,
    N = 4,
}

impl Symbol for DnaSymbol {
    fn as_index(&self) -> usize {
        *self as usize
    }
}

impl TryFrom<char> for DnaSymbol {
    type Error = InvalidSymbol;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'A' => Ok(DnaSymbol::A),
            'C' => Ok(DnaSymbol::C),
            'T' => Ok(DnaSymbol::T),
            'G' => Ok(DnaSymbol::G),
            'N' => Ok(DnaSymbol::N),
            _ => Err(InvalidSymbol(c)),
        }
    }
}

impl Default for DnaSymbol {
    fn default() -> DnaSymbol {
        DnaSymbol::N
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DnaAlphabet;

impl Alphabet for DnaAlphabet {
    type Symbol = DnaSymbol;
    const K: usize = 5;
}
