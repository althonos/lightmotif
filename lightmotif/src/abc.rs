//! Digital encoding for biological sequences using an alphabet.

use std::fmt::Debug;

use generic_array::ArrayLength;
use generic_array::GenericArray;
use typenum::consts::U21;
use typenum::consts::U5;
use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

use super::err::InvalidData;
use super::err::InvalidSymbol;

// --- Symbol ------------------------------------------------------------------

/// A symbol from a biological alphabet.
pub trait Symbol: Default + Sized + Copy {
    /// View this symbol as a zero-based index.
    fn as_index(&self) -> usize;
    /// View this symbol as a string character.
    fn as_char(&self) -> char;
    /// Parse a string character into a symbol.
    fn from_char(c: char) -> Result<Self, InvalidSymbol>;
}

/// A symbol that can be complemented.
pub trait ComplementableSymbol: Symbol {
    /// Get the complement of this symbol.
    fn complement(&self) -> Self;
}

// --- Alphabet ----------------------------------------------------------------

/// A biological alphabet with associated metadata.
pub trait Alphabet: Debug + Copy + Default + 'static {
    type Symbol: Symbol;
    type K: Unsigned + NonZero + ArrayLength<f32>;

    /// Get the default symbol for this alphabet.
    fn default_symbol() -> Self::Symbol {
        Default::default()
    }

    /// Get all the symbols of this alphabet.
    fn symbols() -> &'static [Self::Symbol];
}

// --- ComplementableAlphabet --------------------------------------------------

/// An alphabet that defines the complement operation.
pub trait ComplementableAlphabet: Alphabet {
    /// Get the complement of this symbol.
    fn complement(s: Self::Symbol) -> Self::Symbol;
}

impl<A: Alphabet> ComplementableAlphabet for A
where
    <A as Alphabet>::Symbol: ComplementableSymbol,
{
    fn complement(s: Self::Symbol) -> Self::Symbol {
        s.complement()
    }
}

// --- DNA ---------------------------------------------------------------------

/// The standard DNA alphabet composed of 4 deoxyribonucleotides and a wildcard.
#[derive(Default, Debug, Clone, Copy)]
pub struct Dna;

impl Alphabet for Dna {
    type Symbol = Nucleotide;
    type K = U5;

    fn symbols() -> &'static [Nucleotide] {
        &[
            Nucleotide::A,
            Nucleotide::C,
            Nucleotide::T,
            Nucleotide::G,
            Nucleotide::N,
        ]
    }
}

/// A deoxyribonucleotide.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
pub enum Nucleotide {
    /// Adenine.
    ///
    /// ![adenine.png](https://www.ebi.ac.uk/chebi/displayImage.do?defaultImage=true&imageIndex=0&chebiId=16708)
    A = 0,
    /// Cytosine.
    ///
    /// ![cytosine.png](https://www.ebi.ac.uk/chebi/displayImage.do?defaultImage=true&imageIndex=0&chebiId=16040)
    C = 1,
    /// Thymine.
    ///
    /// ![thymine.png](https://www.ebi.ac.uk/chebi/displayImage.do?defaultImage=true&imageIndex=0&chebiId=17821)
    T = 2,
    /// Guanine.
    ///
    /// ![guanine.png](https://www.ebi.ac.uk/chebi/displayImage.do?defaultImage=true&imageIndex=0&chebiId=16235)
    G = 3,
    /// Unknown base.
    #[default]
    N = 4,
}

impl From<Nucleotide> for char {
    fn from(n: Nucleotide) -> char {
        n.as_char()
    }
}

impl Symbol for Nucleotide {
    fn as_index(&self) -> usize {
        *self as usize
    }

    fn as_char(&self) -> char {
        match self {
            Nucleotide::A => 'A',
            Nucleotide::C => 'C',
            Nucleotide::T => 'T',
            Nucleotide::G => 'G',
            Nucleotide::N => 'N',
        }
    }

    fn from_char(c: char) -> Result<Self, InvalidSymbol> {
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

impl ComplementableSymbol for Nucleotide {
    fn complement(&self) -> Self {
        match *self {
            Nucleotide::A => Nucleotide::T,
            Nucleotide::T => Nucleotide::A,
            Nucleotide::G => Nucleotide::C,
            Nucleotide::C => Nucleotide::G,
            Nucleotide::N => Nucleotide::N,
        }
    }
}

// --- Protein -----------------------------------------------------------------

/// The standard protein alphabet composed of 20 residues and a wildcard.
#[derive(Default, Debug, Clone, Copy)]
pub struct Protein;

impl Alphabet for Protein {
    type Symbol = AminoAcid;
    type K = U21;

    fn symbols() -> &'static [AminoAcid] {
        &[
            AminoAcid::A,
            AminoAcid::C,
            AminoAcid::D,
            AminoAcid::E,
            AminoAcid::F,
            AminoAcid::G,
            AminoAcid::H,
            AminoAcid::I,
            AminoAcid::K,
            AminoAcid::L,
            AminoAcid::M,
            AminoAcid::N,
            AminoAcid::P,
            AminoAcid::Q,
            AminoAcid::R,
            AminoAcid::S,
            AminoAcid::T,
            AminoAcid::V,
            AminoAcid::W,
            AminoAcid::Y,
            AminoAcid::X,
        ]
    }
}

/// A proteinogenic amino acid.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[repr(u8)]
pub enum AminoAcid {
    A = 0,
    C = 1,
    D = 2,
    E = 3,
    F = 4,
    G = 5,
    H = 6,
    I = 7,
    K = 8,
    L = 9,
    M = 10,
    N = 11,
    P = 12,
    Q = 13,
    R = 14,
    S = 15,
    T = 16,
    V = 17,
    W = 18,
    Y = 19,
    #[default]
    X = 20,
}

impl From<AminoAcid> for char {
    fn from(aa: AminoAcid) -> char {
        aa.as_char()
    }
}

impl Symbol for AminoAcid {
    fn as_index(&self) -> usize {
        *self as usize
    }

    fn as_char(&self) -> char {
        match self {
            AminoAcid::A => 'A',
            AminoAcid::C => 'C',
            AminoAcid::D => 'D',
            AminoAcid::E => 'E',
            AminoAcid::F => 'F',
            AminoAcid::G => 'G',
            AminoAcid::H => 'H',
            AminoAcid::I => 'I',
            AminoAcid::K => 'K',
            AminoAcid::L => 'L',
            AminoAcid::M => 'M',
            AminoAcid::N => 'N',
            AminoAcid::P => 'P',
            AminoAcid::Q => 'Q',
            AminoAcid::R => 'R',
            AminoAcid::S => 'S',
            AminoAcid::T => 'T',
            AminoAcid::V => 'V',
            AminoAcid::W => 'W',
            AminoAcid::Y => 'Y',
            AminoAcid::X => 'X',
        }
    }

    fn from_char(c: char) -> Result<Self, InvalidSymbol> {
        match c {
            'A' => Ok(AminoAcid::A),
            'C' => Ok(AminoAcid::C),
            'D' => Ok(AminoAcid::D),
            'E' => Ok(AminoAcid::E),
            'F' => Ok(AminoAcid::F),
            'G' => Ok(AminoAcid::G),
            'H' => Ok(AminoAcid::H),
            'I' => Ok(AminoAcid::I),
            'K' => Ok(AminoAcid::K),
            'L' => Ok(AminoAcid::L),
            'M' => Ok(AminoAcid::M),
            'N' => Ok(AminoAcid::N),
            'P' => Ok(AminoAcid::P),
            'Q' => Ok(AminoAcid::Q),
            'R' => Ok(AminoAcid::R),
            'S' => Ok(AminoAcid::S),
            'T' => Ok(AminoAcid::T),
            'V' => Ok(AminoAcid::V),
            'W' => Ok(AminoAcid::W),
            'Y' => Ok(AminoAcid::Y),
            'X' => Ok(AminoAcid::X),
            _ => Err(InvalidSymbol(c)),
        }
    }
}

// --- Background --------------------------------------------------------------

/// The background frequencies for an alphabet.
#[derive(Clone, Debug)]
pub struct Background<A: Alphabet> {
    frequencies: GenericArray<f32, A::K>,
    alphabet: std::marker::PhantomData<A>,
}

impl<A: Alphabet> Background<A> {
    /// Create a new background with the given frequencies.
    ///
    /// The array must contain valid frequencies, i.e. real numbers between
    /// zero and one that sum to one.
    pub fn new<F>(frequencies: F) -> Result<Self, InvalidData>
    where
        F: Into<GenericArray<f32, A::K>>,
    {
        let frequencies = frequencies.into();
        let mut sum = 0.0;
        for &f in frequencies.iter() {
            if !(0.0..=1.0).contains(&f) {
                return Err(InvalidData);
            }
            sum += f;
        }
        if sum != 1.0 {
            return Err(InvalidData);
        }
        Ok(Self {
            frequencies,
            alphabet: std::marker::PhantomData,
        })
    }

    /// Create a new background with uniform frequencies.
    ///
    /// The non-default symbols from the alphabet `A` will be initialized
    /// with a frequency of `1/(K-1)`, while the default symbol will remain
    /// with a zero frequency.
    ///
    /// # Note
    /// The `Default` implementation for `Background` uses uniform frequencies.
    ///
    /// # Example
    /// ```
    /// # use lightmotif::abc::*;
    /// let bg = Background::<Dna>::uniform();
    /// assert_eq!(bg.frequencies(), &[0.25, 0.25, 0.25, 0.25, 0.0]);
    /// ```
    pub fn uniform() -> Self {
        let frequencies = (0..A::K::USIZE)
            .map(|i| {
                if i != A::default_symbol().as_index() {
                    1.0 / ((A::K::USIZE - 1) as f32)
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            frequencies,
            alphabet: std::marker::PhantomData,
        }
    }

    /// A reference to the raw background frequencies.
    pub fn frequencies(&self) -> &[f32] {
        &self.frequencies
    }
}

impl<A: Alphabet> AsRef<[f32]> for Background<A> {
    fn as_ref(&self) -> &[f32] {
        self.frequencies()
    }
}

impl<A: Alphabet> AsRef<GenericArray<f32, A::K>> for Background<A> {
    fn as_ref(&self) -> &GenericArray<f32, A::K> {
        &self.frequencies
    }
}

impl<A: Alphabet> Default for Background<A> {
    fn default() -> Self {
        Self::uniform()
    }
}

// --- Pseudocounts ------------------------------------------------------------

/// A structure for storing the pseudocounts over an alphabet.
#[derive(Clone, Debug)]
pub struct Pseudocounts<A: Alphabet> {
    counts: GenericArray<f32, A::K>,
    alphabet: std::marker::PhantomData<A>,
}

impl<A: Alphabet> Pseudocounts<A> {
    pub fn counts(&self) -> &[f32] {
        &self.counts
    }
}

impl<A: Alphabet> Default for Pseudocounts<A> {
    fn default() -> Self {
        Self::from(0.0)
    }
}

impl<A: Alphabet> From<GenericArray<f32, A::K>> for Pseudocounts<A> {
    fn from(counts: GenericArray<f32, A::K>) -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            counts,
        }
    }
}

impl<A: Alphabet> From<f32> for Pseudocounts<A> {
    fn from(count: f32) -> Self {
        let counts = (0..A::K::USIZE)
            .map(|i| {
                if i != A::default_symbol().as_index() {
                    count
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            counts,
            alphabet: std::marker::PhantomData,
        }
    }
}

impl<A: Alphabet> AsRef<[f32]> for Pseudocounts<A> {
    fn as_ref(&self) -> &[f32] {
        &self.counts
    }
}

impl<A: Alphabet> AsMut<[f32]> for Pseudocounts<A> {
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.counts
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_background_new() {
        assert!(Background::<Dna>::new([0.3, 0.2, 0.2, 0.3, 0.0]).is_ok());
        assert!(Background::<Dna>::new([0.1, 0.1, 0.1, 0.1, 0.0]).is_err());
    }
}
