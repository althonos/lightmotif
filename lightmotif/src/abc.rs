use std::fmt::Debug;

use super::err::InvalidData;
use super::err::InvalidSymbol;

// --- Symbol ------------------------------------------------------------------

/// Common traits for a biological symbol.
pub trait Symbol: Default + Sized + Copy {
    /// View this symbol as a zero-based index.
    fn as_index(&self) -> usize;
    /// View this symbol as a string character.
    fn as_char(&self) -> char;
    /// Parse a string character into a symbol.
    fn from_char(c: char) -> Result<Self, InvalidSymbol>;
}

/// A deoxyribonucleotide.
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

impl From<Nucleotide> for char {
    fn from(n: Nucleotide) -> char {
        n.as_char()
    }
}

impl Default for Nucleotide {
    fn default() -> Nucleotide {
        Nucleotide::N
    }
}

// --- Alphabet ----------------------------------------------------------------

/// Common traits for a biological alphabet.
pub trait Alphabet: Debug + Copy + Default + 'static {
    type Symbol: Symbol;
    const K: usize;

    /// Get the default symbol for this alphabet.
    fn default_symbol() -> Self::Symbol {
        Default::default()
    }
}

/// The standard DNA alphabet composed of 4 deoxyribonucleotides and a wildcard.
#[derive(Default, Debug, Clone, Copy)]
pub struct Dna;

impl Alphabet for Dna {
    type Symbol = Nucleotide;
    const K: usize = 5;
}

// --- Background --------------------------------------------------------------

/// The background frequencies for an alphabet.
#[derive(Clone, Debug)]
pub struct Background<A: Alphabet, const K: usize> {
    frequencies: [f32; K],
    alphabet: std::marker::PhantomData<A>,
}

impl<A: Alphabet, const K: usize> Background<A, K> {
    /// Create a new background with the given frequencies.
    ///
    /// The array must contain valid frequencies, i.e. real numbers between
    /// zero and one that sum to one.
    pub fn new(frequencies: [f32; K]) -> Result<Self, InvalidData> {
        let mut sum = 0.0;
        for &f in frequencies.iter() {
            if f < 0.0 || f > 1.0 {
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
    /// # use lightmotif::*;
    /// let bg = Background::<Dna, { Dna::K }>::uniform();
    /// assert_eq!(bg.frequencies(), &[0.25, 0.25, 0.25, 0.25, 0.0]);
    /// ```
    pub fn uniform() -> Self {
        let mut frequencies = [0.0; K];
        for i in 0..K {
            if i != A::default_symbol().as_index() {
                frequencies[i] = 1.0 / ((K - 1) as f32);
            }
        }
        Self {
            frequencies,
            alphabet: std::marker::PhantomData,
        }
    }

    /// A reference to the raw background frequencies.
    pub fn frequencies(&self) -> &[f32; K] {
        &self.frequencies
    }
}

impl<A: Alphabet, const K: usize> AsRef<[f32; K]> for Background<A, K> {
    fn as_ref(&self) -> &[f32; K] {
        &self.frequencies
    }
}

impl<A: Alphabet, const K: usize> Default for Background<A, K> {
    fn default() -> Self {
        Self::uniform()
    }
}

// --- Pseudocounts ------------------------------------------------------------

/// A structure for storing the pseudocounts over an alphabet.
#[derive(Clone, Debug)]
pub struct Pseudocounts<A: Alphabet, const K: usize> {
    alphabet: std::marker::PhantomData<A>,
    counts: [f32; K],
}

impl<A: Alphabet, const K: usize> Pseudocounts<A, K> {
    pub fn counts(&self) -> &[f32; K] {
        &self.counts
    }
}

impl<A: Alphabet, const K: usize> Default for Pseudocounts<A, K> {
    fn default() -> Self {
        Self::from([0.0; K])
    }
}

impl<A: Alphabet, const K: usize> From<[f32; K]> for Pseudocounts<A, K> {
    fn from(counts: [f32; K]) -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            counts,
        }
    }
}

impl<A: Alphabet, const K: usize> From<f32> for Pseudocounts<A, K> {
    fn from(count: f32) -> Self {
        let mut counts: [f32; K] = [0.0; K];
        for i in 0..K {
            if i != A::default_symbol().as_index() {
                counts[i] = count;
            }
        }
        Self {
            counts,
            alphabet: std::marker::PhantomData,
        }
    }
}

impl<A: Alphabet, const K: usize> AsRef<[f32; K]> for Pseudocounts<A, K> {
    fn as_ref(&self) -> &[f32; K] {
        &self.counts
    }
}

impl<A: Alphabet, const K: usize> AsMut<[f32; K]> for Pseudocounts<A, K> {
    fn as_mut(&mut self) -> &mut [f32; K] {
        &mut self.counts
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_background_new() {
        assert!(Background::<Dna, { Dna::K }>::new([0.3, 0.2, 0.2, 0.3, 0.0]).is_ok());
        assert!(Background::<Dna, { Dna::K }>::new([0.1, 0.1, 0.1, 0.1, 0.0]).is_err());
    }
}
