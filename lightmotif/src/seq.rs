//! Linear and striped storage for alphabet-encoded sequences.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::str::FromStr;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

use super::abc::Alphabet;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::err::InvalidSymbol;
use super::num::StrictlyPositive;
use super::pwm::ScoringMatrix;

// --- EncodedSequence ---------------------------------------------------------

/// A biological sequence encoded with an alphabet.
#[derive(Clone, Debug)]
pub struct EncodedSequence<A: Alphabet> {
    alphabet: std::marker::PhantomData<A>,
    data: Vec<A::Symbol>,
}

impl<A: Alphabet> EncodedSequence<A> {
    /// Create a new encoded sequence.
    pub fn new(data: Vec<A::Symbol>) -> Self {
        Self {
            data,
            alphabet: std::marker::PhantomData,
        }
    }

    /// Create a new encoded sequence from a textual representation.
    pub fn encode(sequence: &str) -> Result<Self, InvalidSymbol> {
        sequence
            .chars()
            .map(A::Symbol::from_char)
            .collect::<Result<_, _>>()
            .map(Self::new)
    }

    /// Return the number of symbols in the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Iterate over the symbols in the sequence.
    #[inline]
    pub fn iter(&self) -> impl IntoIterator<Item = &A::Symbol> {
        self.data.iter()
    }

    /// Convert the encoded sequence to a striped matrix.
    pub fn to_striped<C: Unsigned + NonZero>(&self) -> StripedSequence<A, C> {
        let length = self.data.len();
        let n = (length + (C::USIZE - 1)) / C::USIZE;
        let mut data = DenseMatrix::new(n);
        for (i, &x) in self.data.iter().enumerate() {
            data[i % n][i / n] = x;
        }
        StripedSequence {
            alphabet: std::marker::PhantomData,
            data,
            length,
            wrap: 0,
        }
    }

    /// Convert the encoded sequence back to its textual representation.
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(self.len());
        for c in self.data.iter() {
            s.push(c.as_char());
        }
        s
    }
}

impl<A: Alphabet> AsRef<EncodedSequence<A>> for EncodedSequence<A> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<A: Alphabet> AsRef<[<A as Alphabet>::Symbol]> for EncodedSequence<A> {
    fn as_ref(&self) -> &[<A as Alphabet>::Symbol] {
        self.data.as_slice()
    }
}

impl<A: Alphabet> Display for EncodedSequence<A> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        for c in self.data.iter() {
            write!(f, "{}", c.as_char())?;
        }
        Ok(())
    }
}

impl<A: Alphabet> FromStr for EncodedSequence<A> {
    type Err = InvalidSymbol;
    fn from_str(seq: &str) -> Result<Self, Self::Err> {
        Self::encode(seq)
    }
}

impl<'a, A: Alphabet> IntoIterator for &'a EncodedSequence<A> {
    type Item = &'a A::Symbol;
    type IntoIter = std::slice::Iter<'a, A::Symbol>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// --- StripedSequence ---------------------------------------------------------

/// An encoded sequence stored in a striped matrix with a fixed column count.
#[derive(Clone, Debug)]
pub struct StripedSequence<A: Alphabet, C: StrictlyPositive> {
    alphabet: std::marker::PhantomData<A>,
    pub length: usize,
    pub wrap: usize,
    pub data: DenseMatrix<A::Symbol, C>,
}

impl<A: Alphabet, C: StrictlyPositive> StripedSequence<A, C> {
    /// Create a new striped sequence from a textual representation.
    pub fn encode(sequence: &str) -> Result<Self, InvalidSymbol> {
        let length = sequence.len();
        let n = (length + (C::USIZE - 1)) / C::USIZE;
        let mut data = DenseMatrix::new(n);
        for (i, x) in sequence.chars().enumerate() {
            data[i % n][i / n] = A::Symbol::from_char(x)?;
        }
        Ok(StripedSequence {
            alphabet: std::marker::PhantomData,
            data,
            length,
            wrap: 0,
        })
    }

    /// Reconfigure the striped sequence for searching with a motif.
    pub fn configure(&mut self, motif: &ScoringMatrix<A>) {
        if motif.len() > 0 {
            self.configure_wrap(motif.len() - 1);
        }
    }

    /// Add wrap-around rows for a motif of length `m`.
    pub fn configure_wrap(&mut self, m: usize) {
        if m > self.wrap {
            let rows = self.data.rows() - self.wrap;
            self.data.resize(self.data.rows() + m - self.wrap);
            for i in 0..m {
                for j in 0..C::USIZE - 1 {
                    self.data[rows + i][j] = self.data[i][j + 1];
                }
            }
            self.wrap = m;
        }
    }
}

impl<A: Alphabet, C: StrictlyPositive> AsRef<StripedSequence<A, C>> for StripedSequence<A, C> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<A: Alphabet, C: StrictlyPositive> From<EncodedSequence<A>> for StripedSequence<A, C> {
    fn from(encoded: EncodedSequence<A>) -> Self {
        encoded.to_striped()
    }
}

impl<A: Alphabet, C: StrictlyPositive> FromStr for StripedSequence<A, C> {
    type Err = InvalidSymbol;
    fn from_str(seq: &str) -> Result<Self, Self::Err> {
        Self::encode(seq)
    }
}

#[cfg(test)]
mod test {
    use typenum::consts::U2;
    use typenum::consts::U4;

    use super::*;

    use crate::abc::Dna;
    use crate::abc::Nucleotide::*;

    #[test]
    fn test_stripe() {
        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let striped = seq.to_striped::<U4>();
        assert_eq!(striped.data.rows(), 2);
        assert_eq!(&striped.data[0], &[A, G, A, N]);
        assert_eq!(&striped.data[1], &[T, C, N, N]);

        let striped = seq.to_striped::<U2>();
        assert_eq!(striped.data.rows(), 3);
        assert_eq!(&striped.data[0], &[A, C,]);
        assert_eq!(&striped.data[1], &[T, A,]);
        assert_eq!(&striped.data[2], &[G, N,]);
    }

    #[test]
    fn test_configure_wrap() {
        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let mut striped = seq.to_striped::<U4>();

        striped.configure_wrap(2);
        assert_eq!(striped.data.rows(), 4);
        assert_eq!(&striped.data[0], &[A, G, A, N]);
        assert_eq!(&striped.data[1], &[T, C, N, N]);
        assert_eq!(&striped.data[2], &[G, A, N, N]);
        assert_eq!(&striped.data[3], &[C, N, N, N]);
    }
}
