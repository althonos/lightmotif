//! Linear and striped storage for alphabet-encoded sequences.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::Index;
use std::str::FromStr;

#[cfg(feature = "sample")]
use rand::Rng;

use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::err::InvalidData;
use super::err::InvalidSymbol;
use super::num::StrictlyPositive;
use super::pli::dispatch::Dispatch;
use super::pli::Encode;
use super::pli::Pipeline;
use super::pli::Stripe;
use super::pwm::ScoringMatrix;

// --- SymbolCount -------------------------------------------------------------

/// A trait for counting the number of occurences of a symbol in a sequence.
pub trait SymbolCount<A: Alphabet> {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize;
}

impl<'a, A: Alphabet, T: IntoIterator<Item = &'a A::Symbol> + Copy> SymbolCount<A> for T {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        self.into_iter().filter(|&&c| c == symbol).count()
    }
}

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
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn encode<S: AsRef<[u8]>>(sequence: S) -> Result<Self, InvalidSymbol> {
        let pli = Pipeline::<A, _>::dispatch();
        pli.encode(sequence.as_ref())
    }

    /// Sample a new random sequence from the given background frequencies.
    #[cfg(feature = "sample")]
    pub fn sample<R: Rng>(rng: R, background: Background<A>, length: usize) -> Self {
        let symbols = <A as Alphabet>::symbols();
        let dist = rand_distr::WeightedAliasIndex::new(background.frequencies().into())
            .expect("`Background` always stores frequencies valid for `WeightedAliasIndex::new`");
        rng.sample_iter(&dist)
            .take(length)
            .map(|i| symbols[i])
            .collect()
    }

    /// Return the number of symbols in the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Iterate over the symbols in the sequence.
    #[inline]
    pub fn iter(&self) -> <&[A::Symbol] as IntoIterator>::IntoIter {
        self.data.as_slice().iter()
    }

    /// Convert the encoded sequence to a striped matrix.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn to_striped<C>(&self) -> StripedSequence<A, C>
    where
        C: StrictlyPositive,
        Pipeline<A, Dispatch>: Stripe<A, C>,
    {
        let pli = Pipeline::<A, _>::dispatch();
        pli.stripe(&self.data)
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

impl<A: Alphabet> Default for EncodedSequence<A> {
    fn default() -> Self {
        Self::new(Vec::new())
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

impl<A: Alphabet> FromIterator<A::Symbol> for EncodedSequence<A> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A::Symbol>,
    {
        Self::new(Vec::from_iter(iter))
    }
}

impl<A: Alphabet> FromStr for EncodedSequence<A> {
    type Err = InvalidSymbol;
    fn from_str(seq: &str) -> Result<Self, Self::Err> {
        Self::encode(seq)
    }
}

impl<A: Alphabet> From<Vec<A::Symbol>> for EncodedSequence<A> {
    fn from(data: Vec<A::Symbol>) -> Self {
        Self::new(data)
    }
}

impl<A: Alphabet> Index<usize> for EncodedSequence<A> {
    type Output = A::Symbol;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, A: Alphabet> IntoIterator for &'a EncodedSequence<A> {
    type Item = &'a A::Symbol;
    type IntoIter = std::slice::Iter<'a, A::Symbol>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<A, S> PartialEq<S> for EncodedSequence<A>
where
    A: Alphabet,
    S: AsRef<[<A as Alphabet>::Symbol]>,
{
    fn eq(&self, other: &S) -> bool {
        let l = self.data.as_slice();
        let r = other.as_ref();
        l == r
    }

    fn ne(&self, other: &S) -> bool {
        let l = self.data.as_slice();
        let r = other.as_ref();
        l != r
    }
}

// --- StripedSequence ---------------------------------------------------------

/// An encoded sequence stored in a striped matrix with a fixed column count.
#[derive(Clone, Debug)]
pub struct StripedSequence<A: Alphabet, C: StrictlyPositive> {
    alphabet: std::marker::PhantomData<A>,
    length: usize,
    wrap: usize,
    data: DenseMatrix<A::Symbol, C>,
}

impl<A: Alphabet, C: StrictlyPositive> StripedSequence<A, C> {
    /// Create a new striped sequence from the given dense matrix.
    ///
    /// # Errors
    /// Returns `InvalidData` when the `DenseMatrix` given as input stores
    /// less symbols than the given `length`.
    pub fn new(data: DenseMatrix<A::Symbol, C>, length: usize) -> Result<Self, InvalidData> {
        if data.rows() * data.columns() < length {
            Err(InvalidData)
        } else {
            Ok(Self {
                data,
                length,
                wrap: 0,
                alphabet: std::marker::PhantomData,
            })
        }
    }

    /// Sample a new random sequence from the given background frequencies.
    #[cfg(feature = "sample")]
    pub fn sample<R: Rng>(mut rng: R, background: Background<A>, length: usize) -> Self {
        let symbols = <A as Alphabet>::symbols();
        let dist = rand_distr::WeightedAliasIndex::new(background.frequencies().into())
            .expect("`Background` always stores frequencies valid for `WeightedAliasIndex::new`");
        let mut data = unsafe { DenseMatrix::uninitialized((length + C::USIZE - 1) / C::USIZE) };
        for row in data.iter_mut() {
            for (x, y) in row.iter_mut().zip((&mut rng).sample_iter(&dist)) {
                *x = symbols[y];
            }
        }
        Self::new(data, length).expect("`EncodedSequence::sample` computes length properly")
    }

    /// Get the length of the sequence.
    #[inline]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Get the number of wrapping rows in the striped matrix.
    #[inline]
    pub const fn wrap(&self) -> usize {
        self.wrap
    }

    /// Get an immutable reference over the underlying matrix storing the sequence.
    #[inline]
    pub const fn matrix(&self) -> &DenseMatrix<A::Symbol, C> {
        &self.data
    }

    /// Extract the underlying matrix where the striped sequence is stored.
    #[inline]
    pub fn into_matrix(self) -> DenseMatrix<A::Symbol, C> {
        self.data
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
                self.data[rows + i][C::USIZE - 1] = A::default_symbol();
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

impl<A: Alphabet, C: StrictlyPositive> AsRef<DenseMatrix<A::Symbol, C>> for StripedSequence<A, C> {
    fn as_ref(&self) -> &DenseMatrix<A::Symbol, C> {
        &self.data
    }
}

impl<A: Alphabet, C: StrictlyPositive> Default for StripedSequence<A, C> {
    fn default() -> Self {
        Self::new(DenseMatrix::new(0), 0).unwrap()
    }
}

impl<A: Alphabet, C: StrictlyPositive> From<StripedSequence<A, C>> for DenseMatrix<A::Symbol, C> {
    fn from(striped: StripedSequence<A, C>) -> Self {
        striped.into_matrix()
    }
}

impl<A: Alphabet, C: StrictlyPositive> From<EncodedSequence<A>> for StripedSequence<A, C>
where
    Pipeline<A, Dispatch>: Stripe<A, C>,
{
    fn from(encoded: EncodedSequence<A>) -> Self {
        encoded.to_striped()
    }
}

impl<A: Alphabet, C: StrictlyPositive> Index<usize> for StripedSequence<A, C> {
    type Output = <A as Alphabet>::Symbol;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let rows = self.data.rows() - self.wrap;
        let col = index / rows;
        let row = index % rows;
        &self.data[row][col]
    }
}

impl<A: Alphabet, C: StrictlyPositive> SymbolCount<A> for &StripedSequence<A, C> {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        self.data
            .iter()
            .map(|row| SymbolCount::<A>::count_symbol(&row, symbol))
            .sum()
    }
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod test {
    use typenum::consts::U2;
    use typenum::consts::U4;

    use super::*;

    use crate::abc::Dna;
    use crate::abc::Nucleotide::*;

    #[test]
    fn empty() {
        let seq = EncodedSequence::<Dna>::from_str("").unwrap();

        let pli = Pipeline::generic();
        let striped = <Pipeline<_, _> as Stripe<Dna, U2>>::stripe(&pli, &seq);
        assert_eq!(striped.matrix().rows(), 0);

        let striped = seq.to_striped();
        assert_eq!(striped.matrix().rows(), 0);
    }

    #[test]
    fn stripe() {
        let pli = Pipeline::generic();

        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let striped = <Pipeline<_, _> as Stripe<Dna, U4>>::stripe(&pli, &seq);
        assert_eq!(striped.data.rows(), 2);
        assert_eq!(&striped.data[0], &[A, G, A, N]);
        assert_eq!(&striped.data[1], &[T, C, N, N]);

        let striped = <Pipeline<_, _> as Stripe<Dna, U2>>::stripe(&pli, &seq);
        assert_eq!(striped.data.rows(), 3);
        assert_eq!(&striped.data[0], &[A, C,]);
        assert_eq!(&striped.data[1], &[T, A,]);
        assert_eq!(&striped.data[2], &[G, N,]);
    }

    #[test]
    fn configure_wrap() {
        let pli = Pipeline::generic();

        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let mut striped = <Pipeline<_, _> as Stripe<Dna, U4>>::stripe(&pli, seq);
        // println!("{:?}", &striped.data);

        striped.configure_wrap(2);
        assert_eq!(striped.data.rows(), 4);
        // println!("{:?}", &striped.data);
        assert_eq!(&striped.data[0], &[A, G, A, N]);
        assert_eq!(&striped.data[1], &[T, C, N, N]);
        assert_eq!(&striped.data[2], &[G, A, N, N]);
        assert_eq!(&striped.data[3], &[C, N, N, N]);
    }

    #[test]
    fn index() {
        let pli = Pipeline::generic();

        let seq = EncodedSequence::<Dna>::from_str("ATGCA").unwrap();
        let striped = <Pipeline<_, _> as Stripe<Dna, U4>>::stripe(&pli, &seq);
        assert_eq!(striped.data.rows(), 2);
        assert_eq!(striped[0], A);
        assert_eq!(striped[1], T);
        assert_eq!(striped[2], G);
        assert_eq!(striped[3], C);
        assert_eq!(striped[4], A);

        let mut striped = <Pipeline<_, _> as Stripe<Dna, U2>>::stripe(&pli, &seq);
        assert_eq!(striped.data.rows(), 3);
        assert_eq!(striped[0], A);
        assert_eq!(striped[1], T);
        assert_eq!(striped[2], G);
        assert_eq!(striped[3], C);
        assert_eq!(striped[4], A);
        striped.configure_wrap(4);
        assert_eq!(striped.data.rows(), 7);
        assert_eq!(striped[0], A);
        assert_eq!(striped[1], T);
        assert_eq!(striped[2], G);
        assert_eq!(striped[3], C);
        assert_eq!(striped[4], A);
    }
}
