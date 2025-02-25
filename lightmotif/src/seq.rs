//! Linear and striped storage for alphabet-encoded sequences.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::fmt::Write;
use std::ops::Index;
use std::str::FromStr;

use generic_array::GenericArray;
#[cfg(feature = "sampling")]
use rand::Rng;

use super::abc::Alphabet;
#[cfg(feature = "sampling")]
use super::abc::Background;
use super::abc::Symbol;
use super::dense::DefaultColumns;
use super::dense::DenseMatrix;
use super::err::InvalidData;
use super::err::InvalidSymbol;
use super::num::StrictlyPositive;
use super::pli::dispatch::Dispatch;
use super::pli::Encode;
use super::pli::Pipeline;
use super::pli::Stripe;
use super::pwm::ScoringMatrix;
use crate::num::ArrayLength;
use crate::num::PositiveLength;

// --- SymbolCount -------------------------------------------------------------

/// A trait for counting the number of occurences of a symbol in a sequence.
pub trait SymbolCount<A: Alphabet> {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize;

    fn count_symbols(&self) -> GenericArray<usize, A::K> {
        let mut counts = GenericArray::default();
        for s in A::symbols() {
            counts[s.as_index()] = self.count_symbol(*s);
        }
        counts
    }
}

impl<'a, A: Alphabet, T: SymbolCount<A>> SymbolCount<A> for &'a T {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        (*self).count_symbol(symbol)
    }

    fn count_symbols(&self) -> GenericArray<usize, A::K> {
        (*self).count_symbols()
    }
}

impl<A: Alphabet> SymbolCount<A> for [A::Symbol] {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        self.as_ref().iter().filter(|&&c| c == symbol).count()
    }

    fn count_symbols(&self) -> GenericArray<usize, A::K> {
        let mut counts = GenericArray::default();
        for c in self.as_ref().iter() {
            counts[c.as_index()] += 1;
        }
        counts
    }
}

impl<'a, A: Alphabet> SymbolCount<A> for &'a [A::Symbol] {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        <[A::Symbol] as SymbolCount<A>>::count_symbol(self, symbol)
    }

    fn count_symbols(&self) -> GenericArray<usize, A::K> {
        <[A::Symbol] as SymbolCount<A>>::count_symbols(self)
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
    #[inline]
    pub fn encode<S: AsRef<[u8]>>(sequence: S) -> Result<Self, InvalidSymbol> {
        let pli = Pipeline::<A, _>::dispatch();
        pli.encode(sequence.as_ref())
    }

    /// Sample a new random sequence from the given background frequencies.
    #[cfg(feature = "sampling")]
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

    /// Check whether the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
    #[inline]
    pub fn to_striped<C>(&self) -> StripedSequence<A, C>
    where
        C: StrictlyPositive + ArrayLength,
        Pipeline<A, Dispatch>: Stripe<A, C>,
    {
        let pli = Pipeline::<A, _>::dispatch();
        pli.stripe(&self.data)
    }
}

impl<A: Alphabet> AsRef<EncodedSequence<A>> for EncodedSequence<A> {
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<A: Alphabet> AsRef<[A::Symbol]> for EncodedSequence<A> {
    #[inline]
    fn as_ref(&self) -> &[A::Symbol] {
        self.data.as_slice()
    }
}

impl<A: Alphabet> Default for EncodedSequence<A> {
    #[inline]
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<A: Alphabet> Display for EncodedSequence<A> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        for c in self.data.iter() {
            f.write_char(c.as_char())?;
        }
        Ok(())
    }
}

impl<A: Alphabet> FromIterator<A::Symbol> for EncodedSequence<A> {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A::Symbol>,
    {
        Self::new(Vec::from_iter(iter))
    }
}

impl<A: Alphabet> FromStr for EncodedSequence<A> {
    type Err = InvalidSymbol;
    #[inline]
    fn from_str(seq: &str) -> Result<Self, Self::Err> {
        Self::encode(seq)
    }
}

impl<A: Alphabet> From<Vec<A::Symbol>> for EncodedSequence<A> {
    #[inline]
    fn from(data: Vec<A::Symbol>) -> Self {
        Self::new(data)
    }
}

impl<A: Alphabet> Index<usize> for EncodedSequence<A> {
    type Output = A::Symbol;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, A: Alphabet> IntoIterator for &'a EncodedSequence<A> {
    type Item = &'a A::Symbol;
    type IntoIter = std::slice::Iter<'a, A::Symbol>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<A, S> PartialEq<S> for EncodedSequence<A>
where
    A: Alphabet,
    S: AsRef<[<A as Alphabet>::Symbol]>,
{
    #[inline]
    fn eq(&self, other: &S) -> bool {
        let l = self.data.as_slice();
        let r = other.as_ref();
        l == r
    }
}

impl<A: Alphabet> SymbolCount<A> for EncodedSequence<A> {
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        self.data.iter().filter(|&&c| c == symbol).count()
    }

    fn count_symbols(&self) -> GenericArray<usize, A::K> {
        let mut counts = GenericArray::default();
        for c in self.data.iter() {
            counts[c.as_index()] += 1;
        }
        counts
    }
}

// --- StripedSequence ---------------------------------------------------------

/// Number of extra rows to add when creating a striped sequence.
///
/// The higher this number, the less frequent the sequence matrix
/// matrix needs to be reallocated when `StripedSequence.configure`
/// is called.
pub(crate) const DEFAULT_EXTRA_ROWS: usize = 32;

/// An encoded sequence stored in a striped matrix with a fixed column count.
#[derive(Clone)]
pub struct StripedSequence<A: Alphabet, C: PositiveLength = DefaultColumns> {
    alphabet: std::marker::PhantomData<A>,
    length: usize,
    wrap: usize,
    data: DenseMatrix<A::Symbol, C>,
}

impl<A: Alphabet, C: PositiveLength> StripedSequence<A, C> {
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
    #[cfg(feature = "sampling")]
    pub fn sample<R: Rng>(mut rng: R, background: Background<A>, length: usize) -> Self {
        let symbols = <A as Alphabet>::symbols();
        let dist = rand_distr::WeightedAliasIndex::new(background.frequencies().into())
            .expect("`Background` always stores frequencies valid for `WeightedAliasIndex::new`");
        let mut data = unsafe { DenseMatrix::uninitialized(length.div_ceil(C::USIZE)) };
        for row in data.iter_mut() {
            for (x, y) in row.iter_mut().zip((&mut rng).sample_iter(&dist)) {
                *x = symbols[y];
            }
        }
        Self::new(data, length).expect("`EncodedSequence::sample` computes length properly")
    }

    /// Get the length of the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check whether the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the number of wrapping rows in the striped matrix.
    #[inline]
    pub fn wrap(&self) -> usize {
        self.wrap
    }

    /// Get an immutable reference over the underlying matrix storing the sequence.
    #[inline]
    pub fn matrix(&self) -> &DenseMatrix<A::Symbol, C> {
        &self.data
    }

    /// Extract the underlying matrix where the striped sequence is stored.
    #[inline]
    pub fn into_matrix(self) -> DenseMatrix<A::Symbol, C> {
        self.data
    }

    /// Reconfigure the striped sequence for searching with a motif.
    #[inline]
    pub fn configure(&mut self, motif: &ScoringMatrix<A>) {
        if !motif.is_empty() {
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

impl<A: Alphabet, C: PositiveLength> AsRef<StripedSequence<A, C>> for StripedSequence<A, C> {
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<A: Alphabet, C: PositiveLength> AsRef<DenseMatrix<A::Symbol, C>> for StripedSequence<A, C> {
    #[inline]
    fn as_ref(&self) -> &DenseMatrix<A::Symbol, C> {
        &self.data
    }
}

impl<A: Alphabet, C: PositiveLength> std::fmt::Debug for StripedSequence<A, C> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("StripedSequence")
            .field("alphabet", &self.alphabet)
            .field("length", &self.length)
            .field("wrap", &self.wrap)
            .field("data", &self.data)
            .finish()
    }
}

impl<A: Alphabet, C: PositiveLength> Default for StripedSequence<A, C> {
    #[inline]
    fn default() -> Self {
        Self::new(DenseMatrix::new(0), 0).unwrap()
    }
}

impl<A: Alphabet, C: PositiveLength> From<StripedSequence<A, C>> for DenseMatrix<A::Symbol, C> {
    #[inline]
    fn from(striped: StripedSequence<A, C>) -> Self {
        striped.into_matrix()
    }
}

impl<A: Alphabet, C: PositiveLength> From<EncodedSequence<A>> for StripedSequence<A, C>
where
    Pipeline<A, Dispatch>: Stripe<A, C>,
{
    #[inline]
    fn from(encoded: EncodedSequence<A>) -> Self {
        encoded.to_striped()
    }
}

impl<A: Alphabet, C: PositiveLength> Index<usize> for StripedSequence<A, C> {
    type Output = <A as Alphabet>::Symbol;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let rows = self.data.rows() - self.wrap;
        let col = index / rows;
        let row = index % rows;
        &self.data[row][col]
    }
}

impl<A: Alphabet, C: PositiveLength> SymbolCount<A> for StripedSequence<A, C> {
    #[inline]
    fn count_symbol(&self, symbol: <A as Alphabet>::Symbol) -> usize {
        let mut count = 0;

        let rows = self.data.rows() - self.wrap;
        let l = self.len();

        for (i, row) in self.data.iter().take(rows).enumerate() {
            for (j, &x) in row.iter().enumerate() {
                if x == symbol {
                    let index = j * rows + i;
                    if index < l {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn count_symbols(&self) -> GenericArray<usize, <A as Alphabet>::K> {
        let mut counts = GenericArray::default();

        let rows = self.data.rows() - self.wrap;
        let l = self.len();

        for (i, row) in self.data.iter().take(rows).enumerate() {
            for (j, &x) in row.iter().enumerate() {
                let index = j * rows + i;
                if index < l {
                    counts[x.as_index()] += 1;
                }
            }
        }

        counts
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

    #[test]
    fn count_symbols() {
        let seq = EncodedSequence::<Dna>::from_str("ATGCAAGGAGATTCTAGAT").unwrap();
        let mut striped: StripedSequence<Dna, _> = seq.to_striped();

        let seq_counts = SymbolCount::<Dna>::count_symbols(&seq);
        let striped_counts = SymbolCount::<Dna>::count_symbols(&striped);
        for s in Dna::symbols() {
            let seq_count = SymbolCount::<Dna>::count_symbol(&seq, *s);
            let striped_count = SymbolCount::<Dna>::count_symbol(&seq, *s);
            assert_eq!(seq_count, striped_count);
            assert_eq!(seq_count, seq_counts[s.as_index()]);
            assert_eq!(striped_count, striped_counts[s.as_index()]);
        }

        striped.configure_wrap(32);

        let seq_counts = SymbolCount::<Dna>::count_symbols(&seq);
        let striped_counts = SymbolCount::<Dna>::count_symbols(&striped);
        for s in Dna::symbols() {
            let seq_count = SymbolCount::<Dna>::count_symbol(&seq, *s);
            let striped_count = SymbolCount::<Dna>::count_symbol(&seq, *s);
            assert_eq!(seq_count, striped_count);
            assert_eq!(seq_count, seq_counts[s.as_index()]);
            assert_eq!(striped_count, striped_counts[s.as_index()]);
        }
    }
}
