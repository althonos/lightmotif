use super::abc::Alphabet;
use super::abc::InvalidSymbol;
use super::dense::DenseMatrix;

#[derive(Clone, Debug)]
pub struct EncodedSequence<A: Alphabet> {
    pub alphabet: A,
    pub data: Vec<A::Symbol>,
}

impl<A: Alphabet> EncodedSequence<A> {
    /// Create a new encoded sequence from a textual representation.
    pub fn from_text(sequence: &str) -> Result<Self, InvalidSymbol> 
        where InvalidSymbol: From<<A::Symbol as TryFrom<char>>::Error>
    {
        let data = sequence.chars()
            .map(|c| A::Symbol::try_from(c))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            data,
            alphabet: Default::default(),
        })
    }

    /// Convert the encoded sequence to a striped matrix.
    pub fn to_striped<const C: usize>(&self) -> StripedSequence<A, C> {
        let length = self.data.len();
        let n = (length + C) / C;
        let mut data = DenseMatrix::new(n);
        for (i, &x) in self.data.iter().enumerate() {
            data[i%n][i/n] = x;
        }
        StripedSequence { 
            alphabet: self.alphabet,
            data,
            length,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StripedSequence<A: Alphabet, const C: usize = 32> {
    pub alphabet: A,
    pub length: usize,
    pub data: DenseMatrix<A::Symbol, C>,
}

impl<A: Alphabet, const C: usize> From<EncodedSequence<A>> for StripedSequence<A, C> {
    fn from(encoded: EncodedSequence<A>) -> Self {
        encoded.to_striped()
    }
}