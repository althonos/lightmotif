use super::abc::Alphabet;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::seq::EncodedSequence;

#[derive(Clone, Debug)]
pub struct CountMatrix<A: Alphabet, const K: usize> {
    pub alphabet: A,
    pub data: DenseMatrix<u32, K>,
}

impl<A: Alphabet, const K: usize> CountMatrix<A, K> {
    pub fn from_sequences<'seq, I>(sequences: I) -> Result<Self, ()>
    where
        I: IntoIterator<Item = &'seq EncodedSequence<A>>,
    {
        let mut data = None;
        for seq in sequences {
            let mut d = match data.as_mut() {
                Some(d) => d,
                None => {
                    data = Some(DenseMatrix::new(seq.len()));
                    data.as_mut().unwrap()
                }
            };
            for (i, x) in seq.data.iter().enumerate() {
                d[i][x.as_index()] += 1;
            }
        }

        Ok(Self {
            alphabet: A::default(),
            data: data.unwrap_or_else(|| DenseMatrix::new(0)),
        })
    }
}
