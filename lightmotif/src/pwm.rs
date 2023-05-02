use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Pseudocounts;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::seq::EncodedSequence;

/// A matrix storing symbol occurences at each position.
#[derive(Clone, Debug)]
pub struct CountMatrix<A: Alphabet, const K: usize> {
    alphabet: std::marker::PhantomData<A>,
    data: DenseMatrix<u32, K>,
}

impl<A: Alphabet, const K: usize> CountMatrix<A, K> {
    /// Create a new count matrix from the given data.
    pub fn new(data: DenseMatrix<u32, K>) -> Result<Self, ()> {
        Ok(Self {
            data,
            alphabet: std::marker::PhantomData,
        })
    }

    /// Create a new count matrix from the given sequences.
    pub fn from_sequences<'seq, I>(sequences: I) -> Result<Self, ()>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: AsRef<EncodedSequence<A>>,
    {
        let mut data = None;
        for seq in sequences {
            let seq = seq.as_ref();
            let d = match data.as_mut() {
                Some(d) => d,
                None => {
                    data = Some(DenseMatrix::new(seq.len()));
                    data.as_mut().unwrap()
                }
            };
            if seq.len() != d.rows() {
                return Err(());
            }
            for (i, x) in seq.data.iter().enumerate() {
                d[i][x.as_index()] += 1;
            }
        }

        Ok(Self {
            alphabet: std::marker::PhantomData,
            data: data.unwrap_or_else(|| DenseMatrix::new(0)),
        })
    }

    /// Build a probability matrix from this count matrix using pseudo-counts.
    pub fn to_probability<P>(&self, pseudo: P) -> ProbabilityMatrix<A, K>
    where
        P: Into<Pseudocounts<A, K>>,
    {
        let p = pseudo.into();
        let mut probas = DenseMatrix::new(self.data.rows());
        for i in 0..self.data.rows() {
            let src = &self.data[i];
            let dst = &mut probas[i];
            for (j, &x) in src.iter().enumerate() {
                dst[j] = x as f32 + p.counts()[j] as f32;
            }
            let s: f32 = dst.iter().sum();
            for x in dst.iter_mut() {
                *x /= s;
            }
        }
        ProbabilityMatrix {
            alphabet: std::marker::PhantomData,
            data: probas,
        }
    }

    /// The raw counts from the count matrix.
    #[inline]
    pub fn counts(&self) -> &DenseMatrix<u32, K> {
        &self.data
    }
}

impl<A: Alphabet, const K: usize> AsRef<DenseMatrix<u32, K>> for CountMatrix<A, K> {
    fn as_ref(&self) -> &DenseMatrix<u32, K> {
        &self.data
    }
}

/// A matrix storing symbol probabilities at each position.
#[derive(Clone, Debug)]
pub struct ProbabilityMatrix<A: Alphabet, const K: usize> {
    alphabet: std::marker::PhantomData<A>,
    data: DenseMatrix<f32, K>,
}

impl<A: Alphabet, const K: usize> ProbabilityMatrix<A, K> {
    /// Convert to a weight matrix using the given background frequencies.
    pub fn to_weight<B>(&self, background: B) -> WeightMatrix<A, K>
    where
        B: Into<Background<A, K>>,
    {
        let b = background.into();
        let mut weight = DenseMatrix::new(self.data.rows());
        for i in 0..self.data.rows() {
            let src = &self.data[i];
            let dst = &mut weight[i];
            for (j, (&x, &f)) in src.iter().zip(b.frequencies()).enumerate() {
                dst[j] = (x / f).log2();
            }
        }
        WeightMatrix {
            background: b,
            alphabet: std::marker::PhantomData,
            data: weight,
        }
    }
}

impl<A: Alphabet, const K: usize> AsRef<DenseMatrix<f32, K>> for ProbabilityMatrix<A, K> {
    fn as_ref(&self) -> &DenseMatrix<f32, K> {
        &self.data
    }
}

/// A matrix storing log-likelihood of symbol probabilities at each position.
#[derive(Clone, Debug)]
pub struct WeightMatrix<A: Alphabet, const K: usize> {
    alphabet: std::marker::PhantomData<A>,
    background: Background<A, K>,
    data: DenseMatrix<f32, K>,
}

impl<A: Alphabet, const K: usize> WeightMatrix<A, K> {
    /// The length of the motif encoded in this weight matrix.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.rows()
    }

    /// The log-likelihoods of the position weight matrix.
    #[inline]
    pub fn weights(&self) -> &DenseMatrix<f32, K> {
        &self.data
    }

    /// The background frequencies of the position weight matrix.
    #[inline]
    pub fn background(&self) -> &Background<A, K> {
        &self.background
    }
}

impl<A: Alphabet, const K: usize> AsRef<WeightMatrix<A, K>> for WeightMatrix<A, K> {
    fn as_ref(&self) -> &Self {
        &self
    }
}

impl<A: Alphabet, const K: usize> AsRef<DenseMatrix<f32, K>> for WeightMatrix<A, K> {
    fn as_ref(&self) -> &DenseMatrix<f32, K> {
        &self.data
    }
}
