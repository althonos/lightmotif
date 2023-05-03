use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Pseudocounts;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::err::InvalidData;
use super::seq::EncodedSequence;

// --- CountMatrix -------------------------------------------------------------

/// A matrix storing symbol occurences at each position.
#[derive(Clone, Debug)]
pub struct CountMatrix<A: Alphabet, const K: usize> {
    /// The alphabet of the count matrix.
    alphabet: std::marker::PhantomData<A>,
    /// The actual counts for each position of the motif.
    data: DenseMatrix<u32, K>,
    /// The number of sequences from which this count matrix was obtained.
    #[allow(unused)]
    n: u32,
}

impl<A: Alphabet, const K: usize> CountMatrix<A, K> {
    /// Create a new count matrix without checking the contents.
    fn new_unchecked(data: DenseMatrix<u32, K>, n: u32) -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            n,
            data,
        }
    }

    /// Create a new count matrix from the given data.
    ///
    /// The matrix must contain count data, for sequences of the same
    /// length, i.e. rows should all sum to the same value.
    pub fn new(data: DenseMatrix<u32, K>) -> Result<Self, InvalidData> {
        // Empty matrices contain valid data.
        if data.rows() == 0 {
            return Ok(Self::new_unchecked(data, 0));
        }
        // Check row sums.
        let n = data.iter().map(|row| row.iter().sum()).max().unwrap();
        Ok(Self::new_unchecked(data, n))
    }

    /// Create a new count matrix from the given sequences.
    pub fn from_sequences<'seq, I>(sequences: I) -> Result<Self, InvalidData>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: AsRef<EncodedSequence<A>>,
    {
        let mut n = 0;
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
                return Err(InvalidData);
            }
            for (i, x) in seq.into_iter().enumerate() {
                d[i][x.as_index()] += 1;
            }
            n += 1;
        }
        match data {
            None => Ok(Self::new_unchecked(DenseMatrix::new(0), n)),
            Some(matrix) => Ok(Self::new_unchecked(matrix, n)),
        }
    }

    /// Build a probability matrix from this count matrix using pseudo-counts.
    pub fn to_freq<P>(&self, pseudo: P) -> FrequencyMatrix<A, K>
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
        FrequencyMatrix {
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

impl<A: Alphabet, const K: usize> FromIterator<EncodedSequence<A>>
    for Result<CountMatrix<A, K>, InvalidData>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = EncodedSequence<A>>,
    {
        CountMatrix::from_sequences(iter)
    }
}

// --- FrequencyMatrix ---------------------------------------------------------

/// A matrix storing symbol frequencies at each position.
#[derive(Clone, Debug)]
pub struct FrequencyMatrix<A: Alphabet, const K: usize> {
    alphabet: std::marker::PhantomData<A>,
    data: DenseMatrix<f32, K>,
}

impl<A: Alphabet, const K: usize> FrequencyMatrix<A, K> {
    /// Convert to a weight matrix using the given background frequencies.
    pub fn to_weight<B>(&self, background: B) -> WeightMatrix<A, K>
    where
        B: Into<Option<Background<A, K>>>,
    {
        let bg = background.into().unwrap_or_default();
        let mut weight = DenseMatrix::new(self.data.rows());
        for (src, dst) in self.data.iter().zip(weight.iter_mut()) {
            for (j, (&x, &f)) in src.iter().zip(bg.frequencies()).enumerate() {
                dst[j] = x / f;
            }
        }
        WeightMatrix {
            background: bg,
            data: weight,
        }
    }

    /// Convert to a scoring matrix using the given background frequencies.
    pub fn to_scoring<B>(&self, background: B) -> ScoringMatrix<A, K>
    where
        B: Into<Option<Background<A, K>>>,
    {
        let bg = background.into().unwrap_or_default();
        let mut weight = DenseMatrix::new(self.data.rows());
        for (src, dst) in self.data.iter().zip(weight.iter_mut()) {
            for (j, (&x, &f)) in src.iter().zip(bg.frequencies()).enumerate() {
                dst[j] = (x / f).log2();
            }
        }
        ScoringMatrix {
            background: bg,
            data: weight,
        }
    }
}

impl<A: Alphabet, const K: usize> AsRef<DenseMatrix<f32, K>> for FrequencyMatrix<A, K> {
    fn as_ref(&self) -> &DenseMatrix<f32, K> {
        &self.data
    }
}

// --- WeightMatrix ------------------------------------------------------------

/// A matrix storing odds ratio of symbol occurences at each position.
#[derive(Clone, Debug)]
pub struct WeightMatrix<A: Alphabet, const K: usize> {
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

    /// Rescale this weight matrix with a different background.
    pub fn rescale<B>(&self, background: B) -> Self
    where
        B: Into<Option<Background<A, K>>>,
    {
        let b = background.into().unwrap_or_default();
        if b.frequencies() != self.background.frequencies() {
            let old_freqs = self.background.frequencies();
            let new_freqs = b.frequencies();
            let mut data = self.data.clone();
            for row in data.iter_mut() {
                for j in 0..K {
                    row[j] *= old_freqs[j] / new_freqs[j];
                }
            }
            Self {
                data,
                background: b,
            }
        } else {
            self.clone()
        }
    }

    /// Get a position-specific scoring matrix from this position weight matrix.
    pub fn to_scoring(&self) -> ScoringMatrix<A, K> {
        let background = self.background.clone();
        let mut data = self.data.clone();
        for row in data.iter_mut() {
            for item in row.iter_mut() {
                *item = item.log2();
            }
        }
        ScoringMatrix { background, data }
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

impl<A: Alphabet, const K: usize> From<ScoringMatrix<A, K>> for WeightMatrix<A, K> {
    fn from(pwm: ScoringMatrix<A, K>) -> Self {
        let background = pwm.background;
        let mut data = pwm.data;
        for row in data.iter_mut() {
            for item in row.iter_mut() {
                *item = 2f32.powf(*item);
            }
        }
        WeightMatrix { background, data }
    }
}

// --- ScoringMatrix -----------------------------------------------------------

/// A matrix storing odds ratio of symbol occurences at each position.
#[derive(Clone, Debug)]
pub struct ScoringMatrix<A: Alphabet, const K: usize> {
    background: Background<A, K>,
    data: DenseMatrix<f32, K>,
}

impl<A: Alphabet, const K: usize> ScoringMatrix<A, K> {
    /// The length of the motif encoded in this scoring matrix.
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

impl<A: Alphabet, const K: usize> AsRef<ScoringMatrix<A, K>> for ScoringMatrix<A, K> {
    fn as_ref(&self) -> &Self {
        &self
    }
}

impl<A: Alphabet, const K: usize> AsRef<DenseMatrix<f32, K>> for ScoringMatrix<A, K> {
    fn as_ref(&self) -> &DenseMatrix<f32, K> {
        &self.data
    }
}

impl<A: Alphabet, const K: usize> From<WeightMatrix<A, K>> for ScoringMatrix<A, K> {
    fn from(pwm: WeightMatrix<A, K>) -> Self {
        let background = pwm.background;
        let mut data = pwm.data;
        for row in data.iter_mut() {
            for item in row.iter_mut() {
                *item = item.log2();
            }
        }
        ScoringMatrix { background, data }
    }
}
