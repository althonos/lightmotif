use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::Range;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

pub use self::vector::Vector;

use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;

// --- Vector ------------------------------------------------------------------

mod vector {
    use typenum::consts::U1;
    use typenum::consts::U16;
    use typenum::consts::U32;
    use typenum::marker_traits::NonZero;
    use typenum::marker_traits::Unsigned;

    mod seal {
        pub trait Sealed {}

        impl Sealed for u8 {}

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl Sealed for std::arch::x86_64::__m128i {}

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl Sealed for std::arch::x86_64::__m256i {}
    }

    /// Sealed trait for concrete vector implementations.
    ///
    /// The trait is defined for the loading vector type, which has `LANES`
    /// lanes of `u8` values. These values are then splat into 4 vectors with
    /// `f32` values to actually compute the scores.
    pub trait Vector: seal::Sealed {
        type LANES: Unsigned + NonZero;
    }

    impl Vector for u8 {
        type LANES = U1;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    impl Vector for std::arch::x86_64::__m128i {
        type LANES = U16;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    impl Vector for std::arch::x86_64::__m256i {
        type LANES = U32;
    }

    #[cfg(target_feature = "avx2")]
    pub type Best = std::arch::x86_64::__m256i;
    #[cfg(all(not(target_feature = "avx2"), target_feature = "ssse3"))]
    pub type Best = std::arch::x86_64::__m128i;
    #[cfg(all(not(target_feature = "avx2"), not(target_feature = "ssse3")))]
    pub type Best = u8;
}

// --- Score -------------------------------------------------------------------

/// Generic trait for computing sequence scores with a PSSM.
pub trait Score<A: Alphabet, V: Vector, C: NonZero + Unsigned = <V as Vector>::LANES> {
    /// Compute the PSSM scores into the given buffer.
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>;

    /// Compute the PSSM scores for every sequence positions.
    fn score<S, M>(seq: S, pssm: M) -> StripedScores<C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        let data = unsafe { DenseMatrix::uninitialized(seq.data.rows() - seq.wrap) };
        let length = seq.length - pssm.len() + 1;
        let mut scores = StripedScores { length, data };
        Self::score_into(seq, pssm, &mut scores);
        scores
    }

    /// Find the sequence position with the highest score.
    fn best_position(scores: &StripedScores<C>) -> Option<usize> {
        if scores.length == 0 {
            return None;
        }

        let mut best_pos = 0;
        let mut best_score = scores.data[0][0];
        for i in 0..scores.length {
            let col = i / scores.data.rows();
            let row = i % scores.data.rows();
            if scores.data[row][col] > best_score {
                best_score = scores.data[row][col];
                best_pos = i;
            }
        }

        Some(best_pos)
    }
}

// --- Pipeline ----------------------------------------------------------------

/// Wrapper implementing score computation for different platforms.
#[derive(Debug, Default, Clone)]
pub struct Pipeline<A: Alphabet, V: Vector = vector::Best> {
    alphabet: std::marker::PhantomData<A>,
    vector: std::marker::PhantomData<V>,
}

/// Scalar scoring implementation.
impl<A: Alphabet, C: NonZero + Unsigned> Score<A, u8, C> for Pipeline<A, u8> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        let seq_rows = seq.data.rows() - seq.wrap;
        let result = &mut scores.data;
        if result.rows() < seq_rows {
            panic!("not enough rows for scores: {}", pssm.len());
        }

        for i in 0..seq.length - pssm.len() + 1 {
            let mut score = 0.0;
            for j in 0..pssm.len() {
                let offset = i + j;
                let col = offset / seq_rows;
                let row = offset % seq_rows;
                score += pssm.weights()[j][seq.data[row][col].as_index()];
            }
            let col = i / result.rows();
            let row = i % result.rows();
            result[row][col] = score;
        }
    }
}

// --- SSSE3 -------------------------------------------------------------------

// --- StripedScores -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StripedScores<C: Unsigned + NonZero> {
    data: DenseMatrix<f32, C>,
    length: usize,
}

impl<C: Unsigned + NonZero> StripedScores<C> {
    /// Create a new striped score matrix with the given length and rows.
    pub fn new(length: usize, rows: usize) -> Self {
        Self {
            length,
            data: DenseMatrix::new(rows),
        }
    }

    /// Return the number of scored positions.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return a reference to the striped matrix storing the scores.
    pub fn matrix(&self) -> &DenseMatrix<f32, C> {
        &self.data
    }

    /// Return a mutable reference to the striped matrix storing the scores.
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        &mut self.data
    }

    /// Create a new matrix large enough to store the scores of `pssm` applied to `seq`.
    pub fn new_for<S, M, A>(seq: S, pssm: M) -> Self
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        Self::new(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap)
    }

    /// Resize the striped scores storage to the given length and number of rows.
    pub fn resize(&mut self, length: usize, rows: usize) {
        self.length = length;
        self.data.resize(rows);
    }

    /// Resize the striped scores storage to store the scores of `pssm` applied to `seq`.
    pub fn resize_for<S, M, A>(&mut self, seq: S, pssm: M)
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        self.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
    }

    /// Iterate over scores of individual sequence positions.
    pub fn iter(&self) -> Iter<'_, C> {
        Iter::new(&self)
    }

    /// Convert the striped scores to a vector of scores.
    pub fn to_vec(&self) -> Vec<f32> {
        self.iter().cloned().collect()
    }
}

impl<C: Unsigned + NonZero> AsRef<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_ref(&self) -> &DenseMatrix<f32, C> {
        self.matrix()
    }
}

impl<C: Unsigned + NonZero> AsMut<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        self.matrix_mut()
    }
}

impl<C: Unsigned + NonZero> Index<usize> for StripedScores<C> {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        let col = index / self.data.rows();
        let row = index % self.data.rows();
        &self.data[row][col]
    }
}

impl<C: Unsigned + NonZero> From<StripedScores<C>> for Vec<f32> {
    fn from(scores: StripedScores<C>) -> Self {
        scores.iter().cloned().collect()
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, C: Unsigned + NonZero> {
    scores: &'a StripedScores<C>,
    indices: Range<usize>,
}

impl<'a, C: Unsigned + NonZero> Iter<'a, C> {
    fn new(scores: &'a StripedScores<C>) -> Self {
        Self {
            scores,
            indices: 0..scores.length,
        }
    }

    fn get(&self, i: usize) -> &'a f32 {
        let col = i / self.scores.data.rows();
        let row = i % self.scores.data.rows();
        &self.scores.data[row][col]
    }
}

impl<'a, C: Unsigned + NonZero> Iterator for Iter<'a, C> {
    type Item = &'a f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|i| self.get(i))
    }
}

impl<'a, C: Unsigned + NonZero> ExactSizeIterator for Iter<'a, C> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, C: Unsigned + NonZero> FusedIterator for Iter<'a, C> {}

impl<'a, C: Unsigned + NonZero> DoubleEndedIterator for Iter<'a, C> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.indices.next_back().map(|i| self.get(i))
    }
}
