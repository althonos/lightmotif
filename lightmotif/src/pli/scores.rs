use std::iter::DoubleEndedIterator;
use std::iter::ExactSizeIterator;
use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::Range;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

use crate::dense::DenseMatrix;

/// Striped matrix storing scores for an equally striped sequence.
#[derive(Clone, Debug)]
pub struct StripedScores<C: Unsigned + NonZero> {
    data: DenseMatrix<f32, C>,
    length: usize,
}

impl<C: Unsigned + NonZero> StripedScores<C> {
    /// Create a new striped score matrix with the given length and data.
    pub fn new(length: usize, data: DenseMatrix<f32, C>) -> Self {
        Self { length, data }
    }

    /// Create an empty score matrix with the given length and row count.
    pub fn empty() -> Self {
        Self::new(0, DenseMatrix::new(0))
    }

    /// Return the number of scored positions.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return whether the scores are empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Return a reference to the striped matrix storing the scores.
    pub fn matrix(&self) -> &DenseMatrix<f32, C> {
        &self.data
    }

    /// Return a mutable reference to the striped matrix storing the scores.
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        &mut self.data
    }

    /// Resize the striped scores storage to the given length and row number.
    #[doc(hidden)]
    pub fn resize(&mut self, length: usize, rows: usize) {
        self.length = length;
        self.data.resize(rows);
    }

    /// Iterate over scores of individual sequence positions.
    pub fn iter(&self) -> Iter<'_, C> {
        Iter::new(self)
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

impl<C: Unsigned + NonZero> Default for StripedScores<C> {
    fn default() -> Self {
        StripedScores::empty()
    }
}

impl<C: Unsigned + NonZero> Index<usize> for StripedScores<C> {
    type Output = f32;
    #[inline]
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
