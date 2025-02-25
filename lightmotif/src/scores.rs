//! Wrapper types for storing scores.

use std::fmt::Debug;
use std::iter::FusedIterator;
use std::ops::Deref;
use std::ops::Index;
use std::ops::Range;

use crate::abc::Dna;
use crate::dense::DefaultColumns;
use crate::dense::DenseMatrix;
use crate::dense::MatrixCoordinates;
use crate::dense::MatrixElement;
use crate::err::InvalidData;
use crate::num::PositiveLength;
use crate::pli::dispatch::Dispatch;
use crate::pli::Maximum;
use crate::pli::Pipeline;
use crate::pli::Threshold;

// --- Scores ------------------------------------------------------------------

/// Simple vector storing scores for a score sequence.
#[derive(Clone, Debug)]
pub struct Scores<T> {
    /// The raw vector storing the scores.
    data: Vec<T>,
}

impl<T> Scores<T> {
    /// Create a new collection from an array of scores.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: PartialOrd> Scores<T> {
    /// Find the position with the highest score.
    pub fn argmax(&self) -> Option<usize> {
        self.data
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i)
    }

    /// Return the positions with score equal to or greater than the threshold.
    pub fn threshold(&self, threshold: &T) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, x)| x >= &threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

impl<T: PartialOrd + Clone> Scores<T> {
    /// Find the highest score.
    pub fn max(&self) -> Option<T> {
        self.data
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .cloned()
    }
}

impl<T> AsRef<Vec<T>> for Scores<T> {
    #[inline]
    fn as_ref(&self) -> &Vec<T> {
        &self.data
    }
}

impl<T> Deref for Scores<T> {
    type Target = Vec<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> From<Vec<T>> for Scores<T> {
    #[inline]
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl<T> From<Scores<T>> for Vec<T> {
    #[inline]
    fn from(scores: Scores<T>) -> Self {
        scores.data
    }
}

// --- StripedScores -----------------------------------------------------------

/// Striped matrix storing scores for an equally striped sequence.
#[derive(Clone)]
pub struct StripedScores<T: MatrixElement, C: PositiveLength = DefaultColumns> {
    /// The raw data matrix storing the scores.
    data: DenseMatrix<T, C>,
    /// The total length of the `StripedSequence` these scores were obtained from.
    max_index: usize,
}

impl<T: MatrixElement, C: PositiveLength> StripedScores<T, C> {
    /// Create a new striped score matrix with the given length and data.
    #[inline]
    fn new(data: DenseMatrix<T, C>, max_index: usize) -> Result<Self, InvalidData> {
        Ok(Self { data, max_index })
    }

    /// Create an empty buffer to store striped scores.
    #[inline]
    pub fn empty() -> Self {
        Self::new(DenseMatrix::new(0), 0).unwrap()
    }

    /// The maximum sequence index (the length of the scored sequence).
    #[inline]
    pub fn max_index(&self) -> usize {
        self.max_index
    }

    /// Return whether the scores are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.rows() == 0
    }

    /// Return a reference to the striped matrix storing the scores.
    #[inline]
    pub fn matrix(&self) -> &DenseMatrix<T, C> {
        &self.data
    }

    /// Return a mutable reference to the striped matrix storing the scores.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<T, C> {
        &mut self.data
    }

    /// Resize the striped scores storage for the given range of rows.
    #[inline]
    pub fn resize(&mut self, rows: usize, max_index: usize) {
        self.data.resize(rows);
        self.max_index = max_index;
    }

    /// Convert coordinates into a column-major offset.
    #[inline]
    pub fn offset(&self, mc: MatrixCoordinates) -> usize {
        mc.col * self.data.rows() + mc.row
    }

    /// Iterate over scores of individual sequence positions.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, C> {
        Iter::new(self)
    }

    /// Convert the striped scores into an array.
    #[inline]
    pub fn unstripe(&self) -> Scores<T> {
        self.iter().cloned().collect::<Vec<T>>().into()
    }
}

impl<T: MatrixElement + PartialOrd, C: PositiveLength> StripedScores<T, C>
where
    Pipeline<Dna, Dispatch>: Maximum<T, C>,
{
    /// Find the highest score.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    #[inline]
    pub fn max(&self) -> Option<T> {
        Pipeline::dispatch().max(self)
    }

    /// Find the position with the highest score.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    #[inline]
    pub fn argmax(&self) -> Option<usize> {
        Pipeline::dispatch().argmax(self).map(|mc| self.offset(mc))
    }
}

impl<T: MatrixElement + PartialOrd, C: PositiveLength> StripedScores<T, C>
where
    Pipeline<Dna, Dispatch>: Threshold<T, C>,
{
    /// Return the positions with score equal to or greater than the threshold.
    ///
    /// The indices are not necessarily returned in a particular order,
    /// since different implementations use a different internal memory
    /// representation.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn threshold(&self, threshold: T) -> Vec<usize> {
        Pipeline::dispatch()
            .threshold(self, threshold)
            .into_iter()
            .map(|m| self.offset(m))
            .collect()
    }
}

impl<T: MatrixElement, C: PositiveLength> AsRef<DenseMatrix<T, C>> for StripedScores<T, C> {
    #[inline]
    fn as_ref(&self) -> &DenseMatrix<T, C> {
        self.matrix()
    }
}

impl<T: MatrixElement, C: PositiveLength> AsMut<DenseMatrix<T, C>> for StripedScores<T, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut DenseMatrix<T, C> {
        self.matrix_mut()
    }
}

impl<T: MatrixElement + Debug, C: PositiveLength> std::fmt::Debug for StripedScores<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("StripedSequence")
            .field("data", &self.data)
            .field("max_index", &self.max_index)
            .finish()
    }
}

impl<T: MatrixElement, C: PositiveLength> Default for StripedScores<T, C> {
    #[inline]
    fn default() -> Self {
        StripedScores::empty()
    }
}

impl<T: MatrixElement, C: PositiveLength> Index<usize> for StripedScores<T, C> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        let col = index / self.data.rows();
        let row = index % self.data.rows();
        &self.data[row][col]
    }
}

impl<T: MatrixElement, C: PositiveLength> From<StripedScores<T, C>> for Vec<T> {
    #[inline]
    fn from(scores: StripedScores<T, C>) -> Self {
        scores.iter().cloned().collect()
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, T: MatrixElement, C: PositiveLength> {
    scores: &'a StripedScores<T, C>,
    indices: Range<usize>,
}

impl<'a, T: MatrixElement, C: PositiveLength> Iter<'a, T, C> {
    #[inline]
    fn new(scores: &'a StripedScores<T, C>) -> Self {
        // Compute the last index
        let end = scores
            .max_index
            .min(scores.data.rows() * scores.data.columns());
        // Create the iterator
        let indices = 0..end;
        Self { scores, indices }
    }

    #[inline]
    fn get(&self, i: usize) -> &'a T {
        let col = i / self.scores.data.rows();
        let row = i % self.scores.data.rows();
        &self.scores.data[row][col]
    }
}

impl<'a, T: MatrixElement, C: PositiveLength> Iterator for Iter<'a, T, C> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|i| self.get(i))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T: MatrixElement, C: PositiveLength> ExactSizeIterator for Iter<'_, T, C> {
    #[inline]
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<T: MatrixElement, C: PositiveLength> FusedIterator for Iter<'_, T, C> {}

impl<T: MatrixElement, C: PositiveLength> DoubleEndedIterator for Iter<'_, T, C> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.indices.next_back().map(|i| self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::num::U4;

    #[test]
    fn iter() {
        let data = DenseMatrix::<f32, U4>::new(6);
        let scores = StripedScores::new(data, 22).unwrap();
        assert_eq!(scores.unstripe().len(), 22);

        let data = DenseMatrix::<f32, U4>::new(3);
        let scores = StripedScores::new(data, 10).unwrap();
        assert_eq!(scores.unstripe().len(), 10);
    }
}
