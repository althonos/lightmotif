use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::Range;

use crate::abc::Dna;
use crate::dense::DenseMatrix;
use crate::err::InvalidData;
use crate::num::StrictlyPositive;
use crate::pli::Pipeline;

use super::dispatch::Dispatch;
use super::Maximum;
use super::Threshold;

/// Striped matrix storing scores for an equally striped sequence.
#[derive(Clone, Debug)]
pub struct StripedScores<C: StrictlyPositive> {
    /// The raw data matrix storing the scores.
    data: DenseMatrix<f32, C>,
    /// The range of rows over the `StripedSequence` these scores were obtained from.
    range: Range<usize>,
    /// The total length of the `StripedSequence` these scores were obtained from.
    max_index: usize,
}

impl<C: StrictlyPositive> StripedScores<C> {
    /// Create a new striped score matrix with the given length and data.
    fn new(
        data: DenseMatrix<f32, C>,
        range: Range<usize>,
        max_index: usize,
    ) -> Result<Self, InvalidData> {
        if data.rows() != range.len() {
            Err(InvalidData)
        } else {
            Ok(Self {
                data,
                range,
                max_index,
            })
        }
    }

    /// Create an empty buffer to store striped scores.
    pub fn empty() -> Self {
        Self::new(DenseMatrix::new(0), 0..0, 0).unwrap()
    }

    /// Get the rows for which these scores were computed.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// FIXME: remove
    pub fn sequence_length(&self) -> usize {
        self.max_index
    }

    pub fn max_index(&self) -> usize {
        self.max_index
    }

    /// Return whether the scores are empty.
    pub fn is_empty(&self) -> bool {
        self.data.rows() == 0
    }

    /// Return a reference to the striped matrix storing the scores.
    pub fn matrix(&self) -> &DenseMatrix<f32, C> {
        &self.data
    }

    /// Return a mutable reference to the striped matrix storing the scores.
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        &mut self.data
    }

    /// Resize the striped scores storage for the given range of rows.
    #[doc(hidden)]
    pub fn resize(&mut self, range: Range<usize>, max_index: usize) {
        self.data.resize(range.len());
        self.range = range;
        self.max_index = max_index;
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

impl<C: StrictlyPositive> StripedScores<C>
where
    Pipeline<Dna, Dispatch>: Maximum<C>,
{
    /// Find the highest score.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn max(&self) -> Option<f32> {
        Pipeline::dispatch().max(self)
    }

    /// Find the position with the highest score.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn argmax(&self) -> Option<usize> {
        Pipeline::dispatch().argmax(self)
    }
}

impl<C: StrictlyPositive> StripedScores<C>
where
    Pipeline<Dna, Dispatch>: Threshold<C>,
{
    /// Return the positions with score equal to or greater than the threshold.
    ///
    /// The indices are not necessarily returned in a particular order,
    /// since different implementations use a different internal memory
    /// representation. The indices are corrected by offset depending on
    /// the range for which the scores were computed.
    ///
    /// # Note
    /// Uses platform-accelerated implementation when available.
    pub fn threshold(&self, threshold: f32) -> Vec<usize> {
        Pipeline::dispatch().threshold(self, threshold)
    }
}

impl<C: StrictlyPositive> AsRef<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_ref(&self) -> &DenseMatrix<f32, C> {
        self.matrix()
    }
}

impl<C: StrictlyPositive> AsMut<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        self.matrix_mut()
    }
}

impl<C: StrictlyPositive> Default for StripedScores<C> {
    fn default() -> Self {
        StripedScores::empty()
    }
}

impl<C: StrictlyPositive> Index<usize> for StripedScores<C> {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        let col = index / self.data.rows();
        let row = index % self.data.rows();
        &self.data[row][col]
    }
}

impl<C: StrictlyPositive> From<StripedScores<C>> for Vec<f32> {
    fn from(scores: StripedScores<C>) -> Self {
        scores.iter().cloned().collect()
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, C: StrictlyPositive> {
    scores: &'a StripedScores<C>,
    indices: Range<usize>,
}

impl<'a, C: StrictlyPositive> Iter<'a, C> {
    fn new(scores: &'a StripedScores<C>) -> Self {
        // Compute number of rows of source sequence
        let seq_rows = (scores.max_index + C::USIZE - 1) / C::USIZE;
        let offset = seq_rows * (C::USIZE - 1) + scores.range.start;

        // Check if the rows range contains some unneeded indices
        let mut i = scores.range.len() * C::USIZE;
        let end = scores.range.end * C::USIZE;
        let indices = if end + offset >= scores.max_index {
            i -= end - scores.max_index;
            0..i
        } else {
            0..i
        };

        // Create the iterator
        Self { scores, indices }
    }

    fn get(&self, i: usize) -> &'a f32 {
        let col = i / self.scores.data.rows();
        let row = i % self.scores.data.rows();
        &self.scores.data[row][col]
    }
}

impl<'a, C: StrictlyPositive> Iterator for Iter<'a, C> {
    type Item = &'a f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|i| self.get(i))
    }
}

impl<'a, C: StrictlyPositive> ExactSizeIterator for Iter<'a, C> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, C: StrictlyPositive> FusedIterator for Iter<'a, C> {}

impl<'a, C: StrictlyPositive> DoubleEndedIterator for Iter<'a, C> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.indices.next_back().map(|i| self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::num::U4;

    #[test]
    fn test_iter() {
        let data = DenseMatrix::<f32, U4>::new(6);
        let scores = StripedScores::new(data, 0..6, 22).unwrap();
        assert_eq!(scores.to_vec().len(), 22);

        let data = DenseMatrix::<f32, U4>::new(3);
        let scores = StripedScores::new(data, 3..6, 22).unwrap();
        assert_eq!(scores.to_vec().len(), 10);
    }
}
