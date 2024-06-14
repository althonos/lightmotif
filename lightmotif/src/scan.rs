//! Scanner implementation using a fixed block size for the scores.
use super::abc::Alphabet;
use super::pli::dispatch::Dispatch;
use super::pli::platform::Backend;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;
use crate::pli::Maximum;
use crate::pli::Pipeline;
use crate::pli::Score;
use crate::pli::Threshold;
use crate::scores::StripedScores;

type C = <Dispatch as Backend>::LANES;

#[derive(Debug)]
enum CowMut<'a, T> {
    Owned(T),
    Borrowed(&'a mut T),
}

impl<T> std::ops::Deref for CowMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            CowMut::Owned(it) => it,
            CowMut::Borrowed(it) => *it,
        }
    }
}

impl<T> std::ops::DerefMut for CowMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            CowMut::Owned(it) => it,
            CowMut::Borrowed(it) => *it,
        }
    }
}

impl<T: Default> Default for CowMut<'_, T> {
    fn default() -> Self {
        CowMut::Owned(T::default())
    }
}

/// A hit describing a scored position somewhere in the sequence.
#[derive(Debug, Clone)]
pub struct Hit {
    pub position: usize,
    pub score: f32,
}

impl Hit {
    /// Create a new hit.
    pub fn new(position: usize, score: f32) -> Self {
        Self { position, score }
    }
}

#[derive(Debug)]
pub struct Scanner<'a, A: Alphabet> {
    pssm: &'a ScoringMatrix<A>,
    seq: &'a StripedSequence<A, C>,
    scores: CowMut<'a, StripedScores<C>>,
    threshold: f32,
    block_size: usize,
    row: usize,
    hits: Vec<Hit>,
}

impl<'a, A: Alphabet> Scanner<'a, A> {
    /// Create a new scanner for the given matrix and sequence.
    pub fn new(pssm: &'a ScoringMatrix<A>, seq: &'a StripedSequence<A, C>) -> Self {
        Self {
            pssm,
            seq,
            scores: CowMut::Owned(StripedScores::empty()),
            threshold: 0.0,
            block_size: 512,
            row: 0,
            hits: Vec::new(),
        }
    }

    /// Use the given `StripedScores` as a buffer.
    pub fn scores(&mut self, scores: &'a mut StripedScores<C>) -> &mut Self {
        self.scores = CowMut::Borrowed(scores);
        self
    }

    /// Change the block size for the scanner.
    pub fn block_size(&mut self, block_size: usize) -> &mut Self {
        self.block_size = block_size;
        self
    }

    /// Change the threshold for the scanner.
    pub fn threshold(&mut self, threshold: f32) -> &mut Self {
        self.threshold = threshold;
        self
    }
}

impl<'a, A: Alphabet> Scanner<'a, A>
where
    Pipeline<A, Dispatch>: Score<A, C>,
{
    /// Consume the scanner to find the best hit.
    pub fn best(&mut self) -> Option<Hit> {
        let pli = Pipeline::dispatch();
        let mut best = std::mem::take(&mut self.hits)
            .into_iter()
            .max_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        while self.row < self.seq.matrix().rows() {
            let end = (self.row + self.block_size).min(self.seq.matrix().rows() - self.seq.wrap());
            pli.score_rows_into(&self.pssm, &self.seq, self.row..end, &mut self.scores);
            let matrix = self.scores.matrix();
            if let Some(c) = pli.argmax(&self.scores) {
                let score = matrix[c];
                if best
                    .as_ref()
                    .map(|hit: &Hit| matrix[c] >= hit.score)
                    .unwrap_or(true)
                {
                    let index =
                        c.col * (self.seq.matrix().rows() - self.seq.wrap()) + self.row + c.row;
                    best = Some(Hit::new(index, score));
                }
            }
            self.row += self.block_size;
        }
        best
    }
}

impl<'a, A: Alphabet> Iterator for Scanner<'a, A>
where
    Pipeline<A, Dispatch>: Score<A, C>,
{
    type Item = Hit;
    fn next(&mut self) -> Option<Self::Item> {
        while self.hits.is_empty() && self.row < self.seq.matrix().rows() {
            let pli = Pipeline::dispatch();
            let end = (self.row + self.block_size).min(self.seq.matrix().rows() - self.seq.wrap());
            pli.score_rows_into(&self.pssm, &self.seq, self.row..end, &mut self.scores);
            let matrix = self.scores.matrix();
            for c in pli.threshold(&self.scores, self.threshold) {
                let index = c.col * (self.seq.matrix().rows() - self.seq.wrap()) + self.row + c.row;
                self.hits.push(Hit::new(index, matrix[c]));
            }
            self.row += self.block_size;
        }
        self.hits.pop()
    }
}
