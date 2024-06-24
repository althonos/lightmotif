//! Scanner implementation using a fixed block size for the scores.
use std::cmp::Ordering;

use super::abc::Alphabet;
use super::pli::dispatch::Dispatch;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;
use crate::num::StrictlyPositive;
use crate::pli::Maximum;
use crate::pli::Pipeline;
use crate::pli::Score;
use crate::pli::Threshold;
use crate::pwm::DiscreteMatrix;
use crate::scores::StripedScores;

#[cfg(target_arch = "x86_64")]
type _DefaultColumns = typenum::consts::U32;
#[cfg(any(target_arch = "x86", target_arch = "arm", target_arch = "aarch64"))]
type _DefaultColumns = typenum::consts::U16;
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64"
)))]
type _DefaultColumns = typenum::consts::U1;

/// The default column number used in scanners.
pub type DefaultColumns = _DefaultColumns;

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
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub position: usize,
    pub score: f32,
}

impl Hit {
    /// Create a new hit.
    pub fn new(position: usize, score: f32) -> Self {
        assert!(!score.is_nan());
        Self { position, score }
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.score.partial_cmp(&other.score)? {
            Ordering::Equal => self.position.partial_cmp(&other.position),
            other => Some(other),
        }
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// A scanner for iterating over scoring matrix hits in a sequence.
#[derive(Debug)]
pub struct Scanner<'a, A: Alphabet, C: StrictlyPositive = DefaultColumns> {
    pssm: &'a ScoringMatrix<A>,
    dm: DiscreteMatrix<A>,
    seq: &'a StripedSequence<A, C>,
    scores: CowMut<'a, StripedScores<f32, C>>,
    dscores: StripedScores<u8, C>,
    threshold: f32,
    block_size: usize,
    row: usize,
    hits: Vec<Hit>,
    pipeline: Pipeline<A, Dispatch>,
}

impl<'a, A: Alphabet, C: StrictlyPositive> Scanner<'a, A, C> {
    /// Create a new scanner for the given matrix and sequence.
    pub fn new(pssm: &'a ScoringMatrix<A>, seq: &'a StripedSequence<A, C>) -> Self {
        Self {
            pssm,
            seq,
            dm: pssm.to_discrete(),
            scores: CowMut::Owned(StripedScores::empty()),
            dscores: StripedScores::empty(),
            threshold: 0.0,
            block_size: 256,
            row: 0,
            hits: Vec::new(),
            pipeline: Pipeline::dispatch(),
        }
    }

    /// Use the given `StripedScores` as a buffer.
    pub fn scores(&mut self, scores: &'a mut StripedScores<f32, C>) -> &mut Self {
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

impl<'a, A, C> Iterator for Scanner<'a, A, C>
where
    A: Alphabet,
    C: StrictlyPositive,
    Pipeline<A, Dispatch>: Score<u8, A, C> + Threshold<u8, C> + Maximum<u8, C>,
{
    type Item = Hit;
    fn next(&mut self) -> Option<Self::Item> {
        let t = self.dm.scale(self.threshold);
        while self.hits.is_empty() && self.row < self.seq.matrix().rows() {
            // compute the row slice to score in the striped sequence matrix
            let end = (self.row + self.block_size)
                .min(self.seq.matrix().rows().saturating_sub(self.seq.wrap()));
            // score the row slice
            self.pipeline
                .score_rows_into(&self.dm, &self.seq, self.row..end, &mut self.dscores);
            // check if any position is higher than the discrete threshold.
            if self.pipeline.max(&self.dscores).unwrap() >= t {
                // scan through the positions above discrete threshold and recompute
                // scores in floating-point to see if they pass the real threshold.
                for c in self.pipeline.threshold(&self.dscores, t) {
                    let index =
                        c.col * (self.seq.matrix().rows() - self.seq.wrap()) + self.row + c.row;
                    let score = self.pssm.score_position(&self.seq, index);
                    if score >= self.threshold {
                        self.hits.push(Hit::new(index, score));
                    }
                }
            }
            // Proceed to the next block.
            self.row += self.block_size;
        }
        self.hits.pop()
    }

    fn max(mut self) -> Option<Self::Item> {
        // Compute the score of the best hit not yet returned, and translate
        // the `f32` score threshold into a discrete, under-estimate `u8`
        // threshold.
        let mut best = std::mem::take(&mut self.hits)
            .into_iter()
            .filter(|hit| hit.score >= self.threshold)
            .max_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        let mut best_discrete = match &best {
            Some(hit) => self.dm.scale(hit.score),
            None => self.dm.scale(self.threshold),
        };

        // Cache th number of sequence rows in the striped sequence matrix.
        let sequence_rows = self.seq.matrix().rows() - self.seq.wrap();

        // Process all rows of the sequence and record the local
        while self.row < self.seq.matrix().rows() {
            // Score the rows of the current block.
            let end = (self.row + self.block_size).min(sequence_rows);
            self.pipeline
                .score_rows_into(&self.dm, &self.seq, self.row..end, &mut self.dscores);
            // Check if the highest score in the block is high enough to be
            // a new global maximum
            if self.pipeline.max(&self.dscores).unwrap() >= best_discrete {
                // Iterate over candidate position in `u8` scores and recalculate
                // scores for candidates passing the threshold.
                for c in self.pipeline.threshold(&self.dscores, best_discrete) {
                    let dscore = self.dscores.matrix()[c];
                    if dscore >= best_discrete {
                        let index = c.col * sequence_rows + self.row + c.row;
                        let score = self.pssm.score_position(&self.seq, index);
                        if let Some(hit) = &best {
                            if (score > hit.score) | (score == hit.score && index > hit.position) {
                                best = Some(Hit::new(index, score));
                                best_discrete = dscore;
                            }
                        } else {
                            best = Some(Hit::new(index, score))
                        }
                    }
                }
            }
            // Proceed to the next block.
            self.row += self.block_size;
        }
        best
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::abc::Dna;
    use crate::pli::Stripe;
    use crate::pwm::CountMatrix;
    use crate::pwm::ScoringMatrix;
    use crate::seq::EncodedSequence;

    const SEQUENCE: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
    const PATTERNS: &[&str] = &["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"];

    fn seq<C: StrictlyPositive>() -> StripedSequence<Dna, C> {
        let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
        Pipeline::generic().stripe(encoded)
    }

    fn pssm() -> ScoringMatrix<Dna> {
        let cm = CountMatrix::<Dna>::from_sequences(
            PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
        )
        .unwrap();
        let pbm = cm.to_freq(0.1);
        let pwm = pbm.to_weight(None);
        pwm.to_scoring()
    }

    #[test]
    fn collect() {
        let pssm = self::pssm();
        let mut striped = self::seq();
        striped.configure(&pssm);

        let mut scanner = Scanner::new(&pssm, &striped);
        scanner.threshold(-10.0);

        let mut hits = scanner.collect::<Vec<_>>();
        assert_eq!(hits.len(), 3);

        hits.sort_by_key(|hit| hit.position);
        assert_eq!(hits[0].position, 18);
        assert_eq!(hits[1].position, 27);
        assert_eq!(hits[2].position, 32);
    }

    #[test]
    fn max() {
        let pssm = self::pssm();
        let mut striped = self::seq();
        striped.configure(&pssm);

        let mut scanner = Scanner::new(&pssm, &striped);
        scanner.threshold(-10.0);

        let hit = scanner.max().unwrap();
        assert!(
            (hit.score - -5.50167).abs() < 1e-5,
            "{} != {}",
            hit.score,
            -5.50167
        );
        assert_eq!(hit.position, 18);
    }
}
