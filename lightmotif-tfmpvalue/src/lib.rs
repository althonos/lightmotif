#![doc = include_str!("../README.md")]

use std::cmp::Ordering;
#[cfg(not(feature = "fnv"))]
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::RangeInclusive;

use lightmotif::abc::Alphabet;
use lightmotif::dense::DenseMatrix;
use lightmotif::num::Unsigned;
use lightmotif::pwm::ScoringMatrix;

#[cfg(feature = "fnv")]
use fnv::FnvHashMap;

#[cfg(feature = "fnv")]
type Map<K, V> = FnvHashMap<K, V>;
#[cfg(not(feature = "fnv"))]
type Map<K, V> = HashMap<K, V>;

/// The TFMPvalue algorithm.
#[derive(Debug)]
pub struct TfmPvalue<A: Alphabet, M: AsRef<ScoringMatrix<A>>> {
    /// A reference to the original scoring matrix.
    matrix: M,
    /// The granularity with which the round matrix has been built.
    granularity: f64,
    /// The round matrix offsets.
    offsets: Vec<i64>,
    /// Rescaled PSSM in integer space.
    int_matrix: DenseMatrix<i64, A::K>,
    /// The maximum error caused by integer rescale.
    error_max: f64,
    /// The maximum integer score reachable at each row of the matrix.
    max_score_rows: Vec<i64>,
    /// The minimum integer score reachable at each row of the matrix.
    min_score_rows: Vec<i64>,
}

#[allow(non_snake_case)]
impl<A: Alphabet, M: AsRef<ScoringMatrix<A>>> TfmPvalue<A, M> {
    /// Initialize the TFM-Pvalue algorithm for the given scoring matrix.
    pub fn new(matrix: M) -> Self {
        let m = matrix.as_ref();
        let M = m.len();
        Self {
            granularity: f64::NAN,
            matrix,
            offsets: vec![0; M],
            int_matrix: DenseMatrix::new(M),
            max_score_rows: vec![0; M],
            min_score_rows: vec![0; M],
            error_max: 0.0,
        }
    }

    /// Return a reference to the wrapped matrix reference.
    pub fn as_inner(&self) -> &M {
        &self.matrix
    }

    /// Extract the wrapped matrix reference.
    pub fn into_inner(self) -> M {
        self.matrix
    }

    /// Compute the approximate score matrix with the given granularity.
    fn recompute(&mut self, granularity: f64) {
        assert!(granularity < 1.0);
        let matrix = self.matrix.as_ref();
        let M: usize = matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // compute effective granularity
        self.granularity = granularity;

        // compute integer matrix
        for i in 0..M {
            for j in 0..K - 1 {
                self.int_matrix[i][j] = (matrix[i][j] as f64 / self.granularity).floor() as i64;
            }
        }

        // compute maximum error by summing max error at each row
        self.error_max = 0.0;
        for i in 1..M {
            let max_e = matrix[i]
                .iter()
                .enumerate()
                .map(|(j, &x)| (x as f64) / self.granularity - self.int_matrix[i][j] as f64)
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Less))
                .unwrap();
            self.error_max += max_e;
        }

        // TODO: sort columns?

        // compute offsets
        for i in 0..M {
            self.offsets[i] = -*self.int_matrix[i][..K - 1].iter().min().unwrap();
            for j in 0..K - 1 {
                self.int_matrix[i][j] += self.offsets[i];
            }
        }

        // look for the minimum score of the matrix for each row
        for i in 0..M {
            self.min_score_rows[i] = *self.int_matrix[i][..K - 1].iter().min().unwrap();
            self.max_score_rows[i] = *self.int_matrix[i][..K - 1].iter().max().unwrap();
        }
    }

    /// Compute the score distribution between `min` and `max`.
    fn distribution(&self, min: i64, max: i64) -> Vec<Map<i64, f64>> {
        let matrix = self.matrix.as_ref();
        let M: usize = matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // background frequencies
        let bg = matrix.background().frequencies();

        // maps for each steps of the computation
        let mut qvalues = vec![Map::default(); M + 1];

        // maximum score reachable with the suffix matrix from i to M-1
        let mut maxs = vec![0; M + 1];
        for i in (0..M).rev() {
            maxs[i] = maxs[i + 1] + self.max_score_rows[i];
        }

        // initialize the map at first position with background frequencies
        for k in 0..K - 1 {
            if self.int_matrix[0][k] + maxs[1] >= min {
                *qvalues[0].entry(self.int_matrix[0][k]).or_default() += bg[k] as f64;
            }
        }

        // compute q values for scores greater or equal to min
        qvalues[M - 1].insert(max + 1, 0.0);
        for pos in 1..M {
            // split the array in two to make the borrow checker happy
            let (l, r) = qvalues.split_at_mut(pos);
            // iterate on every reachable score at the current position
            for key in l[pos - 1].keys() {
                for k in 0..K - 1 {
                    let sc = key + self.int_matrix[pos][k];
                    if sc + maxs[pos + 1] >= min {
                        // the score min can be reached
                        let occ = l[pos - 1][&key] * bg[k] as f64;
                        if sc > max {
                            // the score will be greater than max for all suffixes
                            *r[M - 1 - pos].entry(max + 1).or_default() += occ;
                        } else {
                            *r[0].entry(sc).or_default() += occ;
                        }
                    }
                }
            }
        }

        qvalues
    }

    /// Search the p-value range for the given score.
    fn lookup_pvalue(&self, score: f64) -> RangeInclusive<f64> {
        assert!(!self.granularity.is_nan());
        let matrix = self.matrix.as_ref();
        let M: usize = matrix.len();

        // Compute the integer score range from the given score.
        let scaled = score / self.granularity + self.offsets.iter().sum::<i64>() as f64;
        let avg = scaled.floor() as i64;
        let max = (scaled + self.error_max + 1.0).floor() as i64;
        let min = (scaled - self.error_max - 1.0).floor() as i64;

        // Compute q values for the given scores
        let qvalues = self.distribution(min, max);

        // Compute p-values
        let mut pvalues = Map::default();
        let mut s = max + 1;
        let mut last = qvalues[M - 1].keys().cloned().collect::<Vec<i64>>();
        last.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());
        let mut sum = qvalues[0].get(&(max + 1)).cloned().unwrap_or_default();
        for &l in last.iter().rev() {
            sum += qvalues[M - 1][&l];
            if l >= avg {
                s = l;
            }
            pvalues.insert(l, sum);
        }

        // Find the p-value range for the requested score
        let mut keys = pvalues.keys().cloned().collect::<Vec<i64>>();
        keys.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());
        let mut kmax = keys.iter().position(|&k| k == s).unwrap();
        while kmax > 0 && keys[kmax] as f64 >= s as f64 - self.error_max {
            kmax -= 1;
        }

        // Return p-value range
        let pmax = pvalues[&keys[kmax]];
        let pmin = pvalues[&s];
        RangeInclusive::new(pmin, pmax)
    }

    /// Search the score and p-value range for a given p-value.
    fn lookup_score(&self, pvalue: f64, range: RangeInclusive<i64>) -> (i64, RangeInclusive<f64>) {
        assert!(!self.granularity.is_nan());
        let matrix = self.matrix.as_ref();
        let M: usize = matrix.len();

        // compute score range for target pvalue
        let min = *range.start();
        let max = *range.end();

        // compute q values
        let qvalues = self.distribution(min, max);
        let mut pvalues = Map::default();

        // find most likely scores at the end of the matrix
        let mut keys = qvalues[M - 1].keys().cloned().collect::<Vec<_>>();
        keys.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());

        // compute pvalues
        let mut sum = 0.0;
        let mut riter = keys.len() - 1;
        let alpha;
        let alpha_e;
        while riter > 0 {
            sum += qvalues[M - 1][&keys[riter]];
            pvalues.insert(keys[riter], sum);
            if sum >= pvalue {
                break;
            }
            riter -= 1;
        }

        if sum > pvalue {
            alpha_e = keys[riter];
            alpha = keys[riter + 1];
        } else {
            if riter == 0 {
                alpha = keys[0];
                alpha_e = keys[0];
            } else {
                alpha = keys[riter];
                alpha_e = keys[riter - 1];
                sum += pvalues.get(&alpha_e).cloned().unwrap_or_default();
            }
            pvalues.insert(alpha_e, sum);
        }

        if (alpha - alpha_e) as f64 > self.error_max {
            (alpha, RangeInclusive::new(pvalues[&alpha], pvalues[&alpha]))
        } else {
            (
                alpha,
                RangeInclusive::new(pvalues[&alpha_e], pvalues[&alpha]),
            )
        }
    }

    /// Compute the exact P-value for the given score.
    ///
    /// # Caution
    /// This method internally calls `approximate_pvalue` without bounds on
    /// the granularity, which may require a very large amount of memory for
    /// some scoring matrices. Use `approximate_pvalue` directly to add
    /// limits on the number of iterations or on the granularity.
    pub fn pvalue(&mut self, score: f64) -> f64 {
        let it = self.approximate_pvalue(score).last().unwrap();
        assert!(it.converged); // algorithm should always converge
        *it.range.start()
    }

    /// Iterate with decreasing granularity to compute an approximate P-value for a score.
    pub fn approximate_pvalue(&mut self, score: f64) -> PvaluesIterator<'_, A, M> {
        PvaluesIterator {
            tfmp: self,
            score,
            decay: 0.1,
            granularity: 0.1,
            target: 0.0,
            converged: false,
        }
    }

    /// Compute the exact score associated with a given P-value.
    ///
    /// # Caution
    /// This method internally calls `approximate_score` without bounds on
    /// the granularity, which may require a very large amount of memory for
    /// some scoring matrices. Use `approximate_score` directly to add
    /// limits on the number of iterations or on the granularity.
    pub fn score(&mut self, pvalue: f64) -> f64 {
        let it = self.approximate_score(pvalue).last().unwrap();
        assert!(it.converged); // algorithm should always converge
        it.score
    }

    /// Iterate with decreasing granularity to compute an approximate score for a P-value.
    pub fn approximate_score(&mut self, pvalue: f64) -> ScoresIterator<'_, A, M> {
        self.recompute(0.1);
        ScoresIterator {
            min: self.min_score_rows.iter().sum(),
            max: self.max_score_rows.iter().sum::<i64>() + (self.error_max + 0.5).ceil() as i64,
            tfmp: self,
            pvalue,
            decay: 0.1,
            granularity: 0.1,
            target: 0.0,
            converged: false,
        }
    }
}

impl<A: Alphabet, M: AsRef<ScoringMatrix<A>>> From<M> for TfmPvalue<A, M> {
    fn from(matrix: M) -> Self {
        Self::new(matrix)
    }
}

/// The result of an iteration of the TFMPvalue algorithm.
#[derive(Debug)]
pub struct Iteration {
    /// The score computed for the current iteration, or the query score
    /// if approximating p-value.
    pub score: f64,
    /// The p-value range for the current iteration.
    pub range: RangeInclusive<f64>,
    /// The granularity with which scores and p-values were computed.
    pub granularity: f64,
    /// A flag to mark whether the approximation converged on this iteration.
    pub converged: bool,
    #[allow(unused)]
    _hidden: (),
}

#[derive(Debug)]
pub struct PvaluesIterator<'tfmp, A: Alphabet, M: AsRef<ScoringMatrix<A>>> {
    tfmp: &'tfmp mut TfmPvalue<A, M>,
    score: f64,
    decay: f64,
    granularity: f64,
    target: f64,
    converged: bool,
}

impl<'tfmp, A: Alphabet, M: AsRef<ScoringMatrix<A>>> Iterator for PvaluesIterator<'tfmp, A, M> {
    type Item = Iteration;
    fn next(&mut self) -> Option<Self::Item> {
        if self.converged || self.granularity <= self.target {
            return None;
        }

        self.tfmp.recompute(self.granularity);
        let granularity = self.granularity;
        let range = self.tfmp.lookup_pvalue(self.score);

        self.granularity *= self.decay;
        if range.start() == range.end() {
            self.converged = true;
        }

        Some(Iteration {
            range,
            granularity,
            converged: self.converged,
            score: self.score,
            _hidden: (),
        })
    }
}

#[derive(Debug)]
pub struct ScoresIterator<'tfmp, A: Alphabet, M: AsRef<ScoringMatrix<A>>> {
    tfmp: &'tfmp mut TfmPvalue<A, M>,
    pvalue: f64,
    decay: f64,
    granularity: f64,
    target: f64,
    converged: bool,
    min: i64,
    max: i64,
}

impl<'tfmp, A: Alphabet, M: AsRef<ScoringMatrix<A>>> Iterator for ScoresIterator<'tfmp, A, M> {
    type Item = Iteration;
    fn next(&mut self) -> Option<Self::Item> {
        if self.converged || self.granularity <= self.target {
            return None;
        }

        self.tfmp.recompute(self.granularity);
        let granularity = self.granularity;
        let (iscore, range) = self
            .tfmp
            .lookup_score(self.pvalue, RangeInclusive::new(self.min, self.max));

        self.granularity *= self.decay;
        self.min =
            ((iscore as f64 - (self.tfmp.error_max + 0.5).ceil()) / self.decay).floor() as i64;
        self.max =
            ((iscore as f64 + (self.tfmp.error_max + 0.5).ceil()) / self.decay).floor() as i64;
        if range.start() == range.end() {
            self.converged = true;
        }

        let offset = self.tfmp.offsets.iter().sum::<i64>();
        Some(Iteration {
            granularity,
            range,
            score: (iscore - offset) as f64 * granularity,
            converged: self.converged,
            _hidden: (),
        })
    }
}

#[cfg(test)]
mod test {

    use lightmotif::abc::Dna;
    use lightmotif::dense::DenseMatrix;
    use lightmotif::pwm::CountMatrix;

    use super::*;

    macro_rules! assert_almost_eq {
        ($x:expr, $y:expr, places = $places:expr) => {{
            assert_eq!(
                ($x * 10f64.powi($places)).round(),
                ($y * 10f64.powi($places)).round(),
            )
        }};
    }

    /// Build the MA0045 PSSM from Jaspar using a uniform background and 0.25 pseudocounts.
    fn build_ma0045() -> ScoringMatrix<Dna> {
        #[rustfmt::skip]
        let counts = DenseMatrix::from_rows([
            //A   C   T   G   N
            [ 3,  5,  2,  4,  0],
            [ 7,  0,  4,  3,  0],
            [ 9,  1,  3,  1,  0],
            [ 3,  6,  1,  4,  0],
            [11,  0,  0,  3,  0],
            [11,  0,  1,  2,  0],
            [11,  0,  1,  2,  0],
            [ 3,  3,  6,  2,  0],
            [ 4,  1,  1,  8,  0],
            [ 3,  4,  1,  6,  0],
            [ 8,  5,  0,  1,  0],
            [ 8,  1,  1,  4,  0],
            [ 9,  0,  3,  2,  0],
            [ 9,  5,  0,  0,  0],
            [11,  0,  0,  3,  0],
            [ 2,  7,  5,  0,  0],
        ]);
        CountMatrix::new(counts)
            .unwrap()
            .to_freq(0.25)
            .to_scoring(None)
    }

    #[test]
    fn approximate_pvalue() {
        let pssm = build_ma0045();
        let mut tfmp = TfmPvalue::new(&pssm);
        let mut pvalues = tfmp.approximate_pvalue(10.0);

        // Reference values computed with pytfmpval:
        //
        // granularity  pmin                    pmax
        //         0.1  5.7484256103634834e-05  0.000185822369530797
        //        0.01  0.00011981534771621227  0.00012914929538965225
        //       0.001  0.00012489012442529202  0.0001261131837964058
        //      0.0001  0.00012567872181534767  0.00012605986557900906
        //       1e-05  0.00012601236812770367  0.0001260137651115656
        //       1e-06  0.00012601329945027828  0.0001260137651115656
        //       1e-07  0.00012601329945027828  0.00012601329945027828

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-1, places = 5);
        assert_almost_eq!(it.range.start(), 5.74842561e-5, places = 7);
        assert_almost_eq!(it.range.end(), 0.000185822369, places = 7);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-2, places = 7);
        assert_almost_eq!(it.range.start(), 0.000119815, places = 5);
        assert_almost_eq!(it.range.end(), 0.000129149, places = 7);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-3, places = 5);
        assert_almost_eq!(it.range.start(), 0.000124890, places = 7);
        assert_almost_eq!(it.range.end(), 0.000126113, places = 7);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-4, places = 5);
        assert_almost_eq!(it.range.start(), 0.00012567, places = 5);
        assert_almost_eq!(it.range.end(), 0.000126059, places = 5);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-5, places = 5);
        assert_almost_eq!(it.range.start(), 0.00012601, places = 5);
        assert_almost_eq!(it.range.end(), 0.0001260137, places = 5);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-6, places = 5);
        assert_almost_eq!(it.range.start(), 0.00012601, places = 5);
        assert_almost_eq!(it.range.end(), 0.0001260137, places = 5);
        assert!(!it.converged);

        let it = pvalues.next().unwrap();
        assert_almost_eq!(it.granularity, 1e-7, places = 5);
        assert_almost_eq!(it.range.start(), 0.0001260, places = 5);
        assert_almost_eq!(it.range.end(), 0.0001260132, places = 5);
        assert!(it.converged);

        assert!(pvalues.next().is_none());
    }

    #[test]
    fn pvalue() {
        let pssm = build_ma0045();
        let mut tfmp = TfmPvalue::new(&pssm);

        assert_almost_eq!(tfmp.pvalue(8.882756), 0.0003, places = 5);
        assert_almost_eq!(tfmp.pvalue(12.657785), 0.00001, places = 5);
        assert_almost_eq!(tfmp.pvalue(19.1), 1e-10, places = 5);
    }

    #[test]
    fn score() {
        let pssm = build_ma0045();
        let mut tfmp = TfmPvalue::new(&pssm);

        assert_almost_eq!(tfmp.score(0.00001), 12.657785, places = 4);
        assert_almost_eq!(tfmp.score(0.0003), 8.882756, places = 5);
        assert_almost_eq!(tfmp.score(1e-10), 19.1, places = 5);
    }
}
