#![doc = include_str!("../README.md")]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::ops::RangeInclusive;

use lightmotif::abc::Alphabet;
use lightmotif::dense::DenseMatrix;
use lightmotif::num::Unsigned;
use lightmotif::pwm::ScoringMatrix;

/// A rescaled matrix
#[derive(Debug)]
struct TfmpMatrix<'pssm, A: Alphabet> {
    /// The granularity with which the matrix has been built.
    granularity: f64,

    scale: f64,
    /// The matrix offsets.
    offset: i64,
    offsets: Vec<i64>,

    /// Original PSSM
    matrix: &'pssm ScoringMatrix<A>,

    /// Rescaled PSSM in integer space.
    int_matrix: DenseMatrix<i64, A::K>,

    /// The maximum error caused by integer rescale.
    error_max: f64,

    /// The maximum integer score reachable at each row of the matrix.
    max_score_rows: Vec<i64>,
    /// The minimum integer score reachable at each row of the matrix.
    min_score_rows: Vec<i64>,
}

impl<'pssm, A: Alphabet> TfmpMatrix<'pssm, A> {
    /// Initialize the TFM-Pvalue algorithm for the given matrix.
    pub fn new(granularity: f64, matrix: &'pssm ScoringMatrix<A>) -> Self {
        let M = matrix.len();
        let mut tfmp = Self {
            granularity,
            matrix,
            scale: 1.0,
            offset: 0,
            offsets: vec![0; M],
            int_matrix: DenseMatrix::new(M),
            max_score_rows: vec![0; M],
            min_score_rows: vec![0; M],
            error_max: 0.0,
        };
        tfmp.compute_int_matrix();
        tfmp
    }

    /// Compute the integer matrix for the requested granularity.
    fn compute_int_matrix(&mut self) {
        // matrix dimensions
        let M: usize = self.matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // compute matrix score range
        let min_s = self.matrix.min_score() as f64;
        let max_s = self.matrix.max_score() as f64;
        let score_range = max_s - min_s + 1.0;

        // compute effective granularity
        self.scale = if self.granularity > 1.0 {
            self.granularity / score_range
        } else if self.granularity < 1.0 {
            1.0 / self.granularity
        } else {
            1.0
        };

        // compute integer matrix
        let weights = self.matrix.weights();
        for i in 0..M {
            for j in 0..K - 1 {
                self.int_matrix[i][j] = (weights[i][j] as f64 * self.scale).floor() as i64;
            }
        }

        // compute maximum error by summing max error at each row
        self.error_max = 0.0;
        for i in 1..M {
            let mut max_e = weights[i][0] as f64 * self.scale - self.int_matrix[i][0] as f64;
            for j in 0..K - 1 {
                if max_e < weights[i][j] as f64 * self.scale - self.int_matrix[i][j] as f64 {
                    max_e = weights[i][j] as f64 * self.scale - self.int_matrix[i][j] as f64;
                }
            }
            self.error_max += max_e;
        }

        // TODO: sort columns?

        // FIXME: overflows here
        // compute offsets
        self.offset = 0;
        self.offsets = vec![0; M];
        for i in 0..M {
            self.offsets[i] = -(0..K - 1).map(|j| self.int_matrix[i][j]).min().unwrap();
            for j in 0..K - 1 {
                self.int_matrix[i][j] += self.offsets[i];
            }
            self.offset += self.offsets[i];
        }

        // look for the minimum score of the matrix for each row
        self.min_score_rows = vec![0; M];
        self.max_score_rows = vec![0; M];
        // let mut int_min_s = 0;
        // let mut int_max_s = 0;
        for i in 0..M {
            self.min_score_rows[i] = (0..K - 1).map(|j| self.int_matrix[i][j]).min().unwrap();
            self.max_score_rows[i] = (0..K - 1).map(|j| self.int_matrix[i][j]).max().unwrap();
            // int_min_s += min_score_rows[i];
            // int_max_s += self.max_score_rows[i];
        }
        // let mut int_score_range = int_max_s - int_min_s + 1;

        // backtrack to find the best/worst score at each row
        // let mut best_score = vec![0; M];
        // let mut worst_score = vec![0; M];
        // best_score[M-1] = int_max_s;
        // worst_score[M-1] = int_min_s;
        // for i in (0..M-1).rev() {
        //     best_score[i] = best_score[i+1] - self.max_score_rows[i+1];
        //     worst_score[i] = worst_score[i+1] - min_score_rows[i+1];
        // }
    }

    /// Compute the score distribution between `min` and `max`.
    fn distribution(&self, min: i64, max: i64) -> Vec<HashMap<i64, f64>> {
        // matrix dimensions
        let M: usize = self.matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // background frequencies
        let bg = self.matrix.background().frequencies();

        // maps for each steps of the computation
        let mut nbocc = vec![HashMap::new(); M + 1];

        // maximum score reachable with the suffix matrix from i to M-1
        let mut maxs = vec![0; M + 1];
        for i in (0..M).rev() {
            maxs[i] = maxs[i + 1] + self.max_score_rows[i];
        }

        // initialize the map at first position with background frequencies
        for k in 0..K - 1 {
            if self.int_matrix[0][k] + maxs[1] >= min {
                *nbocc[0].entry(self.int_matrix[0][k]).or_default() += bg[k] as f64;
            }
        }

        // compute q values for scores greater or equal to min
        nbocc[M - 1].insert(max + 1, 0.0);
        for pos in 1..M {
            // split the array in two to make the borrow checker happy
            let (l, r) = nbocc.split_at_mut(pos);
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

        nbocc
    }

    /// Compute a p-value range for the given score.
    fn lookup_pvalue(&self, score: f64) -> RangeInclusive<f64> {
        // matrix dimensions
        let M: usize = self.matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // Compute the integer score range from the given score.
        let avg = (score * self.scale + self.offset as f64).floor() as i64;
        let max = (score * self.scale + self.offset as f64 + self.error_max + 1.0).floor() as i64;
        let min = (score * self.scale + self.offset as f64 - self.error_max - 1.0).floor() as i64;
        // println!("offset: {:?}", self.offset);
        // println!("lookup_pvalue({:?} {:?} {:?})", avg, max, min);

        // compute q values for the given scores
        let mut nbocc = self.distribution(min, max);

        // split to make borrow checker happy
        let (l, mut r) = nbocc.split_at_mut(M);

        // compute p-values into nbocc[M]
        let mut sum = l[0].get(&(max + 1)).cloned().unwrap_or_default();
        let mut s = max + 1;
        for &first in l[M - 1].keys() {
            sum += l[M - 1][&first];
            if first >= avg {
                s = first;
            }
            r[0].insert(first, sum);
        }

        // sort
        let mut keys = nbocc[M].keys().cloned().collect::<Vec<i64>>();
        keys.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());
        let mut kmax = keys.iter().position(|&k| k == s).unwrap();
        while kmax > 0 && keys[kmax] as f64 >= s as f64 - self.error_max {
            kmax -= 1;
        }

        let pmax = nbocc[M][&keys[kmax]];
        let pmin = nbocc[M][&s];
        RangeInclusive::new(pmax, pmin)
    }

    /// Compute a score range for a given p-value.
    fn lookup_score(&self, pvalue: f64, range: RangeInclusive<i64>) -> (i64, f64, f64) {
        // matrix dimensions
        let M: usize = self.matrix.len();
        let K: usize = <A as Alphabet>::K::USIZE;

        // compute score range for target pvalue
        let min = *range.start();
        let max = *range.end();

        println!("max_s: {}", max);
        println!("min_s: {}", min);

        // compute q values
        let mut nbocc = self.distribution(min, max);

        // compute p-values into nbocc[M]
        let mut keys = nbocc[M - 1].keys().cloned().collect::<Vec<_>>();
        keys.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());

        let mut sum = 0.0;
        let mut riter = keys.len() - 1;
        let mut alpha = keys[riter] + 1;
        let mut alpha_e = alpha;

        while riter > 0 {
            sum += nbocc[M - 1][&keys[riter]];
            nbocc[M].insert(keys[riter], sum);
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
                sum += nbocc[M].get(&alpha_e).cloned().unwrap_or_default();
            }
            nbocc[M].insert(alpha_e, sum);
        }

        if (alpha - alpha_e) as f64 > self.error_max {
            alpha_e = alpha;
        }

        // let smax = nbocc[M][&alpha];
        // let smin = nbocc[M][&alpha_e];
        // RangeInclusive::new(smin, smax)
        (alpha, nbocc[M][&alpha_e], nbocc[M][&alpha])
    }
}

fn score2pval<A: Alphabet>(matrix: &ScoringMatrix<A>, score: f64) -> f64 {
    let mut granularity = 0.1;
    let max_granularity = 1e-10;
    let dec_granularity = 0.1;

    let mut pmin = f64::NAN;
    let mut pmax = f64::NAN;
    let mut tfmp;

    while granularity > max_granularity && pmin != pmax {
        tfmp = TfmpMatrix::new(granularity, matrix);

        // println!("mat: {:?}", tfmp.matrix.weights());
        // println!("matInt: {:?}", tfmp.int_matrix);
        // println!("error_max: {:?}", tfmp.error_max);
        // println!("offset: {:?}", tfmp.offsets.iter().sum::<i64>());
        // println!("offsets: {:?}", tfmp.offsets);

        (pmin, pmax) = tfmp.lookup_pvalue(score).into_inner();
        // println!("{:?}: {:?}", granularity, (pmin, pmax));
        if pmin == pmax {
            break;
        }

        // break;

        // let avg_s = score * matrix.granularity + matrix.offset;
        // let max_s = avg_s + matrix.error_max + 1;
        // let min_s = avg_s - matrix.error_max - 1;

        // look_for_pvalue(matrix, avg_s, min_s, max_s);
        granularity *= dec_granularity;
    }

    pmax
}

fn pval2score<A: Alphabet>(matrix: &ScoringMatrix<A>, pvalue: f64) -> f64 {
    let mut granularity = 0.1;
    let max_granularity = 1e-10;
    let dec_granularity = 0.1;

    let mut score = 0;
    let mut fmin = 0.0;
    let mut fmax = 0.0;

    let mut tfmp = TfmpMatrix::new(granularity, matrix);
    let mut min = tfmp.min_score_rows.iter().sum::<i64>();
    let mut max = tfmp.max_score_rows.iter().sum::<i64>() + (tfmp.error_max + 0.5).ceil() as i64;

    while granularity > max_granularity {
        tfmp = TfmpMatrix::new(granularity, matrix);

        // println!("mat: {:?}", tfmp.matrix.weights());
        // println!("matInt: {:?}", tfmp.int_matrix);
        // println!("error_max: {:?}", tfmp.error_max);
        // println!("offset: {:?}", tfmp.offsets.iter().sum::<i64>());
        // println!("offsets: {:?}", tfmp.offsets);

        (score, fmin, fmax) = tfmp.lookup_score(pvalue, RangeInclusive::new(min, max));
        // println!("score={:?} fmin={:?} fmax={:?}", score, fmin, fmax);
        if fmin == fmax {
            break;
        }

        // let avg_s = score * matrix.granularity + matrix.offset;
        // let max_s = avg_s + matrix.error_max + 1;
        // let min_s = avg_s - matrix.error_max - 1;

        // look_for_pvalue(matrix, avg_s, min_s, max_s);
        min = ((score as f64 - (tfmp.error_max + 0.5).ceil()) / dec_granularity).floor() as i64;
        max = ((score as f64 + (tfmp.error_max + 0.5).ceil()) / dec_granularity).floor() as i64;
        granularity *= dec_granularity;
    }

    (score - tfmp.offset) as f64 / tfmp.scale
}

#[cfg(test)]
mod test {
    use lightmotif::abc::Alphabet;
    use lightmotif::abc::Background;
    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;
    use lightmotif::abc::Symbol;
    use lightmotif::dense::DenseMatrix;
    use lightmotif::num::Unsigned;
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

    #[test]
    fn test_score2pval() {
        let a = [3, 7, 9, 3, 11, 11, 11, 3, 4, 3, 8, 8, 9, 9, 11, 2];
        let c = [5, 0, 1, 6, 0, 0, 0, 3, 1, 4, 5, 1, 0, 5, 0, 7];
        let g = [4, 3, 1, 4, 3, 2, 2, 2, 8, 6, 1, 4, 2, 0, 3, 0];
        let t = [2, 4, 3, 1, 0, 1, 1, 6, 1, 1, 0, 1, 3, 0, 0, 5];

        let mut counts = DenseMatrix::<u32, <Dna as Alphabet>::K>::new(a.len());
        for i in 0..a.len() {
            counts[i][Nucleotide::A.as_index()] = a[i];
            counts[i][Nucleotide::C.as_index()] = c[i];
            counts[i][Nucleotide::G.as_index()] = g[i];
            counts[i][Nucleotide::T.as_index()] = t[i];
        }

        let bg = Background::<Dna>::uniform();
        let mut log_odds = DenseMatrix::new(a.len());
        for i in 0..counts.rows() {
            let sum = counts[i].iter().sum::<u32>() as f32;
            for j in 0..counts.columns() {
                log_odds[i][j] = (((counts[i][j] as f32 + 0.25) / (sum + 1.0)).log2()
                    - bg.frequencies()[j].log2());
            }
        }

        let pssm = ScoringMatrix::new(bg, log_odds);
        assert_almost_eq!(score2pval(&pssm, 8.882756), 0.0003, places = 5);
        assert_almost_eq!(score2pval(&pssm, 12.657785), 0.00001, places = 5);
        assert_almost_eq!(score2pval(&pssm, 19.1), 1e-10, places = 4);
    }

    #[test]
    fn test_pval2score() {
        let a = [3, 7, 9, 3, 11, 11, 11, 3, 4, 3, 8, 8, 9, 9, 11, 2];
        let c = [5, 0, 1, 6, 0, 0, 0, 3, 1, 4, 5, 1, 0, 5, 0, 7];
        let g = [4, 3, 1, 4, 3, 2, 2, 2, 8, 6, 1, 4, 2, 0, 3, 0];
        let t = [2, 4, 3, 1, 0, 1, 1, 6, 1, 1, 0, 1, 3, 0, 0, 5];

        let mut counts = DenseMatrix::<u32, <Dna as Alphabet>::K>::new(a.len());
        for i in 0..a.len() {
            counts[i][Nucleotide::A.as_index()] = a[i];
            counts[i][Nucleotide::C.as_index()] = c[i];
            counts[i][Nucleotide::G.as_index()] = g[i];
            counts[i][Nucleotide::T.as_index()] = t[i];
        }

        let bg = Background::<Dna>::uniform();
        let mut log_odds = DenseMatrix::new(a.len());
        for i in 0..counts.rows() {
            let sum = counts[i].iter().sum::<u32>() as f32;
            for j in 0..counts.columns() {
                log_odds[i][j] = (((counts[i][j] as f32 + 0.25) / (sum + 1.0)).log2()
                    - bg.frequencies()[j].log2());
            }
        }

        let pssm = ScoringMatrix::new(bg, log_odds);
        assert_almost_eq!(pval2score(&pssm, 0.00001), 12.657785, places = 4);
        assert_almost_eq!(pval2score(&pssm, 0.0003), 8.882756, places = 4);
        assert_almost_eq!(pval2score(&pssm, 1e-10), 19.1, places = 4);
    }
}
