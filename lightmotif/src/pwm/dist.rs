//! Discretized score distributions for *p-value* approximation.
//!
//! This module implements the discretized computation of the scores
//! distribution for a [`ScoringMatrix`] as implemented in MEME[\[1\]](#ref1).
//! The PSSM is first discretized with a position-independent rescaling into
//! a limited range (N=1000 by default). A density is then generated using the
//! discrete scores, and then used to generate the cumulative distribution.
//!
//! # Example
//! ```
//! # use lightmotif::*;
//! # let counts = CountMatrix::<Dna>::from_sequences(
//! #  ["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"]
//! #        .into_iter()
//! #        .map(|s| EncodedSequence::encode(s).unwrap()),
//! # )
//! # .unwrap();
//! # let pssm = counts.to_freq(0.1).to_scoring(None);
//! // Create a `ScoreDistribution` from a `ScoringMatrix`
//! let dist = pssm.to_score_distribution();
//!
//! // Get the score corresponding to a *p-value* of 1e-5
//! let score = dist.pvalue(1e-5);
//!
//! // Get the *p-value* for a score of 13.1
//! let pvalue = dist.score(13.1);
//! ```
//!
//! ## 📚 References
//! - <a id="ref1">\[1\]</a> Grant, Charles E., Timothy L. Bailey, and William Stafford Noble. ‘FIMO: Scanning for Occurrences of a given Motif’. Bioinformatics 27, no. 7 (1 April 2011): 1017–18. [doi:10.1093/bioinformatics/btr064](https://doi.org/10.1093/bioinformatics/btr064).

#[cfg(feature = "sampling")]
use rand::distributions::uniform::Uniform;
#[cfg(feature = "sampling")]
use rand::distributions::Distribution;

use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::num::Unsigned;

use super::ScoringMatrix;

// --- ScoreDistribution -------------------------------------------------------

/// The default range for CDF approximation used in MEME.
const CDF_RANGE: usize = 1000;

/// An approximate distribution of the scores of a [`ScoringMatrix`].
#[derive(Debug, Clone)]
pub struct ScoreDistribution<A: Alphabet> {
    /// The column-wise scale for the matrix
    scale: f64,
    /// The column-wise offsets for the matrix.
    offset: i32,
    /// The score range.
    #[allow(unused)]
    range: usize,
    /// The discrete score matrix.
    data: DenseMatrix<i32, A::K>,
    /// A sampled distribution of the survival function.
    sf: Vec<f64>,
    /// The minimum score that can reached by the discrete matrix.
    min_score: i32,
    /// The maximum score that can reached by the discrete matrix.
    max_score: i32,
}

impl<A: Alphabet> ScoreDistribution<A> {
    /// Get the distribution of the survival function.
    pub fn sf(&self) -> &[f64] {
        self.sf.as_slice()
    }

    /// Scale the given score to an integer score using the matrix scale.
    #[inline]
    pub fn scale(&self, score: f32) -> i32 {
        let w = self.data.rows() as i32;
        f64::round((score as f64 - (w * self.offset) as f64) * self.scale) as i32
    }

    /// Unscale the given integer score into a score using the matrix scale.
    #[inline]
    pub fn unscale(&self, score: i32) -> f32 {
        let w = self.data.rows() as i32;
        (score as f32) / (self.scale as f32) + (w * self.offset) as f32
    }

    /// Get the *p-value* for the given score.
    #[inline]
    pub fn pvalue(&self, score: f32) -> f64 {
        let scaled = self.scale(score);
        if scaled < self.min_score {
            1.0
        } else if scaled as usize >= self.sf.len() {
            0.0
        } else {
            self.sf[scaled as usize]
        }
    }

    /// Get the score for a given *p-value*.
    #[inline]
    pub fn score(&self, pvalue: f64) -> f32 {
        if pvalue >= 1.0 {
            self.unscale(self.min_score)
        } else if pvalue <= 0.0 {
            self.unscale(self.max_score)
        } else {
            match self.sf.binary_search_by(|x| pvalue.partial_cmp(x).unwrap()) {
                Ok(x) => self.unscale(x as i32),
                Err(x) => self.unscale(x as i32),
            }
        }
    }

    /// Get the minimum p-value attainable by the PSSM.
    ///
    /// The minimum p-value is the p-value of the largest score. It has a lower
    /// bound set by the distribution of PSSM scores, which depends on the
    /// width of the scoring matrix `w` and the number of symbols in the
    /// alphabet `K`, such that `P >= K^-w`.
    #[inline]
    pub fn min_pvalue(&self) -> f64 {
        self.sf[self.max_score as usize]
    }
}

impl<A: Alphabet, S: AsRef<ScoringMatrix<A>>> From<S> for ScoreDistribution<A> {
    fn from(pssm: S) -> Self {
        let pssm = pssm.as_ref();
        // scale pssm and set the scale/offset (see scale_score_matrix)
        let mut small = *pssm
            .matrix()
            .iter()
            .flatten()
            .filter(|x| !x.is_infinite())
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap() as f64;
        let large = *pssm
            .matrix()
            .iter()
            .flatten()
            .filter(|x| !x.is_infinite())
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap() as f64;
        if small == large {
            small = large - 1.0;
        }

        // compute offset and scale
        let offset = small.floor();
        let scale = ((CDF_RANGE as f64) / (large - offset)).floor();

        // compute discretized matrix
        let mut data = DenseMatrix::<i32, A::K>::new(pssm.matrix().rows());
        for (src_row, dst_row) in pssm.matrix().iter().zip(data.iter_mut()) {
            for i in 0..A::K::USIZE {
                dst_row[i] = f64::round((src_row[i] as f64 - offset) * scale) as i32;
            }
        }

        // compute pdf
        let pdf = {
            let range = CDF_RANGE;
            let size = data.rows() * range + 1;
            let mut pdf_old = vec![0.0; size];
            let mut pdf_new = vec![0.0; size];
            pdf_new[0] = 1.0;

            for (i, row) in data.iter().enumerate() {
                let max = i * range;

                std::mem::swap(&mut pdf_old, &mut pdf_new);
                pdf_new[..=max + range].fill(0.0);

                for a in A::symbols().iter() {
                    let s = row[a.as_index()];
                    if s != i32::MIN {
                        for k in 0..=max {
                            let old = pdf_old[k];
                            if old != 0.0 {
                                pdf_new[k + s as usize] += old * pssm.background[*a] as f64;
                            }
                        }
                    }
                }
            }

            pdf_new
        };

        // compute survival function
        let mut min_score = 0;
        let mut max_score = 0;
        let sf = {
            let mut sf = pdf;
            for i in (0..=sf.len() - 2).rev() {
                let p_iplus1 = sf[i + 1];
                let p_i = sf[i];
                let p = p_i + p_iplus1;

                sf[i] = p.min(1.0);

                if max_score == 0 && p_iplus1 > 0.0 {
                    max_score = i as i32 + 1;
                }
                if p_i > 0.0 {
                    min_score = i as i32;
                }
            }
            sf
        };

        Self {
            scale,
            offset: offset as i32,
            range: CDF_RANGE,
            data,
            sf,
            min_score,
            max_score,
        }
    }
}

#[cfg(feature = "sampling")]
impl<A: Alphabet> Distribution<f32> for ScoreDistribution<A> {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        let s = Uniform::new_inclusive(0.0, 1.0);
        let p = s.sample(rng);
        self.score(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abc::Dna;
    use crate::pwm::CountMatrix;

    macro_rules! assert_almost_eq {
        ($x:expr, $y:expr, places = $places:expr) => {{
            assert_eq!(
                ($x * 10.0_f64.powi($places)).round(),
                ($y * 10.0_f64.powi($places)).round(),
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
    fn pvalue() {
        let pssm = build_ma0045();
        let cdf = ScoreDistribution::from(&pssm);

        assert_almost_eq!(cdf.pvalue(8.89385), 0.0003, places = 5);
        assert_almost_eq!(cdf.pvalue(12.66480), 0.00001, places = 5);
        assert_almost_eq!(cdf.pvalue(17.71508), 1e-9, places = 9);
    }

    #[test]
    fn score() {
        let pssm = build_ma0045();
        let cdf = ScoreDistribution::from(&pssm);

        println!("max_score={:?}", pssm.max_score());

        assert_almost_eq!(cdf.score(0.00001) as f64, 12.66480, places = 5);
        assert_almost_eq!(cdf.score(0.0003) as f64, 8.89385, places = 5);
        assert_almost_eq!(cdf.score(1e-9) as f64, 17.71508, places = 4);
    }
}
