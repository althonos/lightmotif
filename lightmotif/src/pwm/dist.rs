//! Discretized score distributions for *p-value* approximation.
//!
//! This module implements the discretized computation of the scores
//! distribution for a [`ScoringMatrix`] as implemented in MEME[\[1\]](#ref1).
//! The PSSM is first discretized with a position-independent rescaling into
//! a limited range (N=1000 by default). A density is then generated using the
//! discrete scores, and then used to generate the cumulative distribution.
//!
//! ## 📚 References
//! - <a id="ref1">\[1\]</a> Grant, Charles E., Timothy L. Bailey, and William Stafford Noble. ‘FIMO: Scanning for Occurrences of a given Motif’. Bioinformatics 27, no. 7 (1 April 2011): 1017–18. [doi:10.1093/bioinformatics/btr064](https://doi.org/10.1093/bioinformatics/btr064).

use crate::abc::Alphabet;
use crate::abc::Background;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::num::Unsigned;

use super::ScoringMatrix;

// --- ScoreDistribution -------------------------------------------------------

/// The default range for CDF approximation used in MEME.
const CDF_RANGE: usize = 1000;

/// An approximate cumulative distribution for the scores of a [`ScoringMatrix`].
pub struct ScoreDistribution<A: Alphabet> {
    background: Background<A>,
    scale: f64,
    offset: i32,
    range: usize,
    data: DenseMatrix<i32, A::K>,
    sf: Vec<f64>,
    min_score: i32,
    max_score: i32,
}

impl<A: Alphabet> ScoreDistribution<A> {
    /// Scale the given score to an integer score using the matrix scale.
    pub fn scale(&self, score: f32) -> i32 {
        let w = self.data.rows() as i32;
        f64::round((score as f64 - (w * self.offset) as f64) * self.scale) as i32
    }

    /// Unscale the given integer score into a score using the matrix scale.
    pub fn unscale(&self, score: i32) -> f32 {
        let w = self.data.rows() as i32;
        (score as f32) / (self.scale as f32) + (w * self.offset) as f32
    }

    /// Get the *p-value* for the given score.
    pub fn pvalue(&self, score: f32) -> f64 {
        let scaled = self.scale(score);
        if scaled <= self.min_score {
            1.0
        } else if scaled >= self.max_score {
            0.0
        } else {
            self.cdf[scaled as usize]
        }
    }

    /// Get the score for a given *p-value*.
    pub fn score(&self, pvalue: f64) -> f32 {
        if pvalue >= 1.0 {
            self.unscale(self.min_score)
        } else if pvalue <= 0.0 {
            self.unscale(self.max_score)
        } else {
            match self
                .cdf
                .binary_search_by(|x| pvalue.partial_cmp(x).unwrap())
            {
                Ok(x) => self.unscale(x as i32),
                Err(x) => self.unscale(x as i32),
            }
        }
    }
}

impl<A: Alphabet> From<&'_ ScoringMatrix<A>> for ScoreDistribution<A> {
    fn from(pssm: &'_ ScoringMatrix<A>) -> Self {
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
                dst_row[i] = f64::round((src_row[i] as f64 - offset as f64) * scale) as i32;
            }
        }

        // compute pdf
        let pdf = {
            let range = CDF_RANGE;
            let size = data.rows() * range + 1;
            let mut pdf_old = vec![0.0; size];
            let mut pdf_new = vec![0.0; size];
            pdf_new[0] = 1.0;

            for i in 0..data.rows() {
                let max = i * range;

                std::mem::swap(&mut pdf_old, &mut pdf_new);
                for k in 0..=max + range {
                    pdf_new[k] = 0.0;
                }

                for a in A::symbols().iter() {
                    let s = data[crate::dense::MatrixCoordinates::new(i, a.as_index())];
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
            background: pssm.background.clone(),
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
