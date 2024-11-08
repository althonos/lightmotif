use crate::abc::Alphabet;
use crate::abc::Background;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::num::Unsigned;

use super::ScoringMatrix;

// --- CumulativeDistribution --------------------------------------------------

/// The default range for CDF approximation used in MEME.
const CDF_RANGE: usize = 1000;

/// An approximate cumulative distribution for the scores of a [`ScoringMatrix`].
pub struct CumulativeDistribution<A: Alphabet> {
    background: Background<A>,
    scale: f64,
    offset: i32,
    range: usize,
    data: DenseMatrix<i32, A::K>,
    cdf: Vec<f64>,

    min_score: i32,
    max_score: i32,
}

impl<A: Alphabet> CumulativeDistribution<A> {
    /// Rescale a float score to discretized units.
    fn scale(&self, score: f32) -> i32 {
        let w = self.data.rows() as i32;
        f64::round((score as f64 - (w * self.offset) as f64) * self.scale) as i32
    }

    /// Unscale a discrete score to float.
    fn unscale(&self, score: i32) -> f32 {
        println!("unscale(score={:?})", score);
        let w = self.data.rows() as i32;
        (score as f32) / (self.scale as f32) + (w * self.offset) as f32
    }

    /// Get the *p-value* for the given score.
    pub fn pvalue(&self, score: f32) -> f64 {
        let scaled = self.scale(score);
        if scaled < 0 {
            1.0
        } else if scaled as usize >= self.cdf.len() {
            0.0
        } else {
            self.cdf[scaled as usize]
        }
    }

    /// Get the score for a given *p-value*.
    pub fn score(&self, pvalue: f64) -> f32 {
        match self
            .cdf
            .binary_search_by(|x| pvalue.partial_cmp(x).unwrap())
        {
            Ok(x) => self.unscale(x as i32),
            Err(x) => self.unscale(x as i32),
        }
    }
}

impl<A: Alphabet> From<&'_ ScoringMatrix<A>> for CumulativeDistribution<A> {
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

        // let mut matrix = Self {
        //     background: pssm.background.clone(),
        //     scale,
        //     offset: offset as i32,
        //     data: DenseMatrix::new(0),
        //     range: CDF_RANGE,
        // };

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

        // compute cdf
        let mut min_score = 0;
        let mut max_score = 0;
        let cdf = {
            let mut cdf = pdf;
            for i in (0..=cdf.len() - 2).rev() {
                let p_iplus1 = cdf[i + 1];
                let p_i = cdf[i];
                let p = p_i + p_iplus1;

                cdf[i] = p.min(1.0);

                if max_score == 0 && p_iplus1 > 0.0 {
                    max_score = i as i32 + 1;
                }
                if p_i > 0.0 {
                    min_score = i as i32;
                }
            }
            cdf
        };

        Self {
            background: pssm.background.clone(),
            scale,
            offset: offset as i32,
            range: CDF_RANGE,
            data,
            cdf,
            min_score,
            max_score,
        }
    }
}
