#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

use self::seal::Vector;
use super::abc::Alphabet;
use super::abc::DnaAlphabet;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::pwm::WeightMatrix;
use super::seq::EncodedSequence;
use super::seq::StripedSequence;

mod seal {
    pub trait Vector {}

    impl Vector for f32 {}

    #[cfg(target_feature = "avx2")]
    impl Vector for std::arch::x86_64::__m256 {}
}

pub struct Pipeline<A: Alphabet, V: Vector> {
    alphabet: A,
    vector: std::marker::PhantomData<V>,
}

impl<A: Alphabet, V: Vector> Pipeline<A, V> {
    pub fn new() -> Self {
        Self {
            alphabet: A::default(),
            vector: std::marker::PhantomData,
        }
    }
}

impl Pipeline<DnaAlphabet, f32> {
    pub fn score<const C: usize>(
        &self,
        seq: &StripedSequence<DnaAlphabet, C>,
        pwm: &WeightMatrix<DnaAlphabet, { DnaAlphabet::K }>,
    ) -> DenseMatrix<f32, C> {
        let mut result = DenseMatrix::<f32, C>::new(seq.data.rows());
        for i in 0..seq.length - pwm.data.rows() + 1 {
            let mut score = 0.0;
            for j in 0..pwm.data.rows() {
                let offset = i + j;
                let col = offset / seq.data.rows();
                let row = offset % seq.data.rows();
                score += pwm.data[j][seq.data[row][col].as_index()];
            }
            let col = i / result.rows();
            let row = i % result.rows();
            result[row][col] = score;
        }
        result
    }
}

#[cfg(target_feature = "avx2")]
impl Pipeline<DnaAlphabet, __m256> {
    pub fn score(
        &self,
        seq: &StripedSequence<DnaAlphabet, 32>,
        pwm: &WeightMatrix<DnaAlphabet, { DnaAlphabet::K }>,
    ) -> DenseMatrix<f32, 32> {
        let mut result = DenseMatrix::new(seq.data.rows());
        unsafe {
            // mask vectors for broadcasting:
            let m1: __m256i = _mm256_set_epi32(
                0xFFFFFF03u32 as i32,
                0xFFFFFF02u32 as i32,
                0xFFFFFF01u32 as i32,
                0xFFFFFF00u32 as i32,
                0xFFFFFF03u32 as i32,
                0xFFFFFF02u32 as i32,
                0xFFFFFF01u32 as i32,
                0xFFFFFF00u32 as i32,
            );
            let m2: __m256i = _mm256_set_epi32(
                0xFFFFFF07u32 as i32,
                0xFFFFFF06u32 as i32,
                0xFFFFFF05u32 as i32,
                0xFFFFFF04u32 as i32,
                0xFFFFFF07u32 as i32,
                0xFFFFFF06u32 as i32,
                0xFFFFFF05u32 as i32,
                0xFFFFFF04u32 as i32,
            );
            let m3: __m256i = _mm256_set_epi32(
                0xFFFFFF0Bu32 as i32,
                0xFFFFFF0Au32 as i32,
                0xFFFFFF09u32 as i32,
                0xFFFFFF08u32 as i32,
                0xFFFFFF0Bu32 as i32,
                0xFFFFFF0Au32 as i32,
                0xFFFFFF09u32 as i32,
                0xFFFFFF08u32 as i32,
            );
            let m4: __m256i = _mm256_set_epi32(
                0xFFFFFF0Fu32 as i32,
                0xFFFFFF0Eu32 as i32,
                0xFFFFFF0Du32 as i32,
                0xFFFFFF0Cu32 as i32,
                0xFFFFFF0Fu32 as i32,
                0xFFFFFF0Eu32 as i32,
                0xFFFFFF0Du32 as i32,
                0xFFFFFF0Cu32 as i32,
            );
            // loop over every row of the sequence data
            for i in 0..seq.data.rows() - pwm.data.rows() + 1 {
                let mut s1 = _mm256_setzero_ps();
                let mut s2 = _mm256_setzero_ps();
                let mut s3 = _mm256_setzero_ps();
                let mut s4 = _mm256_setzero_ps();

                // for j in 0..pwm.data.rows() {
                //     // load table
                //     let row = pwm.data[j].as_ptr();
                //     let c = _mm_load_ps(row);
                //     let t = _mm256_set_m128(c, c);
                //     // load text
                //     let x = _mm256_loadu_si256(seq.data[i+j].as_ptr() as *const __m256i);
                //     // compute probabilities using an external lookup table
                //     let p1 = _mm256_permutevar_ps(t, _mm256_shuffle_epi8(x, m1));
                //     let p2 = _mm256_permutevar_ps(t, _mm256_shuffle_epi8(x, m2));
                //     let p3 = _mm256_permutevar_ps(t, _mm256_shuffle_epi8(x, m3));
                //     let p4 = _mm256_permutevar_ps(t, _mm256_shuffle_epi8(x, m4));
                //     // add log odds
                //     s1 = _mm256_add_ps(s1, p1);
                //     s2 = _mm256_add_ps(s2, p2);
                //     s3 = _mm256_add_ps(s3, p3);
                //     s4 = _mm256_add_ps(s4, p4);
                // }

                for j in 0..pwm.data.rows() {
                    let x = _mm256_loadu_si256(seq.data[i + j].as_ptr() as *const __m256i);
                    let row = pwm.data[j].as_ptr();
                    // compute probabilities using an external lookup table
                    let p1 = _mm256_i32gather_ps(
                        row,
                        _mm256_shuffle_epi8(x, m1),
                        std::mem::size_of::<f32>() as _,
                    );
                    let p2 = _mm256_i32gather_ps(
                        row,
                        _mm256_shuffle_epi8(x, m2),
                        std::mem::size_of::<f32>() as _,
                    );
                    let p3 = _mm256_i32gather_ps(
                        row,
                        _mm256_shuffle_epi8(x, m3),
                        std::mem::size_of::<f32>() as _,
                    );
                    let p4 = _mm256_i32gather_ps(
                        row,
                        _mm256_shuffle_epi8(x, m4),
                        std::mem::size_of::<f32>() as _,
                    );
                    // add log odds
                    s1 = _mm256_add_ps(s1, p1);
                    s2 = _mm256_add_ps(s2, p2);
                    s3 = _mm256_add_ps(s3, p3);
                    s4 = _mm256_add_ps(s4, p4);
                }

                let row: &mut [f32] = &mut result[i];
                _mm256_storeu_ps(&mut row[0x0], s1);
                _mm256_storeu_ps(&mut row[0x4], s1);
                _mm256_storeu_ps(&mut row[0x8], s1);
                _mm256_storeu_ps(&mut row[0xc], s1);
            }
        }
        result
    }
}
