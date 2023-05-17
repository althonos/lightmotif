//! Intel 128-bit vector implementation, for 16 elements column width.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Div;
use std::ops::Rem;

use typenum::consts::U16;
use typenum::marker_traits::Unsigned;
use typenum::marker_traits::Zero;

use super::Backend;
use crate::abc::Alphabet;
use crate::num::StrictlyPositive;
use crate::pli::scores::StripedScores;
use crate::pwm::ScoringMatrix;
use crate::seq::StripedSequence;

/// A marker type for the SSE2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Sse2;

impl Backend for Sse2 {
    type LANES = U16;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn score_sse2<A, C>(
    seq: &StripedSequence<A, C>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<C>,
) where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    // mask vectors for broadcasting uint8x16_t to uint32x4_t to floatx4_t
    let zero = _mm_setzero_si128();
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..<C as Div<U16>>::Output::USIZE).map(|i| i * <Sse2 as Backend>::LANES::USIZE) {
        // process every position of the sequence data
        for i in 0..seq.data.rows() - seq.wrap {
            // reset sums for current position
            let mut s1 = _mm_setzero_ps();
            let mut s2 = _mm_setzero_ps();
            let mut s3 = _mm_setzero_ps();
            let mut s4 = _mm_setzero_ps();
            // advance position in the position weight matrix
            for j in 0..pssm.len() {
                // load sequence row and broadcast to f32
                let x = _mm_load_si128(seq.data[i + j].as_ptr().add(offset) as *const __m128i);
                let hi = _mm_unpackhi_epi8(x, zero);
                let lo = _mm_unpacklo_epi8(x, zero);
                let x1 = _mm_unpacklo_epi8(lo, zero);
                let x2 = _mm_unpackhi_epi8(lo, zero);
                let x3 = _mm_unpacklo_epi8(hi, zero);
                let x4 = _mm_unpackhi_epi8(hi, zero);
                // load row for current weight matrix position
                let row = pssm.weights()[j].as_ptr();
                // index lookup table with each bases incrementally
                for k in 0..A::K::USIZE {
                    let sym = _mm_set1_epi32(k as i32);
                    let lut = _mm_load1_ps(row.add(k));
                    let p1 = _mm_castsi128_ps(_mm_cmpeq_epi32(x1, sym));
                    let p2 = _mm_castsi128_ps(_mm_cmpeq_epi32(x2, sym));
                    let p3 = _mm_castsi128_ps(_mm_cmpeq_epi32(x3, sym));
                    let p4 = _mm_castsi128_ps(_mm_cmpeq_epi32(x4, sym));
                    s1 = _mm_add_ps(s1, _mm_and_ps(lut, p1));
                    s2 = _mm_add_ps(s2, _mm_and_ps(lut, p2));
                    s3 = _mm_add_ps(s3, _mm_and_ps(lut, p3));
                    s4 = _mm_add_ps(s4, _mm_and_ps(lut, p4));
                }
            }
            // record the score for the current position
            let row = &mut data[i];
            _mm_storeu_ps(row[offset..].as_mut_ptr(), s1);
            _mm_storeu_ps(row[offset + 4..].as_mut_ptr(), s2);
            _mm_storeu_ps(row[offset + 8..].as_mut_ptr(), s3);
            _mm_storeu_ps(row[offset + 12..].as_mut_ptr(), s4);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn best_position_sse2(scores: &StripedScores<<Sse2 as Backend>::LANES>) -> Option<usize> {
    if scores.len() == 0 {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm_setzero_ps();
            let mut p2 = _mm_setzero_ps();
            let mut p3 = _mm_setzero_ps();
            let mut p4 = _mm_setzero_ps();
            // store the best scores for each column
            let mut s1 = _mm_load_ps(data[0][0x00..].as_ptr());
            let mut s2 = _mm_load_ps(data[0][0x04..].as_ptr());
            let mut s3 = _mm_load_ps(data[0][0x08..].as_ptr());
            let mut s4 = _mm_load_ps(data[0][0x0c..].as_ptr());
            // process all rows iteratively
            for (i, row) in data.iter().enumerate() {
                // record the current row index
                let index = _mm_castsi128_ps(_mm_set1_epi32(i as i32));
                // load scores for the current row
                let r1 = _mm_load_ps(row[0x00..].as_ptr());
                let r2 = _mm_load_ps(row[0x04..].as_ptr());
                let r3 = _mm_load_ps(row[0x08..].as_ptr());
                let r4 = _mm_load_ps(row[0x0c..].as_ptr());
                // compare scores to local maxima
                let c1 = _mm_cmplt_ps(s1, r1);
                let c2 = _mm_cmplt_ps(s2, r2);
                let c3 = _mm_cmplt_ps(s3, r3);
                let c4 = _mm_cmplt_ps(s4, r4);
                // NOTE: code below could use `_mm_blendv_ps` instead,
                //       but this instruction is only available on SSE4.1
                //       while the rest of the code is actually using SSE2
                //       instructions only.
                // replace indices of new local maxima
                p1 = _mm_or_ps(_mm_andnot_ps(c1, p1), _mm_and_ps(index, c1));
                p2 = _mm_or_ps(_mm_andnot_ps(c2, p2), _mm_and_ps(index, c2));
                p3 = _mm_or_ps(_mm_andnot_ps(c3, p3), _mm_and_ps(index, c3));
                p4 = _mm_or_ps(_mm_andnot_ps(c4, p4), _mm_and_ps(index, c4));
                // replace values of new local maxima
                s1 = _mm_or_ps(_mm_andnot_ps(c1, s1), _mm_and_ps(r1, c1));
                s2 = _mm_or_ps(_mm_andnot_ps(c2, s2), _mm_and_ps(r2, c2));
                s3 = _mm_or_ps(_mm_andnot_ps(c3, s3), _mm_and_ps(r3, c3));
                s4 = _mm_or_ps(_mm_andnot_ps(c4, s4), _mm_and_ps(r4, c4));
            }
            // find the global maximum across all columns
            let mut x: [u32; 16] = [0; 16];
            _mm_storeu_si128(x[0x00..].as_mut_ptr() as *mut _, _mm_castps_si128(p1));
            _mm_storeu_si128(x[0x04..].as_mut_ptr() as *mut _, _mm_castps_si128(p2));
            _mm_storeu_si128(x[0x08..].as_mut_ptr() as *mut _, _mm_castps_si128(p3));
            _mm_storeu_si128(x[0x0c..].as_mut_ptr() as *mut _, _mm_castps_si128(p4));
            let mut best_pos = 0;
            let mut best_score = -f32::INFINITY;
            for (col, &row) in x.iter().enumerate() {
                if data[row as usize][col] > best_score {
                    best_score = data[row as usize][col];
                    best_pos = col * data.rows() + row as usize;
                }
            }
            Some(best_pos)
        }
    }
}

impl Sse2 {
    #[allow(unused)]
    pub fn score_into<A, C, S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        A: Alphabet,
        C: StrictlyPositive + Rem<U16> + Div<U16>,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
        <C as Rem<U16>>::Output: Zero,
        <C as Div<U16>>::Output: Unsigned,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap < pssm.len() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.len()
            );
        }

        scores.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_sse2(seq, pssm, scores);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn best_position(scores: &StripedScores<<Sse2 as Backend>::LANES>) -> Option<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            best_position_sse2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }
}
