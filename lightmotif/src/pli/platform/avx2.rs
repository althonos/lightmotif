//! Intel 256-bit vector implementation, for 32 elements column width.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use typenum::consts::U32;

use super::Backend;
use crate::abc::Dna;
use crate::abc::Nucleotide;
use crate::abc::Symbol;
use crate::pli::scores::StripedScores;
use crate::pwm::ScoringMatrix;
use crate::seq::StripedSequence;

/// A marker type for the AVX2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Avx2;

impl Backend for Avx2 {
    type LANES = U32;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(overflowing_literals)]
unsafe fn score_avx2(
    seq: &StripedSequence<Dna, <Avx2 as Backend>::LANES>,
    pssm: &ScoringMatrix<Dna>,
    scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
) {
    let data = scores.matrix_mut();
    // constant vector for comparing unknown bases
    let n = _mm256_set1_epi8(Nucleotide::N as i8);
    // mask vectors for broadcasting uint8x32_t to uint32x8_t to floatx8_t
    #[rustfmt::skip]
    let m1 = _mm256_set_epi32(
        0xFFFFFF03, 0xFFFFFF02, 0xFFFFFF01, 0xFFFFFF00,
        0xFFFFFF03, 0xFFFFFF02, 0xFFFFFF01, 0xFFFFFF00,
    );
    #[rustfmt::skip]
    let m2 = _mm256_set_epi32(
        0xFFFFFF07, 0xFFFFFF06, 0xFFFFFF05, 0xFFFFFF04,
        0xFFFFFF07, 0xFFFFFF06, 0xFFFFFF05, 0xFFFFFF04,
    );
    #[rustfmt::skip]
    let m3 = _mm256_set_epi32(
        0xFFFFFF0B, 0xFFFFFF0A, 0xFFFFFF09, 0xFFFFFF08,
        0xFFFFFF0B, 0xFFFFFF0A, 0xFFFFFF09, 0xFFFFFF08,
    );
    #[rustfmt::skip]
    let m4 = _mm256_set_epi32(
        0xFFFFFF0F, 0xFFFFFF0E, 0xFFFFFF0D, 0xFFFFFF0C,
        0xFFFFFF0F, 0xFFFFFF0E, 0xFFFFFF0D, 0xFFFFFF0C,
    );
    // process every position of the sequence data
    for i in 0..seq.data.rows() - seq.wrap {
        // reset sums for current position
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        // advance position in the position weight matrix
        for j in 0..pssm.len() {
            // load sequence row and broadcast to f32
            let x = _mm256_load_si256(seq.data[i + j].as_ptr() as *const __m256i);
            let x1 = _mm256_shuffle_epi8(x, m1);
            let x2 = _mm256_shuffle_epi8(x, m2);
            let x3 = _mm256_shuffle_epi8(x, m3);
            let x4 = _mm256_shuffle_epi8(x, m4);
            // load row for current weight matrix position
            let row = pssm.weights()[j].as_ptr();
            let t = _mm256_broadcast_ps(&*(row as *const __m128));
            let u = _mm256_broadcast_ss(&*(row.add(crate::abc::Nucleotide::N.as_index())));
            // check which bases from the sequence are unknown
            let mask = _mm256_cmpeq_epi8(x, n);
            let unk1 = _mm256_castsi256_ps(_mm256_shuffle_epi8(mask, m1));
            let unk2 = _mm256_castsi256_ps(_mm256_shuffle_epi8(mask, m2));
            let unk3 = _mm256_castsi256_ps(_mm256_shuffle_epi8(mask, m3));
            let unk4 = _mm256_castsi256_ps(_mm256_shuffle_epi8(mask, m4));
            // index A/T/G/C lookup table with the bases
            let p1 = _mm256_permutevar_ps(t, x1);
            let p2 = _mm256_permutevar_ps(t, x2);
            let p3 = _mm256_permutevar_ps(t, x3);
            let p4 = _mm256_permutevar_ps(t, x4);
            // blend together known and unknown scores
            let b1 = _mm256_blendv_ps(p1, u, unk1);
            let b2 = _mm256_blendv_ps(p2, u, unk2);
            let b3 = _mm256_blendv_ps(p3, u, unk3);
            let b4 = _mm256_blendv_ps(p4, u, unk4);
            // add log odds to the running sum
            s1 = _mm256_add_ps(s1, b1);
            s2 = _mm256_add_ps(s2, b2);
            s3 = _mm256_add_ps(s3, b3);
            s4 = _mm256_add_ps(s4, b4);
        }
        // permute lanes so that scores are in the right order
        let r1 = _mm256_permute2f128_ps(s1, s2, 0x20);
        let r2 = _mm256_permute2f128_ps(s3, s4, 0x20);
        let r3 = _mm256_permute2f128_ps(s1, s2, 0x31);
        let r4 = _mm256_permute2f128_ps(s3, s4, 0x31);
        // record the score for the current position
        let row = &mut data[i];
        _mm256_store_ps(row[0x00..].as_mut_ptr(), r1);
        _mm256_store_ps(row[0x08..].as_mut_ptr(), r2);
        _mm256_store_ps(row[0x10..].as_mut_ptr(), r3);
        _mm256_store_ps(row[0x18..].as_mut_ptr(), r4);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn best_position_avx2(scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
    if scores.len() == 0 {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm256_setzero_ps();
            let mut p2 = _mm256_setzero_ps();
            let mut p3 = _mm256_setzero_ps();
            let mut p4 = _mm256_setzero_ps();
            // store the best scores for each column
            let mut s1 = _mm256_load_ps(data[0][0x00..].as_ptr());
            let mut s2 = _mm256_load_ps(data[0][0x08..].as_ptr());
            let mut s3 = _mm256_load_ps(data[0][0x10..].as_ptr());
            let mut s4 = _mm256_load_ps(data[0][0x18..].as_ptr());
            // process all rows iteratively
            for (i, row) in data.iter().enumerate() {
                // record the current row index
                let index = _mm256_castsi256_ps(_mm256_set1_epi32(i as i32));
                // load scores for the current row
                let r1 = _mm256_load_ps(row[0x00..].as_ptr());
                let r2 = _mm256_load_ps(row[0x08..].as_ptr());
                let r3 = _mm256_load_ps(row[0x10..].as_ptr());
                let r4 = _mm256_load_ps(row[0x18..].as_ptr());
                // compare scores to local maximums
                let c1 = _mm256_cmp_ps(s1, r1, _CMP_LT_OS);
                let c2 = _mm256_cmp_ps(s2, r2, _CMP_LT_OS);
                let c3 = _mm256_cmp_ps(s3, r3, _CMP_LT_OS);
                let c4 = _mm256_cmp_ps(s4, r4, _CMP_LT_OS);
                // replace indices of new local maximums
                p1 = _mm256_blendv_ps(p1, index, c1);
                p2 = _mm256_blendv_ps(p2, index, c2);
                p3 = _mm256_blendv_ps(p3, index, c3);
                p4 = _mm256_blendv_ps(p4, index, c4);
                // replace values of new local maximums
                s1 = _mm256_blendv_ps(s1, r1, c1);
                s2 = _mm256_blendv_ps(s2, r2, c2);
                s3 = _mm256_blendv_ps(s3, r3, c3);
                s4 = _mm256_blendv_ps(s4, r4, c4);
            }
            // find the global maximum across all columns
            let mut x: [u32; 32] = [0; 32];
            _mm256_storeu_si256(x[0x00..].as_mut_ptr() as *mut _, _mm256_castps_si256(p1));
            _mm256_storeu_si256(x[0x08..].as_mut_ptr() as *mut _, _mm256_castps_si256(p2));
            _mm256_storeu_si256(x[0x10..].as_mut_ptr() as *mut _, _mm256_castps_si256(p3));
            _mm256_storeu_si256(x[0x18..].as_mut_ptr() as *mut _, _mm256_castps_si256(p4));
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

/// Intel 256-bit vector implementation, for 32 elements column width.
impl Avx2 {
    #[allow(unused)]
    pub fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<<Avx2 as Backend>::LANES>)
    where
        S: AsRef<StripedSequence<Dna, <Avx2 as Backend>::LANES>>,
        M: AsRef<ScoringMatrix<Dna>>,
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
            score_avx2(seq, pssm, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn best_position(scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            best_position_avx2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }
}
