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
    if scores.len() > u32::MAX as usize {
        panic!(
            "This implementation only supports sequences with at most {} positions, found a sequence with {} positions. Contact the developers at https://github.com/althonos/lightmotif.",
            u32::MAX, scores.len()
        );
    } else if scores.len() == 0 {
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn threshold_avx2(
    scores: &StripedScores<<Avx2 as Backend>::LANES>,
    threshold: f32,
) -> Vec<usize> {
    if scores.len() >= u32::MAX as usize {
        panic!(
            "This implementation only supports sequences with at most {} positions, found a sequence with {} positions. Contact the developers at https://github.com/althonos/lightmotif.",
            u32::MAX, scores.len()
        );
    } else if scores.len() == 0 {
        Vec::new()
    } else {
        let data = scores.matrix();
        let rows = data.rows();
        let mut indices = vec![0u32; data.columns() * rows];
        unsafe {
            // NOTE(@althonos): Using `u32::MAX` as a sentinel instead of `0`
            //                  because `0` may be a valid index.
            let max = _mm256_set1_epi32(u32::MAX as i32);
            let t = _mm256_set1_ps(threshold);
            let ones = _mm256_set1_epi32(1);
            let mut dst = indices.as_mut_ptr() as *mut __m256i;
            // compute real sequence index for each column of the striped scores
            let mut x1 = _mm256_set_epi32(
                (7 * rows) as i32,
                (6 * rows) as i32,
                (5 * rows) as i32,
                (4 * rows) as i32,
                (3 * rows) as i32,
                (2 * rows) as i32,
                (1 * rows) as i32,
                (0 * rows) as i32,
            );
            let mut x2 = _mm256_set_epi32(
                (15 * rows) as i32,
                (14 * rows) as i32,
                (13 * rows) as i32,
                (12 * rows) as i32,
                (11 * rows) as i32,
                (10 * rows) as i32,
                (9 * rows) as i32,
                (8 * rows) as i32,
            );
            let mut x3 = _mm256_set_epi32(
                (23 * rows) as i32,
                (22 * rows) as i32,
                (21 * rows) as i32,
                (20 * rows) as i32,
                (19 * rows) as i32,
                (18 * rows) as i32,
                (17 * rows) as i32,
                (16 * rows) as i32,
            );
            let mut x4 = _mm256_set_epi32(
                (31 * rows) as i32,
                (30 * rows) as i32,
                (29 * rows) as i32,
                (28 * rows) as i32,
                (27 * rows) as i32,
                (26 * rows) as i32,
                (25 * rows) as i32,
                (24 * rows) as i32,
            );
            // Process rows iteratively
            for row in data.iter() {
                // load scores for the current row
                let r1 = _mm256_load_ps(row[0x00..].as_ptr());
                let r2 = _mm256_load_ps(row[0x08..].as_ptr());
                let r3 = _mm256_load_ps(row[0x10..].as_ptr());
                let r4 = _mm256_load_ps(row[0x18..].as_ptr());
                // check whether scores are greater or equal to the threshold
                let m1 = _mm256_castps_si256(_mm256_cmp_ps(r1, t, _CMP_LT_OS));
                let m2 = _mm256_castps_si256(_mm256_cmp_ps(r2, t, _CMP_LT_OS));
                let m3 = _mm256_castps_si256(_mm256_cmp_ps(r3, t, _CMP_LT_OS));
                let m4 = _mm256_castps_si256(_mm256_cmp_ps(r4, t, _CMP_LT_OS));
                // Mask indices that should be removed
                let i1 = _mm256_blendv_epi8(x1, max, m1);
                let i2 = _mm256_blendv_epi8(x2, max, m2);
                let i3 = _mm256_blendv_epi8(x3, max, m3);
                let i4 = _mm256_blendv_epi8(x4, max, m4);
                // Store masked indices into the destination vector
                _mm256_storeu_si256(dst, i1);
                _mm256_storeu_si256(dst.add(1), i2);
                _mm256_storeu_si256(dst.add(2), i3);
                _mm256_storeu_si256(dst.add(3), i4);
                // Advance result buffer to next row
                dst = dst.add(4);
                // Advance sequence indices to next row
                x1 = _mm256_add_epi32(x1, ones);
                x2 = _mm256_add_epi32(x2, ones);
                x3 = _mm256_add_epi32(x3, ones);
                x4 = _mm256_add_epi32(x4, ones);
            }
        }

        // NOTE: Benchmarks suggest that `indices.retain(...)` is faster than
        //       `indices.into_iter().filter(...).

        // FIXME: The `Vec::retain` implementation may not be optimal for this,
        //        since it takes extra care of the vector elements deallocation
        //        because they may implement `Drop`. It may be faster to use
        //        a double-pointer algorithm, swapping sentinels and concrete
        //        values until the end of the vector is reached, and then
        //        clipping the vector with `indices.set_len`.

        // Remove all masked items and convert the indices to usize
        indices.retain(|&x| (x as usize) < scores.len());
        indices.into_iter().map(|i| i as usize).collect()
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

    #[allow(unused)]
    pub fn threshold(
        scores: &StripedScores<<Avx2 as Backend>::LANES>,
        threshold: f32,
    ) -> Vec<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            threshold_avx2(scores, threshold)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }
}
