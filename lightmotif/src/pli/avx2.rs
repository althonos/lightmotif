use std::arch::x86_64::*;

use typenum::Unsigned;

use super::Pipeline;
use super::Score;
use super::StripedScores;
use super::Vector;
use crate::abc::Alphabet;
use crate::abc::Dna;
use crate::abc::Nucleotide;
use crate::abc::Symbol;
use crate::pwm::ScoringMatrix;
use crate::seq::StripedSequence;

#[target_feature(enable = "avx2")]
unsafe fn score_avx2(
    seq: &StripedSequence<Dna, <__m256i as Vector>::LANES>,
    pssm: &ScoringMatrix<Dna>,
    scores: &mut StripedScores<<__m256i as Vector>::LANES>,
) {
    // constant vector for comparing unknown bases
    let n = _mm256_set1_epi8(Nucleotide::N as i8);
    // mask vectors for broadcasting uint8x32_t to uint32x8_t to floatx8_t
    let m1 = _mm256_set_epi32(
        0xFFFFFF03u32 as i32,
        0xFFFFFF02u32 as i32,
        0xFFFFFF01u32 as i32,
        0xFFFFFF00u32 as i32,
        0xFFFFFF03u32 as i32,
        0xFFFFFF02u32 as i32,
        0xFFFFFF01u32 as i32,
        0xFFFFFF00u32 as i32,
    );
    let m2 = _mm256_set_epi32(
        0xFFFFFF07u32 as i32,
        0xFFFFFF06u32 as i32,
        0xFFFFFF05u32 as i32,
        0xFFFFFF04u32 as i32,
        0xFFFFFF07u32 as i32,
        0xFFFFFF06u32 as i32,
        0xFFFFFF05u32 as i32,
        0xFFFFFF04u32 as i32,
    );
    let m3 = _mm256_set_epi32(
        0xFFFFFF0Bu32 as i32,
        0xFFFFFF0Au32 as i32,
        0xFFFFFF09u32 as i32,
        0xFFFFFF08u32 as i32,
        0xFFFFFF0Bu32 as i32,
        0xFFFFFF0Au32 as i32,
        0xFFFFFF09u32 as i32,
        0xFFFFFF08u32 as i32,
    );
    let m4 = _mm256_set_epi32(
        0xFFFFFF0Fu32 as i32,
        0xFFFFFF0Eu32 as i32,
        0xFFFFFF0Du32 as i32,
        0xFFFFFF0Cu32 as i32,
        0xFFFFFF0Fu32 as i32,
        0xFFFFFF0Eu32 as i32,
        0xFFFFFF0Du32 as i32,
        0xFFFFFF0Cu32 as i32,
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
            let c = _mm_load_ps(row);
            let t = _mm256_set_m128(c, c);
            let u = _mm256_set1_ps(*row.add(crate::abc::Nucleotide::N.as_index()));
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
        let row = &mut scores.data[i];
        _mm256_store_ps(row[0x00..].as_mut_ptr(), r1);
        _mm256_store_ps(row[0x08..].as_mut_ptr(), r2);
        _mm256_store_ps(row[0x10..].as_mut_ptr(), r3);
        _mm256_store_ps(row[0x18..].as_mut_ptr(), r4);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn best_position_avx2(scores: &StripedScores<<__m256i as Vector>::LANES>) -> Option<usize> {
    if scores.length == 0 {
        None
    } else {
        unsafe {
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm256_setzero_ps();
            let mut p2 = _mm256_setzero_ps();
            let mut p3 = _mm256_setzero_ps();
            let mut p4 = _mm256_setzero_ps();
            // store the best scores for each column
            let mut s1 = _mm256_load_ps(scores.data[0][0x00..].as_ptr());
            let mut s2 = _mm256_load_ps(scores.data[0][0x08..].as_ptr());
            let mut s3 = _mm256_load_ps(scores.data[0][0x10..].as_ptr());
            let mut s4 = _mm256_load_ps(scores.data[0][0x18..].as_ptr());
            // process all rows iteratively
            for (i, row) in scores.data.iter().enumerate() {
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
                if scores.data[row as usize][col] > best_score {
                    best_score = scores.data[row as usize][col];
                    best_pos = col * scores.data.rows() + row as usize;
                }
            }
            Some(best_pos)
        }
    }
}

/// Intel 256-bit vector implementation, for 32 elements column width.
impl Score<Dna, __m256i> for Pipeline<Dna, __m256i> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<<__m256i as Vector>::LANES>)
    where
        S: AsRef<StripedSequence<Dna, <__m256i as Vector>::LANES>>,
        M: AsRef<ScoringMatrix<Dna>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        let result = &mut scores.data;

        if seq.wrap < pssm.len() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.len()
            );
        }
        if result.rows() < (seq.data.rows() - seq.wrap) {
            panic!("not enough rows for scores: {}", pssm.len());
        }

        scores.length = seq.length - pssm.len() + 1;
        unsafe {
            score_avx2(seq, pssm, scores);
        }
    }

    fn best_position(scores: &StripedScores<<__m256i as Vector>::LANES>) -> Option<usize> {
        unsafe { best_position_avx2(scores) }
    }
}
