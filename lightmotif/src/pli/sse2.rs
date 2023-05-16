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

#[target_feature(enable = "sse2")]
unsafe fn score_sse2<A: Alphabet>(
    seq: &StripedSequence<A, <__m128i as Vector>::LANES>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<<__m128i as Vector>::LANES>,
) {
    // mask vectors for broadcasting uint8x16_t to uint32x4_t to floatx4_t
    let zero = _mm_setzero_si128();
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
            let x = _mm_load_si128(seq.data[i + j].as_ptr() as *const __m128i);
            let hi = _mm_unpackhi_epi8(x, zero);
            let lo = _mm_unpacklo_epi8(x, zero);
            let x1 = _mm_unpacklo_epi8(lo, zero);
            let x2 = _mm_unpackhi_epi8(lo, zero);
            let x3 = _mm_unpacklo_epi8(hi, zero);
            let x4 = _mm_unpackhi_epi8(hi, zero);
            // load row for current weight matrix position
            let row = pssm.weights()[j].as_ptr();
            // index lookup table with each bases incrementally
            for i in 0..A::K::USIZE {
                let sym = _mm_set1_epi32(i as i32);
                let lut = _mm_set1_ps(*row.add(i as usize));
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
        let row = &mut scores.data[i];
        _mm_storeu_ps(row[0..].as_mut_ptr(), s1);
        _mm_storeu_ps(row[4..].as_mut_ptr(), s2);
        _mm_storeu_ps(row[8..].as_mut_ptr(), s3);
        _mm_storeu_ps(row[12..].as_mut_ptr(), s4);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn best_position_sse2(scores: &StripedScores<<__m128i as Vector>::LANES>) -> Option<usize> {
    if scores.length == 0 {
        None
    } else {
        unsafe {
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm_setzero_ps();
            let mut p2 = _mm_setzero_ps();
            let mut p3 = _mm_setzero_ps();
            let mut p4 = _mm_setzero_ps();
            // store the best scores for each column
            let mut s1 = _mm_load_ps(scores.data[0][0x00..].as_ptr());
            let mut s2 = _mm_load_ps(scores.data[0][0x04..].as_ptr());
            let mut s3 = _mm_load_ps(scores.data[0][0x08..].as_ptr());
            let mut s4 = _mm_load_ps(scores.data[0][0x0c..].as_ptr());
            // process all rows iteratively
            for (i, row) in scores.data.iter().enumerate() {
                // record the current row index
                let index = _mm_castsi128_ps(_mm_set1_epi32(i as i32));
                // load scores for the current row
                let r1 = _mm_load_ps(row[0x00..].as_ptr());
                let r2 = _mm_load_ps(row[0x04..].as_ptr());
                let r3 = _mm_load_ps(row[0x08..].as_ptr());
                let r4 = _mm_load_ps(row[0x0c..].as_ptr());
                // compare scores to local maximums
                let c1 = _mm_cmplt_ps(s1, r1);
                let c2 = _mm_cmplt_ps(s2, r2);
                let c3 = _mm_cmplt_ps(s3, r3);
                let c4 = _mm_cmplt_ps(s4, r4);
                // NOTE: code below could use `_mm_blendv_ps` instead,
                //       but this instruction is only available on SSE4.1
                //       while the rest of the code is actually using at
                //       most SSSE3 instructions.
                // replace indices of new local maximums
                p1 = _mm_or_ps(_mm_andnot_ps(c1, p1), _mm_and_ps(index, c1));
                p2 = _mm_or_ps(_mm_andnot_ps(c2, p2), _mm_and_ps(index, c2));
                p3 = _mm_or_ps(_mm_andnot_ps(c3, p3), _mm_and_ps(index, c3));
                p4 = _mm_or_ps(_mm_andnot_ps(c4, p4), _mm_and_ps(index, c4));
                // replace values of new local maximums
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
                if scores.data[row as usize][col] > best_score {
                    best_score = scores.data[row as usize][col];
                    best_pos = col * scores.data.rows() + row as usize;
                }
            }
            Some(best_pos)
        }
    }
}

/// Intel 128-bit vector implementation, for 16 elements column width.
impl<A: Alphabet> Score<A, __m128i> for Pipeline<A, __m128i> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<<__m128i as Vector>::LANES>)
    where
        S: AsRef<StripedSequence<A, <__m128i as Vector>::LANES>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap < pssm.len() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.len()
            );
        }
        if scores.data.rows() < (seq.data.rows() - seq.wrap) {
            panic!("not enough rows for scores: {}", pssm.len());
        }

        scores.length = seq.length - pssm.len() + 1;
        unsafe {
            score_sse2(seq, pssm, scores);
        }
    }

    fn best_position(scores: &StripedScores<<__m128i as Vector>::LANES>) -> Option<usize> {
        unsafe { best_position_sse2(scores) }
    }
}
