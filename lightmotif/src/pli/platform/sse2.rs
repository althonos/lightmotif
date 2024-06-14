//! Intel 128-bit vector implementation, for 16 elements column width.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Range;
use std::ops::Rem;

use super::Backend;
use crate::abc::Alphabet;
use crate::dense::MatrixCoordinates;
use crate::num::consts::U16;
use crate::num::MultipleOf;
use crate::num::StrictlyPositive;
use crate::num::Unsigned;
use crate::pwm::ScoringMatrix;
use crate::scores::StripedScores;
use crate::seq::StripedSequence;

/// A marker type for the SSE2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Sse2;

impl Backend for Sse2 {
    type LANES = U16;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn score_sse2<A: Alphabet, C: MultipleOf<<Sse2 as Backend>::LANES>>(
    pssm: &ScoringMatrix<A>,
    seq: &StripedSequence<A, C>,
    rows: Range<usize>,
    scores: &mut StripedScores<C>,
) {
    // mask vectors for broadcasting uint8x16_t to uint32x4_t to floatx4_t
    let zero = _mm_setzero_si128();
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..C::Quotient::USIZE).map(|i| i * <Sse2 as Backend>::LANES::USIZE) {
        let mut rowptr = data[0].as_mut_ptr().add(offset);
        // process every position of the sequence data
        for i in rows.clone() {
            // reset sums for current position
            let mut s1 = _mm_setzero_ps();
            let mut s2 = _mm_setzero_ps();
            let mut s3 = _mm_setzero_ps();
            let mut s4 = _mm_setzero_ps();
            // reset position
            let mut dataptr = seq.matrix()[i].as_ptr().add(offset);
            let mut pssmptr = pssm.matrix()[0].as_ptr();
            // advance position in the position weight matrix
            for _ in 0..pssm.len() {
                // load sequence row and broadcast to f32
                let x = _mm_load_si128(dataptr as *const __m128i);
                let hi = _mm_unpackhi_epi8(x, zero);
                let lo = _mm_unpacklo_epi8(x, zero);
                let x1 = _mm_unpacklo_epi8(lo, zero);
                let x2 = _mm_unpackhi_epi8(lo, zero);
                let x3 = _mm_unpacklo_epi8(hi, zero);
                let x4 = _mm_unpackhi_epi8(hi, zero);
                // index lookup table with each bases incrementally
                for k in 0..A::K::USIZE {
                    let sym = _mm_set1_epi32(k as i32);
                    let lut = _mm_load1_ps(pssmptr.add(k));
                    let p1 = _mm_castsi128_ps(_mm_cmpeq_epi32(x1, sym));
                    let p2 = _mm_castsi128_ps(_mm_cmpeq_epi32(x2, sym));
                    let p3 = _mm_castsi128_ps(_mm_cmpeq_epi32(x3, sym));
                    let p4 = _mm_castsi128_ps(_mm_cmpeq_epi32(x4, sym));
                    s1 = _mm_add_ps(s1, _mm_and_ps(lut, p1));
                    s2 = _mm_add_ps(s2, _mm_and_ps(lut, p2));
                    s3 = _mm_add_ps(s3, _mm_and_ps(lut, p3));
                    s4 = _mm_add_ps(s4, _mm_and_ps(lut, p4));
                }
                // advance to next row in sequence and PSSM matrices
                dataptr = dataptr.add(seq.matrix().stride());
                pssmptr = pssmptr.add(pssm.matrix().stride());
            }
            // record the score for the current position
            _mm_stream_ps(rowptr.add(0x00), s1);
            _mm_stream_ps(rowptr.add(0x04), s2);
            _mm_stream_ps(rowptr.add(0x08), s3);
            _mm_stream_ps(rowptr.add(0x0c), s4);
            rowptr = rowptr.add(data.stride());
        }
    }

    // Required before returning to code that may set atomic flags that invite concurrent reads,
    // as LLVM lowers `AtomicBool::store(flag, true, Release)` to ordinary stores on x86-64
    // instead of SFENCE, even though SFENCE is required in the presence of nontemporal stores.
    _mm_sfence();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn argmax_sse2<C: MultipleOf<<Sse2 as Backend>::LANES>>(
    scores: &StripedScores<C>,
) -> Option<MatrixCoordinates> {
    if scores.max_index() > u32::MAX as usize {
        panic!(
            "This implementation only supports sequences with at most {} positions, found a sequence with {} positions. Contact the developers at https://github.com/althonos/lightmotif.",
            u32::MAX, scores.max_index()
        );
    } else if scores.is_empty() {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            let mut output = [0u32; 16];
            let mut best_col = 0;
            let mut best_row = 0;
            let mut best_score = -f32::INFINITY;

            for offset in (0..C::Quotient::USIZE).map(|i| i * <Sse2 as Backend>::LANES::USIZE) {
                let mut dataptr = data[0].as_ptr().add(offset);
                // the row index for the best score in each column
                // (these are 32-bit integers but for use with `_mm256_blendv_ps`
                // they get stored in 32-bit float vectors).
                let mut p1 = _mm_setzero_ps();
                let mut p2 = _mm_setzero_ps();
                let mut p3 = _mm_setzero_ps();
                let mut p4 = _mm_setzero_ps();
                // store the best scores for each column
                let mut s1 = _mm_load_ps(dataptr.add(0x00));
                let mut s2 = _mm_load_ps(dataptr.add(0x04));
                let mut s3 = _mm_load_ps(dataptr.add(0x08));
                let mut s4 = _mm_load_ps(dataptr.add(0x0c));
                // process all rows iteratively
                for i in 0..data.rows() {
                    // record the current row index
                    let index = _mm_castsi128_ps(_mm_set1_epi32(i as i32));
                    // load scores for the current row
                    let r1 = _mm_load_ps(dataptr.add(0x00));
                    let r2 = _mm_load_ps(dataptr.add(0x04));
                    let r3 = _mm_load_ps(dataptr.add(0x08));
                    let r4 = _mm_load_ps(dataptr.add(0x0c));
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
                    // advance to next row
                    dataptr = dataptr.add(data.stride());
                }
                // find the global maximum across all columns
                _mm_storeu_si128(output[0x00..].as_mut_ptr() as *mut _, _mm_castps_si128(p1));
                _mm_storeu_si128(output[0x04..].as_mut_ptr() as *mut _, _mm_castps_si128(p2));
                _mm_storeu_si128(output[0x08..].as_mut_ptr() as *mut _, _mm_castps_si128(p3));
                _mm_storeu_si128(output[0x0c..].as_mut_ptr() as *mut _, _mm_castps_si128(p4));
                for k in 0..U16::USIZE {
                    let row = output[k] as usize;
                    let col = k + offset;
                    let score = data[row][col];
                    if score > best_score
                        || (score == best_score && (row, col) < (best_row, best_col))
                    {
                        best_score = data[row][col];
                        best_row = row;
                        best_col = col;
                    }
                }
            }
            Some(MatrixCoordinates::new(best_row, best_col))
        }
    }
}

impl Sse2 {
    #[allow(unused)]
    pub fn score_rows_into<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<C>,
    ) where
        A: Alphabet,
        C: MultipleOf<<Sse2 as Backend>::LANES>,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.len() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.len()
            );
        }

        if seq.len() < pssm.len() {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), seq.len() - pssm.len() + 1);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_sse2(pssm, seq, rows, scores);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn argmax<C: MultipleOf<<Sse2 as Backend>::LANES>>(
        scores: &StripedScores<C>,
    ) -> Option<MatrixCoordinates> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            argmax_sse2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }
}
