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
    for offset in (0..<C as Div<U16>>::Output::USIZE).map(|i| i * U16::USIZE) {
        let mut rowptr = data[0].as_mut_ptr().add(offset);
        // process every position of the sequence data
        for i in 0..seq.data.rows() - seq.wrap {
            // reset sums for current position
            let mut s1 = _mm_setzero_ps();
            let mut s2 = _mm_setzero_ps();
            let mut s3 = _mm_setzero_ps();
            let mut s4 = _mm_setzero_ps();
            // reset position
            let mut dataptr = seq.data[i].as_ptr().add(offset);
            let mut pssmptr = pssm.weights()[0].as_ptr();
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
                dataptr = dataptr.add(seq.data.stride());
                pssmptr = pssmptr.add(pssm.weights().stride());
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
unsafe fn argmax_sse2<C>(scores: &StripedScores<C>) -> Option<usize>
where
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
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
            let mut best_col = [0u32; 16];
            let mut best_pos = 0;
            let mut best_score = -f32::INFINITY;
            for offset in (0..<C as Div<U16>>::Output::USIZE).map(|i| i * 16) {
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
                _mm_storeu_si128(
                    best_col[0x00..].as_mut_ptr() as *mut _,
                    _mm_castps_si128(p1),
                );
                _mm_storeu_si128(
                    best_col[0x04..].as_mut_ptr() as *mut _,
                    _mm_castps_si128(p2),
                );
                _mm_storeu_si128(
                    best_col[0x08..].as_mut_ptr() as *mut _,
                    _mm_castps_si128(p3),
                );
                _mm_storeu_si128(
                    best_col[0x0c..].as_mut_ptr() as *mut _,
                    _mm_castps_si128(p4),
                );
                for k in 0..U16::USIZE {
                    let row = best_col[k] as usize;
                    let col = k + offset;
                    let pos = col * data.rows() + row as usize;
                    let score = data[row][col];
                    if score > best_score || (score == best_score && pos < best_pos) {
                        best_score = data[row][col];
                        best_pos = pos;
                    }
                }
            }
            Some(best_pos)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn threshold_sse2<C>(scores: &StripedScores<C>, threshold: f32) -> Vec<usize>
where
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
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
        let mut indices = Vec::<u32>::with_capacity(data.columns() * rows);
        unsafe {
            indices.set_len(indices.capacity());
            // NOTE(@althonos): Using `u32::MAX` as a sentinel instead of `0`
            //                  because `0` may be a valid index.
            let max = _mm_set1_epi32(u32::MAX as i32);
            let t = _mm_set1_ps(threshold);
            let ones = _mm_set1_epi32(1);
            let mut dst = indices.as_mut_ptr() as *mut __m128i;
            for offset in (0..<C as Div<U16>>::Output::USIZE).map(|i| i * 16) {
                // compute real sequence index for each column of the striped scores
                let mut x1 = _mm_set_epi32(
                    ((offset + 3) * rows) as i32,
                    ((offset + 2) * rows) as i32,
                    ((offset + 1) * rows) as i32,
                    ((offset + 0) * rows) as i32,
                );
                let mut x2 = _mm_set_epi32(
                    ((offset + 7) * rows) as i32,
                    ((offset + 6) * rows) as i32,
                    ((offset + 5) * rows) as i32,
                    ((offset + 4) * rows) as i32,
                );
                let mut x3 = _mm_set_epi32(
                    ((offset + 11) * rows) as i32,
                    ((offset + 10) * rows) as i32,
                    ((offset + 9) * rows) as i32,
                    ((offset + 8) * rows) as i32,
                );
                let mut x4 = _mm_set_epi32(
                    ((offset + 15) * rows) as i32,
                    ((offset + 14) * rows) as i32,
                    ((offset + 13) * rows) as i32,
                    ((offset + 12) * rows) as i32,
                );
                // Process rows iteratively
                let mut dataptr = data[0].as_ptr();
                for _ in 0..data.rows() {
                    // load scores for the current row
                    let r1 = _mm_load_ps(dataptr.add(offset + 0x00));
                    let r2 = _mm_load_ps(dataptr.add(offset + 0x04));
                    let r3 = _mm_load_ps(dataptr.add(offset + 0x08));
                    let r4 = _mm_load_ps(dataptr.add(offset + 0x0c));
                    // check whether scores are greater or equal to the threshold
                    let m1 = _mm_castps_si128(_mm_cmplt_ps(t, r1));
                    let m2 = _mm_castps_si128(_mm_cmplt_ps(t, r2));
                    let m3 = _mm_castps_si128(_mm_cmplt_ps(t, r3));
                    let m4 = _mm_castps_si128(_mm_cmplt_ps(t, r4));
                    // NOTE: Code below could use `_mm_blendv_ps` instead,
                    //       but this instruction is only available on SSE4.1
                    //       while the rest of the code is actually using SSE2
                    //       instructions only.
                    // Mask indices that should be removed
                    let i1 = _mm_or_si128(_mm_and_si128(x1, m1), _mm_andnot_si128(m1, max));
                    let i2 = _mm_or_si128(_mm_and_si128(x2, m2), _mm_andnot_si128(m2, max));
                    let i3 = _mm_or_si128(_mm_and_si128(x3, m3), _mm_andnot_si128(m3, max));
                    let i4 = _mm_or_si128(_mm_and_si128(x4, m4), _mm_andnot_si128(m4, max));
                    // Store masked indices into the destination vector
                    _mm_storeu_si128(dst, i1);
                    _mm_storeu_si128(dst.add(1), i2);
                    _mm_storeu_si128(dst.add(2), i3);
                    _mm_storeu_si128(dst.add(3), i4);
                    // Advance result buffer to next row
                    dst = dst.add(4);
                    // Advance sequence indices to next row
                    x1 = _mm_add_epi32(x1, ones);
                    x2 = _mm_add_epi32(x2, ones);
                    x3 = _mm_add_epi32(x3, ones);
                    x4 = _mm_add_epi32(x4, ones);
                    // Advance data pointer to next row
                    dataptr = dataptr.add(data.stride());
                }
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

        if seq.length < pssm.len() {
            scores.resize(0, 0);
            return;
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
    pub fn argmax<C>(scores: &StripedScores<C>) -> Option<usize>
    where
        C: StrictlyPositive + Rem<U16> + Div<U16>,
        <C as Rem<U16>>::Output: Zero,
        <C as Div<U16>>::Output: Unsigned,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            argmax_sse2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn threshold<C>(scores: &StripedScores<C>, threshold: f32) -> Vec<usize>
    where
        C: StrictlyPositive + Rem<U16> + Div<U16>,
        <C as Rem<U16>>::Output: Zero,
        <C as Div<U16>>::Output: Unsigned,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            threshold_sse2(scores, threshold)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run SSE2 code on a non-x86 host")
    }
}
