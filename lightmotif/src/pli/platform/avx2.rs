//! Intel 256-bit vector implementation, for 32 elements column width.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Range;

use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::dense::MatrixCoordinates;
use crate::err::InvalidSymbol;
use crate::num::IsLessOrEqual;
use crate::num::NonZero;
use crate::num::Unsigned;
use crate::num::U16;
use crate::num::U32;
use crate::num::U5;
use crate::num::U8;
use crate::pli::Encode;
use crate::pli::Pipeline;
use crate::pwm::ScoringMatrix;
use crate::scores::StripedScores;
use crate::seq::StripedSequence;

use super::Backend;

/// A marker type for the AVX2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Avx2;

impl Backend for Avx2 {
    type LANES = U32;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(overflowing_literals)]
unsafe fn encode_into_avx2<A>(seq: &[u8], dst: &mut [A::Symbol]) -> Result<(), InvalidSymbol>
where
    A: Alphabet,
{
    const STRIDE: usize = std::mem::size_of::<__m256i>();

    let alphabet = A::as_str().as_bytes();
    let g = Pipeline::<A, _>::generic();
    let l = seq.len();
    assert_eq!(seq.len(), dst.len());

    unsafe {
        // Use raw pointers since we cannot be sure `seq` and `dst` are aligned.
        let mut i = 0;
        let mut src_ptr = seq.as_ptr();
        let mut dst_ptr = dst.as_mut_ptr();

        // Store a flag to know if invalid letters have been encountered.
        let mut error = _mm256_setzero_si256();

        // Process the beginning of the sequence in SIMD while possible.
        while i + STRIDE <= l {
            // Load current row and reset buffers for the encoded result.
            let letters = _mm256_loadu_si256(src_ptr as *const __m256i);
            let mut encoded = _mm256_set1_epi8(A::K::USIZE as i8);
            let mut unknown = _mm256_set1_epi8(0xFF);
            // Check symbols one by one and match them to the letters.
            for a in 0..A::K::USIZE {
                let index = _mm256_set1_epi8(a as i8);
                let ascii = _mm256_set1_epi8(alphabet[a] as i8);
                let m = _mm256_cmpeq_epi8(letters, ascii);
                encoded = _mm256_blendv_epi8(encoded, index, m);
                unknown = _mm256_andnot_si256(m, unknown);
            }
            // Record is some symbols of the current vector are unknown.
            error = _mm256_or_si256(error, unknown);
            // Store the encoded result to the output buffer.
            _mm256_storeu_si256(dst_ptr as *mut __m256i, encoded);
            // Advance to the next addresses in input and output.
            src_ptr = src_ptr.add(STRIDE);
            dst_ptr = dst_ptr.add(STRIDE);
            i += STRIDE;
        }

        // If an invalid symbol was encountered, recover which one.
        // FIXME: run a vectorize the error search?
        if _mm256_testz_si256(error, error) != 1 {
            for i in 0..l {
                A::Symbol::from_ascii(seq[i])?;
            }
        }

        // Encode the rest of the sequence using the generic implementation.
        if i < l {
            g.encode_into(&seq[i..], &mut dst[i..])?;
        }
    }

    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(overflowing_literals)]
unsafe fn score_f32_avx2_permute<A>(
    pssm: &DenseMatrix<f32, A::K>,
    seq: &StripedSequence<A, <Avx2 as Backend>::LANES>,
    rows: Range<usize>,
    scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
) where
    A: Alphabet,
    <A as Alphabet>::K: IsLessOrEqual<U8>,
    <<A as Alphabet>::K as IsLessOrEqual<U8>>::Output: NonZero,
{
    use crate::dense::DenseMatrix;

    let data = scores.matrix_mut();
    debug_assert!(data.rows() > 0);

    let mut rowptr = data[0].as_mut_ptr();
    // constant vector for comparing unknown bases
    let n = _mm256_set1_epi32(<A as Alphabet>::K::I32 - 1);
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
    for i in rows {
        // reset sums for current position
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        // reset pointers to row
        let mut seqptr = seq.matrix()[i].as_ptr();
        let mut pssmptr = pssm[0].as_ptr();
        // advance position in the position weight matrix
        for _ in 0..pssm.rows() {
            // load sequence row and broadcast to f32
            debug_assert_eq!(seqptr as usize & 0x1f, 0);
            let x = _mm256_load_si256(seqptr as *const __m256i);
            let x1 = _mm256_shuffle_epi8(x, m1);
            let x2 = _mm256_shuffle_epi8(x, m2);
            let x3 = _mm256_shuffle_epi8(x, m3);
            let x4 = _mm256_shuffle_epi8(x, m4);
            // load row for current weight matrix position
            // debug_assert_eq!(pssmptr as usize & 0x1f, 0);
            debug_assert_eq!(pssmptr as usize & 0x1f, 0);
            let t = _mm256_load_ps(pssmptr);
            // index A/T/G/C/N lookup table with the bases
            let b1 = _mm256_permutevar8x32_ps(t, x1);
            let b2 = _mm256_permutevar8x32_ps(t, x2);
            let b3 = _mm256_permutevar8x32_ps(t, x3);
            let b4 = _mm256_permutevar8x32_ps(t, x4);
            // add log odds to the running sum
            s1 = _mm256_add_ps(s1, b1);
            s2 = _mm256_add_ps(s2, b2);
            s3 = _mm256_add_ps(s3, b3);
            s4 = _mm256_add_ps(s4, b4);
            // advance to next row in PSSM and sequence matrices
            seqptr = seqptr.add(seq.matrix().stride());
            pssmptr = pssmptr.add(pssm.stride());
        }
        // permute lanes so that scores are in the right order
        let r1 = _mm256_permute2f128_ps(s1, s2, 0x20);
        let r2 = _mm256_permute2f128_ps(s3, s4, 0x20);
        let r3 = _mm256_permute2f128_ps(s1, s2, 0x31);
        let r4 = _mm256_permute2f128_ps(s3, s4, 0x31);
        // record the score for the current position
        _mm256_stream_ps(rowptr.add(0x00), r1);
        _mm256_stream_ps(rowptr.add(0x08), r2);
        _mm256_stream_ps(rowptr.add(0x10), r3);
        _mm256_stream_ps(rowptr.add(0x18), r4);
        rowptr = rowptr.add(data.stride());
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(overflowing_literals)]
unsafe fn score_f32_avx2_gather<A>(
    pssm: &DenseMatrix<f32, A::K>,
    seq: &StripedSequence<A, <Avx2 as Backend>::LANES>,
    rows: Range<usize>,
    scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
) where
    A: Alphabet,
{
    let data = scores.matrix_mut();
    let mut rowptr = data[0].as_mut_ptr();
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
    for i in rows {
        // reset sums for current position
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        // reset pointers to row
        let mut seqptr = seq.matrix()[i].as_ptr();
        let mut pssmptr = pssm[0].as_ptr();
        // advance position in the position weight matrix
        for _ in 0..pssm.rows() {
            // load sequence row and broadcast to f32
            debug_assert_eq!(seqptr as usize & 0x1f, 0);
            let x = _mm256_load_si256(seqptr as *const __m256i);
            let x1 = _mm256_shuffle_epi8(x, m1);
            let x2 = _mm256_shuffle_epi8(x, m2);
            let x3 = _mm256_shuffle_epi8(x, m3);
            let x4 = _mm256_shuffle_epi8(x, m4);
            // gather scores for the sequence elements
            let b1 = _mm256_i32gather_ps(pssmptr, x1, std::mem::size_of::<f32>() as i32);
            let b2 = _mm256_i32gather_ps(pssmptr, x2, std::mem::size_of::<f32>() as i32);
            let b3 = _mm256_i32gather_ps(pssmptr, x3, std::mem::size_of::<f32>() as i32);
            let b4 = _mm256_i32gather_ps(pssmptr, x4, std::mem::size_of::<f32>() as i32);
            // add log odds to the running sum
            s1 = _mm256_add_ps(s1, b1);
            s2 = _mm256_add_ps(s2, b2);
            s3 = _mm256_add_ps(s3, b3);
            s4 = _mm256_add_ps(s4, b4);
            // advance to next row in PSSM and sequence matrices
            seqptr = seqptr.add(seq.matrix().stride());
            pssmptr = pssmptr.add(pssm.stride());
        }
        // permute lanes so that scores are in the right order
        let r1 = _mm256_permute2f128_ps(s1, s2, 0x20);
        let r2 = _mm256_permute2f128_ps(s3, s4, 0x20);
        let r3 = _mm256_permute2f128_ps(s1, s2, 0x31);
        let r4 = _mm256_permute2f128_ps(s3, s4, 0x31);
        // record the score for the current position
        _mm256_stream_ps(rowptr.add(0x00), r1);
        _mm256_stream_ps(rowptr.add(0x08), r2);
        _mm256_stream_ps(rowptr.add(0x10), r3);
        _mm256_stream_ps(rowptr.add(0x18), r4);
        rowptr = rowptr.add(data.stride());
    }

    // Required before returning to code that may set atomic flags that invite concurrent reads,
    // as LLVM lowers `AtomicBool::store(flag, true, Release)` to ordinary stores on x86-64
    // instead of SFENCE, even though SFENCE is required in the presence of nontemporal stores.
    _mm_sfence();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn score_u8_avx2_shuffle<A>(
    pssm: &DenseMatrix<u8, A::K>,
    seq: &StripedSequence<A, <Avx2 as Backend>::LANES>,
    rows: Range<usize>,
    scores: &mut StripedScores<u8, <Avx2 as Backend>::LANES>,
) where
    A: Alphabet,
{
    let data = scores.matrix_mut();
    let mut rowptr = data[0].as_mut_ptr() as *mut i8;
    // process every position of the sequence data
    for i in rows {
        // reset sums for current position
        let mut s = _mm256_setzero_si256();
        // reset pointers to row
        let mut seqptr = seq.matrix()[i].as_ptr();
        let mut pssmptr = pssm[0].as_ptr();
        // advance position in the position weight matrix
        for _ in 0..pssm.rows() {
            // load sequence row and broadcast to f32
            let x = _mm256_load_si256(seqptr as *const __m256i);
            // load row for current weight matrix position
            // NB: we need to broadcast it to the two lanes of the __m256i vector
            //     because in AVX2 shuffle operates on the two halves independently.
            let t = _mm256_broadcastsi128_si256(_mm_load_si128(&*(pssmptr as *const __m128i)));
            // load scores for given sequence
            let y = _mm256_shuffle_epi8(t, x);
            // add scores to the running sum
            s = _mm256_adds_epu8(s, y);
            // advance to next row in PSSM and sequence matrices
            seqptr = seqptr.add(seq.matrix().stride());
            pssmptr = pssmptr.add(pssm.stride());
        }
        // record the score for the current position
        _mm256_stream_si256(rowptr as *mut __m256i, s);
        rowptr = rowptr.add(data.stride());
    }
    _mm_sfence();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn argmax_f32_avx2(
    scores: &StripedScores<f32, <Avx2 as Backend>::LANES>,
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
            let mut dataptr = data[0].as_ptr();
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm256_setzero_ps();
            let mut p2 = _mm256_setzero_ps();
            let mut p3 = _mm256_setzero_ps();
            let mut p4 = _mm256_setzero_ps();
            // store the best scores for each column
            let mut s1 = _mm256_load_ps(dataptr.add(0x00));
            let mut s2 = _mm256_load_ps(dataptr.add(0x08));
            let mut s3 = _mm256_load_ps(dataptr.add(0x10));
            let mut s4 = _mm256_load_ps(dataptr.add(0x18));
            // process all rows iteratively
            for i in 0..data.rows() {
                // record the current row index
                let index = _mm256_castsi256_ps(_mm256_set1_epi32(i as i32));
                // load scores for the current row
                let r1 = _mm256_load_ps(dataptr.add(0x00));
                let r2 = _mm256_load_ps(dataptr.add(0x08));
                let r3 = _mm256_load_ps(dataptr.add(0x10));
                let r4 = _mm256_load_ps(dataptr.add(0x18));
                // compare scores to local maximums
                let c1 = _mm256_cmp_ps(s1, r1, _CMP_LE_OS);
                let c2 = _mm256_cmp_ps(s2, r2, _CMP_LE_OS);
                let c3 = _mm256_cmp_ps(s3, r3, _CMP_LE_OS);
                let c4 = _mm256_cmp_ps(s4, r4, _CMP_LE_OS);
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
                // advance to next row
                dataptr = dataptr.add(data.stride());
            }
            // find the global maximum across all columns
            let mut x: [u32; 32] = [0; 32];
            _mm256_storeu_si256(x[0x00..].as_mut_ptr() as *mut _, _mm256_castps_si256(p1));
            _mm256_storeu_si256(x[0x08..].as_mut_ptr() as *mut _, _mm256_castps_si256(p2));
            _mm256_storeu_si256(x[0x10..].as_mut_ptr() as *mut _, _mm256_castps_si256(p3));
            _mm256_storeu_si256(x[0x18..].as_mut_ptr() as *mut _, _mm256_castps_si256(p4));

            let mut best_pos = MatrixCoordinates::default();
            let mut best_score = data[best_pos];

            for (col, &row) in x.iter().enumerate() {
                let pos = MatrixCoordinates::new(row as usize, col);
                let score = data[pos];
                if score > best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }

            //     if score > best_score || (score == best_score && (row, col) < (best_row, best_col))
            //     {
            //         best_score = data[row as usize][col];
            //         best_row = row;
            //         best_col = col;
            //     }
            // }
            Some(best_pos)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn max_f32_avx2(scores: &StripedScores<f32, <Avx2 as Backend>::LANES>) -> Option<f32> {
    if scores.is_empty() {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            let mut dataptr = data[0].as_ptr();
            // store the best scores for each column
            let mut m1 = _mm256_setzero_ps();
            let mut m2 = _mm256_setzero_ps();
            let mut m3 = _mm256_setzero_ps();
            let mut m4 = _mm256_setzero_ps();
            // process all rows iteratively
            for i in 0..data.rows() {
                // load scores for the current row
                let r1 = _mm256_load_ps(dataptr as *const _);
                let r2 = _mm256_load_ps(dataptr.add(0x08) as *const _);
                let r3 = _mm256_load_ps(dataptr.add(0x10) as *const _);
                let r4 = _mm256_load_ps(dataptr.add(0x18) as *const _);
                // find highest score
                m1 = _mm256_max_ps(m1, r1);
                m2 = _mm256_max_ps(m2, r2);
                m3 = _mm256_max_ps(m3, r3);
                m4 = _mm256_max_ps(m4, r4);
                // advance to next row
                dataptr = dataptr.add(data.stride());
            }

            //
            let m = _mm256_max_ps(_mm256_max_ps(m1, m2), _mm256_max_ps(m3, m4));

            // find the global maximum across all columns
            let mut x: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(x.as_mut_ptr() as *mut _, m);
            x.into_iter().reduce(f32::max)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn argmax_u8_avx2(
    scores: &StripedScores<u8, <Avx2 as Backend>::LANES>,
) -> Option<MatrixCoordinates> {
    if scores.matrix().rows() > u16::MAX as usize + 1 {
        panic!(
            "This implementation only supports matrices with at most {} rows, found a sequence with {} rows. Contact the developers at https://github.com/althonos/lightmotif.",
            u16::MAX, scores.matrix().rows()
        );
    } else if scores.is_empty() {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut dataptr = data[0].as_ptr();
            // the row index for the best score in each column
            // (these are 32-bit integers but for use with `_mm256_blendv_ps`
            // they get stored in 32-bit float vectors).
            let mut p1 = _mm256_setzero_si256();
            let mut p2 = _mm256_setzero_si256();
            // store the best scores for each column
            let mut s1 = _mm256_set1_epi16(-1);
            let mut s2 = _mm256_set1_epi16(-1);
            // process all rows iteratively
            for i in 0..data.rows() {
                // record the current row index
                let index = _mm256_set1_epi16(i as i16);
                // load scores for the current row
                let r = _mm256_load_si256(dataptr as *const _);
                // unpack scores into 16-bit vectors (we can't use 8-bit
                // vectors directly because AVX2 doesn't support unsigned
                // comparisons with 8-bit integers, so we need to translate
                // them to signed comparisons in 16-bit space)
                let r1 = _mm256_unpacklo_epi8(r, _mm256_setzero_si256());
                let r2 = _mm256_unpackhi_epi8(r, _mm256_setzero_si256());
                // compare scores to local maximums
                let c1 = _mm256_cmpgt_epi16(r1, s1);
                let c2 = _mm256_cmpgt_epi16(r2, s2);
                // replace indices of new local maximums
                p1 = _mm256_blendv_epi8(p1, index, c1);
                p2 = _mm256_blendv_epi8(p2, index, c2);
                // replace values of new local maximums (minus one, so that
                // we can do a `_mm256_cmpgt_epi16` comparison instead of a
                // `_mm256_cmpge_epi16` which doesn't exist on AVX2)
                s1 = _mm256_blendv_epi8(s1, _mm256_sub_epi16(r1, ones), c1);
                s2 = _mm256_blendv_epi8(s2, _mm256_sub_epi16(r2, ones), c2);
                // advance to next row
                dataptr = dataptr.add(data.stride());
            }
            // record the column-local maxima
            let mut x: [u16; 32] = [0; 32];
            _mm256_storeu_si256(x.as_mut_ptr() as *mut _, p1);
            _mm256_storeu_si256(x[16..].as_mut_ptr() as *mut _, p2);
            // find the global maximum across all columns
            x.into_iter()
                .enumerate()
                .map(|(col, row)| MatrixCoordinates::new(row as usize, col))
                .max_by_key(|&pos| &data[pos])
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn max_u8_avx2(scores: &StripedScores<u8, <Avx2 as Backend>::LANES>) -> Option<u8> {
    if scores.is_empty() {
        None
    } else {
        let data = scores.matrix();
        unsafe {
            let mut dataptr = data[0].as_ptr();
            // store the best scores for each column
            let mut m = _mm256_setzero_si256();
            // process all rows iteratively
            for i in 0..data.rows() {
                // load scores for the current row
                let r = _mm256_load_si256(dataptr as *const _);
                // find highest score
                m = _mm256_max_epu8(m, r);
                // advance to next row
                dataptr = dataptr.add(data.stride());
            }
            // find the global maximum across all columns
            let mut x: [u8; 32] = [0; 32];
            _mm256_storeu_si256(x.as_mut_ptr() as *mut _, m);
            x.into_iter().max()
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn stripe_avx2<A>(
    seq: &[A::Symbol],
    striped: &mut StripedSequence<A, <Avx2 as Backend>::LANES>,
) where
    A: Alphabet,
{
    macro_rules! unpack {
        (epi128, $a:ident, $b:ident) => {{
            let t = $a;
            $a = _mm256_permute2x128_si256(t, $b, 32);
            $b = _mm256_permute2x128_si256(t, $b, 49);
        }};
        (epi8, $a:ident, $b:ident) => {{
            let t = $a;
            $a = _mm256_unpacklo_epi8(t, $b);
            $b = _mm256_unpackhi_epi8(t, $b);
        }};
        (epi16, $a:ident, $b:ident) => {{
            let t = $a;
            $a = _mm256_unpacklo_epi16(t, $b);
            $b = _mm256_unpackhi_epi16(t, $b);
        }};
        (epi32, $a:ident, $b:ident) => {{
            let t = $a;
            $a = _mm256_unpacklo_epi32(t, $b);
            $b = _mm256_unpackhi_epi32(t, $b);
        }};
        (epi64, $a:ident, $b:ident) => {{
            let t = $a;
            $a = _mm256_unpacklo_epi64(t, $b);
            $b = _mm256_unpackhi_epi64(t, $b);
        }};
    }

    // Compute sequence and matrix dimensions
    let s = seq;
    let length = s.len();
    let mut src = s.as_ptr() as *const u8;
    let src_stride =
        (length + (<Avx2 as Backend>::LANES::USIZE - 1)) / <Avx2 as Backend>::LANES::USIZE;

    // Get the matrix from the given striped sequence and resize it
    let mut matrix = std::mem::take(striped).into_matrix();
    matrix.resize(src_stride);

    // Early exit if sequence is empty (no allocated matrix).
    if length == 0 {
        return;
    }

    /// Get a pointer to the matrix
    let mut out = matrix[0].as_mut_ptr();
    let out_stride = matrix.stride();
    assert_eq!(matrix.rows(), src_stride);

    // Process sequence block by block
    let mut i = 0;
    while i + <Avx2 as Backend>::LANES::USIZE <= src_stride {
        let mut r00 = _mm256_loadu_si256(src.add(00 * src_stride) as _);
        let mut r01 = _mm256_loadu_si256(src.add(src_stride) as _);
        let mut r02 = _mm256_loadu_si256(src.add(02 * src_stride) as _);
        let mut r03 = _mm256_loadu_si256(src.add(03 * src_stride) as _);
        let mut r04 = _mm256_loadu_si256(src.add(04 * src_stride) as _);
        let mut r05 = _mm256_loadu_si256(src.add(05 * src_stride) as _);
        let mut r06 = _mm256_loadu_si256(src.add(06 * src_stride) as _);
        let mut r07 = _mm256_loadu_si256(src.add(07 * src_stride) as _);
        let mut r08 = _mm256_loadu_si256(src.add(08 * src_stride) as _);
        let mut r09 = _mm256_loadu_si256(src.add(09 * src_stride) as _);
        let mut r10 = _mm256_loadu_si256(src.add(10 * src_stride) as _);
        let mut r11 = _mm256_loadu_si256(src.add(11 * src_stride) as _);
        let mut r12 = _mm256_loadu_si256(src.add(12 * src_stride) as _);
        let mut r13 = _mm256_loadu_si256(src.add(13 * src_stride) as _);
        let mut r14 = _mm256_loadu_si256(src.add(14 * src_stride) as _);
        let mut r15 = _mm256_loadu_si256(src.add(15 * src_stride) as _);
        let mut r16 = _mm256_loadu_si256(src.add(16 * src_stride) as _);
        let mut r17 = _mm256_loadu_si256(src.add(17 * src_stride) as _);
        let mut r18 = _mm256_loadu_si256(src.add(18 * src_stride) as _);
        let mut r19 = _mm256_loadu_si256(src.add(19 * src_stride) as _);
        let mut r20 = _mm256_loadu_si256(src.add(20 * src_stride) as _);
        let mut r21 = _mm256_loadu_si256(src.add(21 * src_stride) as _);
        let mut r22 = _mm256_loadu_si256(src.add(22 * src_stride) as _);
        let mut r23 = _mm256_loadu_si256(src.add(23 * src_stride) as _);
        let mut r24 = _mm256_loadu_si256(src.add(24 * src_stride) as _);
        let mut r25 = _mm256_loadu_si256(src.add(25 * src_stride) as _);
        let mut r26 = _mm256_loadu_si256(src.add(26 * src_stride) as _);
        let mut r27 = _mm256_loadu_si256(src.add(27 * src_stride) as _);
        let mut r28 = _mm256_loadu_si256(src.add(28 * src_stride) as _);
        let mut r29 = _mm256_loadu_si256(src.add(29 * src_stride) as _);
        let mut r30 = _mm256_loadu_si256(src.add(30 * src_stride) as _);
        let mut r31 = _mm256_loadu_si256(src.add(31 * src_stride) as _);

        unpack!(epi8, r00, r01);
        unpack!(epi8, r02, r03);
        unpack!(epi8, r04, r05);
        unpack!(epi8, r06, r07);
        unpack!(epi8, r08, r09);
        unpack!(epi8, r10, r11);
        unpack!(epi8, r12, r13);
        unpack!(epi8, r14, r15);
        unpack!(epi8, r16, r17);
        unpack!(epi8, r18, r19);
        unpack!(epi8, r20, r21);
        unpack!(epi8, r22, r23);
        unpack!(epi8, r24, r25);
        unpack!(epi8, r26, r27);
        unpack!(epi8, r28, r29);
        unpack!(epi8, r30, r31);

        unpack!(epi16, r00, r02);
        unpack!(epi16, r01, r03);
        unpack!(epi16, r04, r06);
        unpack!(epi16, r05, r07);
        unpack!(epi16, r08, r10);
        unpack!(epi16, r09, r11);
        unpack!(epi16, r12, r14);
        unpack!(epi16, r13, r15);
        unpack!(epi16, r16, r18);
        unpack!(epi16, r17, r19);
        unpack!(epi16, r20, r22);
        unpack!(epi16, r21, r23);
        unpack!(epi16, r24, r26);
        unpack!(epi16, r25, r27);
        unpack!(epi16, r28, r30);
        unpack!(epi16, r29, r31);

        unpack!(epi32, r00, r04);
        unpack!(epi32, r02, r06);
        unpack!(epi32, r01, r05);
        unpack!(epi32, r03, r07);
        unpack!(epi32, r08, r12);
        unpack!(epi32, r10, r14);
        unpack!(epi32, r09, r13);
        unpack!(epi32, r11, r15);
        unpack!(epi32, r16, r20);
        unpack!(epi32, r18, r22);
        unpack!(epi32, r17, r21);
        unpack!(epi32, r19, r23);
        unpack!(epi32, r24, r28);
        unpack!(epi32, r26, r30);
        unpack!(epi32, r25, r29);
        unpack!(epi32, r27, r31);

        unpack!(epi64, r00, r08);
        unpack!(epi64, r04, r12);
        unpack!(epi64, r02, r10);
        unpack!(epi64, r06, r14);
        unpack!(epi64, r01, r09);
        unpack!(epi64, r05, r13);
        unpack!(epi64, r03, r11);
        unpack!(epi64, r07, r15);
        unpack!(epi64, r16, r24);
        unpack!(epi64, r20, r28);
        unpack!(epi64, r18, r26);
        unpack!(epi64, r22, r30);
        unpack!(epi64, r17, r25);
        unpack!(epi64, r21, r29);
        unpack!(epi64, r19, r27);
        unpack!(epi64, r23, r31);

        unpack!(epi128, r00, r16);
        unpack!(epi128, r08, r24);
        unpack!(epi128, r04, r20);
        unpack!(epi128, r12, r28);
        unpack!(epi128, r02, r18);
        unpack!(epi128, r10, r26);
        unpack!(epi128, r06, r22);
        unpack!(epi128, r14, r30);
        unpack!(epi128, r01, r17);
        unpack!(epi128, r09, r25);
        unpack!(epi128, r05, r21);
        unpack!(epi128, r13, r29);
        unpack!(epi128, r03, r19);
        unpack!(epi128, r11, r27);
        unpack!(epi128, r07, r23);
        unpack!(epi128, r15, r31);

        _mm256_stream_si256(out.add(0x00 * out_stride) as _, r00);
        _mm256_stream_si256(out.add(out_stride) as _, r08);
        _mm256_stream_si256(out.add(0x02 * out_stride) as _, r04);
        _mm256_stream_si256(out.add(0x03 * out_stride) as _, r12);
        _mm256_stream_si256(out.add(0x04 * out_stride) as _, r02);
        _mm256_stream_si256(out.add(0x05 * out_stride) as _, r10);
        _mm256_stream_si256(out.add(0x06 * out_stride) as _, r06);
        _mm256_stream_si256(out.add(0x07 * out_stride) as _, r14);
        _mm256_stream_si256(out.add(0x08 * out_stride) as _, r01);
        _mm256_stream_si256(out.add(0x09 * out_stride) as _, r09);
        _mm256_stream_si256(out.add(0x0a * out_stride) as _, r05);
        _mm256_stream_si256(out.add(0x0b * out_stride) as _, r13);
        _mm256_stream_si256(out.add(0x0c * out_stride) as _, r03);
        _mm256_stream_si256(out.add(0x0d * out_stride) as _, r11);
        _mm256_stream_si256(out.add(0x0e * out_stride) as _, r07);
        _mm256_stream_si256(out.add(0x0f * out_stride) as _, r15);
        _mm256_stream_si256(out.add(0x10 * out_stride) as _, r16);
        _mm256_stream_si256(out.add(0x11 * out_stride) as _, r24);
        _mm256_stream_si256(out.add(0x12 * out_stride) as _, r20);
        _mm256_stream_si256(out.add(0x13 * out_stride) as _, r28);
        _mm256_stream_si256(out.add(0x14 * out_stride) as _, r18);
        _mm256_stream_si256(out.add(0x15 * out_stride) as _, r26);
        _mm256_stream_si256(out.add(0x16 * out_stride) as _, r22);
        _mm256_stream_si256(out.add(0x17 * out_stride) as _, r30);
        _mm256_stream_si256(out.add(0x18 * out_stride) as _, r17);
        _mm256_stream_si256(out.add(0x19 * out_stride) as _, r25);
        _mm256_stream_si256(out.add(0x1a * out_stride) as _, r21);
        _mm256_stream_si256(out.add(0x1b * out_stride) as _, r29);
        _mm256_stream_si256(out.add(0x1c * out_stride) as _, r19);
        _mm256_stream_si256(out.add(0x1d * out_stride) as _, r27);
        _mm256_stream_si256(out.add(0x1e * out_stride) as _, r23);
        _mm256_stream_si256(out.add(0x1f * out_stride) as _, r31);

        out = out.add(0x20 * out_stride);
        src = src.add(0x20);
        i += <Avx2 as Backend>::LANES::USIZE;
    }

    // Required before returning to code that may set atomic flags that invite concurrent reads,
    // as LLVM lowers `AtomicBool::store(flag, true, Release)` to ordinary stores on x86-64
    // instead of SFENCE, even though SFENCE is required in the presence of nontemporal stores.
    _mm_sfence();

    // Take care of remaining columns.
    while i < matrix.rows() {
        for j in 0..32 {
            if j * src_stride + i < s.len() {
                matrix[i][j] = s[j * src_stride + i];
            }
        }
        i += 1;
    }

    // Fill end of the matrix after the sequence end.
    for k in s.len()..matrix.columns() * matrix.rows() {
        matrix[k % src_stride][k / src_stride] = A::default_symbol();
    }

    // Replace original striped sequence.
    *striped = StripedSequence::new(matrix, seq.len()).unwrap();
}

/// Intel 256-bit vector implementation, for 32 elements column width.
impl Avx2 {
    #[allow(unused)]
    pub fn encode_into<A>(seq: &[u8], dst: &mut [A::Symbol]) -> Result<(), InvalidSymbol>
    where
        A: Alphabet,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            encode_into_avx2::<A>(seq, dst)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host");
    }

    #[allow(unused)]
    pub fn score_f32_rows_into_permute<A, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) where
        A: Alphabet,
        <A as Alphabet>::K: IsLessOrEqual<U8>,
        <<A as Alphabet>::K as IsLessOrEqual<U8>>::Output: NonZero,
        S: AsRef<StripedSequence<A, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, A::K>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.rows() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.rows()
            );
        }

        if seq.len() < pssm.rows() || rows.len() == 0 {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_f32_avx2_permute(pssm, seq, rows, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn score_f32_rows_into_gather<A, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) where
        A: Alphabet,
        S: AsRef<StripedSequence<A, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<f32, A::K>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.rows() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.rows()
            );
        }

        if seq.len() < pssm.rows() || rows.len() == 0 {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_f32_avx2_gather(pssm, seq, rows, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn score_u8_rows_into_shuffle<A, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<u8, <Avx2 as Backend>::LANES>,
    ) where
        A: Alphabet,
        <A as Alphabet>::K: IsLessOrEqual<U16>,
        <<A as Alphabet>::K as IsLessOrEqual<U16>>::Output: NonZero,
        S: AsRef<StripedSequence<A, <Avx2 as Backend>::LANES>>,
        M: AsRef<DenseMatrix<u8, A::K>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.rows() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.rows()
            );
        }

        if seq.len() < pssm.rows() || rows.len() == 0 {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_u8_avx2_shuffle(pssm, seq, rows, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn argmax_f32(
        scores: &StripedScores<f32, <Avx2 as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            argmax_f32_avx2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn max_f32(scores: &StripedScores<f32, <Avx2 as Backend>::LANES>) -> Option<f32> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            max_f32_avx2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn argmax_u8(
        scores: &StripedScores<u8, <Avx2 as Backend>::LANES>,
    ) -> Option<MatrixCoordinates> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            argmax_u8_avx2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn max_u8(scores: &StripedScores<u8, <Avx2 as Backend>::LANES>) -> Option<u8> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            max_u8_avx2(scores)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn stripe_into<A, S>(seq: S, matrix: &mut StripedSequence<A, <Avx2 as Backend>::LANES>)
    where
        A: Alphabet,
        S: AsRef<[A::Symbol]>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            stripe_avx2(seq.as_ref(), matrix)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }
}
