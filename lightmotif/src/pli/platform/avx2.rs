//! Intel 256-bit vector implementation, for 32 elements column width.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use typenum::consts::U32;
use typenum::consts::U5;
use typenum::IsLessOrEqual;
use typenum::NonZero;
use typenum::Unsigned;

use super::Backend;
use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::err::InvalidSymbol;
use crate::pli::scores::StripedScores;
use crate::pli::Encode;
use crate::pli::Pipeline;
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
unsafe fn encode_into_avx2<A>(seq: &[u8], dst: &mut [A::Symbol]) -> Result<(), InvalidSymbol>
where
    A: Alphabet,
{
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
        while i + std::mem::size_of::<__m256i>() < l {
            // Load current row and reset buffers for the encoded result.
            let letters = _mm256_loadu_si256(src_ptr as *const __m256i);
            let mut encoded = _mm256_setzero_si256();
            let mut unknown = _mm256_set1_epi8(0xFF);
            // Check symbols one by one and match them to the letters.
            for a in A::symbols() {
                let index = _mm256_set1_epi8(a.as_index() as i8);
                let ascii = _mm256_set1_epi8(a.as_ascii() as i8);
                let m = _mm256_cmpeq_epi8(letters, ascii);
                encoded = _mm256_blendv_epi8(encoded, index, m);
                unknown = _mm256_andnot_si256(m, unknown);
            }
            // Record is some symbols of the current vector are unknown.
            error = _mm256_or_si256(error, unknown);
            // Store the encoded result to the output buffer.
            _mm256_storeu_si256(dst_ptr as *mut __m256i, encoded);
            // Advance to the next addresses in input and output.
            src_ptr = src_ptr.add(std::mem::size_of::<__m256i>());
            dst_ptr = dst_ptr.add(std::mem::size_of::<__m256i>());
            i += std::mem::size_of::<__m256i>();
        }

        // If an invalid symbol was encountered, recover which one.
        // FIXME: run a vectorize the error search?
        if _mm256_testz_si256(error, error) != 1 {
            for i in 0..l {
                A::Symbol::from_ascii(seq[i])?;
            }
        }

        // Encode the rest of the sequence using the generic implementation.
        g.encode_into(&seq[i..], &mut dst[i..])?;
    }

    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(overflowing_literals)]
unsafe fn score_avx2_permute<A>(
    seq: &StripedSequence<A, <Avx2 as Backend>::LANES>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
) where
    A: Alphabet,
    <A as Alphabet>::K: IsLessOrEqual<U5>,
    <<A as Alphabet>::K as IsLessOrEqual<U5>>::Output: NonZero,
{
    let data = scores.matrix_mut();
    let mut rowptr = data[0].as_mut_ptr();
    // constant vector for comparing unknown bases
    let n = _mm256_set1_epi8(<A as Alphabet>::K::I8 - 1);
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
        // reset pointers to row
        let mut seqptr = seq.data[i].as_ptr();
        let mut pssmptr = pssm.weights()[0].as_ptr();
        // advance position in the position weight matrix
        for _ in 0..pssm.len() {
            // load sequence row and broadcast to f32
            let x = _mm256_load_si256(seqptr as *const __m256i);
            let x1 = _mm256_shuffle_epi8(x, m1);
            let x2 = _mm256_shuffle_epi8(x, m2);
            let x3 = _mm256_shuffle_epi8(x, m3);
            let x4 = _mm256_shuffle_epi8(x, m4);
            // load row for current weight matrix position
            let t = _mm256_broadcast_ps(&*(pssmptr as *const __m128));
            let u = _mm256_broadcast_ss(&*(pssmptr.add(<A as Alphabet>::K::USIZE - 1)));
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
            // advance to next row in PSSM and sequence matrices
            seqptr = seqptr.add(seq.data.stride());
            pssmptr = pssmptr.add(pssm.weights().stride());
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
unsafe fn score_avx2_gather<A>(
    seq: &StripedSequence<A, <Avx2 as Backend>::LANES>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
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
    for i in 0..seq.data.rows() - seq.wrap {
        // reset sums for current position
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut s4 = _mm256_setzero_ps();
        // reset pointers to row
        let mut seqptr = seq.data[i].as_ptr();
        let mut pssmptr = pssm.weights()[0].as_ptr();
        // advance position in the position weight matrix
        for _ in 0..pssm.len() {
            // load sequence row and broadcast to f32
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
            seqptr = seqptr.add(seq.data.stride());
            pssmptr = pssmptr.add(pssm.weights().stride());
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
unsafe fn argmax_avx2(scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
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
                // advance to next row
                dataptr = dataptr.add(data.stride());
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
                let pos = col * data.rows() + row as usize;
                let score = data[row as usize][col];
                if score > best_score || (score == best_score && pos < best_pos) {
                    best_score = data[row as usize][col];
                    best_pos = pos;
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
        let mut indices = vec![u32::MAX; data.columns() * rows];
        unsafe {
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
            let mut dataptr = data[0].as_ptr();
            for _ in 0..data.rows() {
                // load scores for the current row
                let r1 = _mm256_load_ps(dataptr.add(0x00));
                let r2 = _mm256_load_ps(dataptr.add(0x08));
                let r3 = _mm256_load_ps(dataptr.add(0x10));
                let r4 = _mm256_load_ps(dataptr.add(0x18));
                // check whether scores are greater or equal to the threshold
                let m1 = _mm256_castps_si256(_mm256_cmp_ps(r1, t, _CMP_GE_OS));
                let m2 = _mm256_castps_si256(_mm256_cmp_ps(r2, t, _CMP_GE_OS));
                let m3 = _mm256_castps_si256(_mm256_cmp_ps(r3, t, _CMP_GE_OS));
                let m4 = _mm256_castps_si256(_mm256_cmp_ps(r4, t, _CMP_GE_OS));
                // store masked indices into the destination vector
                _mm256_maskstore_epi32(dst as *mut _, m1, x1);
                _mm256_maskstore_epi32(dst.add(1) as *mut _, m2, x2);
                _mm256_maskstore_epi32(dst.add(2) as *mut _, m3, x3);
                _mm256_maskstore_epi32(dst.add(3) as *mut _, m4, x4);
                // advance result buffer to next row
                dst = dst.add(4);
                // advance sequence indices to next row
                x1 = _mm256_add_epi32(x1, ones);
                x2 = _mm256_add_epi32(x2, ones);
                x3 = _mm256_add_epi32(x3, ones);
                x4 = _mm256_add_epi32(x4, ones);
                // Advance data pointer to next row
                dataptr = dataptr.add(data.stride());
            }
        }

        // NOTE: Benchmarks suggest that `indices.retain(...)` is faster than
        //       `indices.into_iter().filter(...).

        // Remove all masked items and convert the indices to usize
        indices.retain(|&x| (x as usize) < scores.len());
        indices.into_iter().map(|i| i as usize).collect()
    }
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
            return encode_into_avx2::<A>(seq, dst);
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            panic!("attempting to run AVX2 code on a non-x86 host");
            unreachable!()
        }
    }

    #[allow(unused)]
    pub fn score_into_permute<A, S, M>(
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        A: Alphabet,
        <A as Alphabet>::K: IsLessOrEqual<U5>,
        <<A as Alphabet>::K as IsLessOrEqual<U5>>::Output: NonZero,
        S: AsRef<StripedSequence<A, <Avx2 as Backend>::LANES>>,
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

        if seq.length < pssm.len() {
            scores.resize(0, 0);
            return;
        }

        scores.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_avx2_permute(seq, pssm, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn score_into_gather<A, S, M>(
        seq: S,
        pssm: M,
        scores: &mut StripedScores<<Avx2 as Backend>::LANES>,
    ) where
        A: Alphabet,
        S: AsRef<StripedSequence<A, <Avx2 as Backend>::LANES>>,
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

        scores.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            score_avx2_gather(seq, pssm, scores)
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        panic!("attempting to run AVX2 code on a non-x86 host")
    }

    #[allow(unused)]
    pub fn argmax(scores: &StripedScores<<Avx2 as Backend>::LANES>) -> Option<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            argmax_avx2(scores)
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
