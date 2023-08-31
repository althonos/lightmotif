//! Arm 128-bit vector implementation, for 16 elements column width.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
use std::ops::Div;
use std::ops::Rem;

use typenum::consts::U16;
use typenum::marker_traits::Unsigned;
use typenum::marker_traits::Zero;

use super::Backend;
use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::err::InvalidSymbol;
use crate::num::StrictlyPositive;
use crate::pli::scores::StripedScores;
use crate::pli::Encode;
use crate::pli::Pipeline;

use crate::pwm::ScoringMatrix;
use crate::seq::StripedSequence;

/// A marker type for the SSE2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    type LANES = U16;
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
#[allow(overflowing_literals)]
unsafe fn encode_into_neon<A>(seq: &[u8], dst: &mut [A::Symbol]) -> Result<(), InvalidSymbol>
where
    A: Alphabet,
{
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
        let mut error = uint8x16x4_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));

        // Process the beginning of the sequence in SIMD while possible.
        while i + std::mem::size_of::<uint8x16_t>() * 4 < l {
            // Load current row and reset buffers for the encoded result.
            let letters = vld1q_u8_x4(src_ptr);
            let mut encoded = uint8x16x4_t(
                vdupq_n_u8(0x00),
                vdupq_n_u8(0x00),
                vdupq_n_u8(0x00),
                vdupq_n_u8(0x00),
            );
            let mut unknown = uint8x16x4_t(
                vdupq_n_u8(0xFF),
                vdupq_n_u8(0xFF),
                vdupq_n_u8(0xFF),
                vdupq_n_u8(0xFF),
            );
            // Check symbols one by one and match them to the letters.
            for a in 0..A::K::USIZE {
                let index = vdupq_n_u8(a as u8);
                let ascii = vdupq_n_u8(alphabet[a]);
                let m = uint8x16x4_t(
                    vceqq_u8(letters.0, ascii),
                    vceqq_u8(letters.1, ascii),
                    vceqq_u8(letters.2, ascii),
                    vceqq_u8(letters.3, ascii),
                );
                encoded.0 = vbslq_u8(m.0, index, encoded.0);
                unknown.0 = vandq_u8(unknown.0, vmvnq_u8(m.0));
                encoded.1 = vbslq_u8(m.1, index, encoded.1);
                unknown.1 = vandq_u8(unknown.1, vmvnq_u8(m.1));
                encoded.2 = vbslq_u8(m.2, index, encoded.2);
                unknown.2 = vandq_u8(unknown.2, vmvnq_u8(m.2));
                encoded.3 = vbslq_u8(m.3, index, encoded.3);
                unknown.3 = vandq_u8(unknown.3, vmvnq_u8(m.3));
            }
            // Record is some symbols of the current vector are unknown.
            error.0 = vorrq_u8(error.0, unknown.0);
            error.1 = vorrq_u8(error.1, unknown.1);
            error.2 = vorrq_u8(error.2, unknown.2);
            error.3 = vorrq_u8(error.3, unknown.3);
            // Store the encoded result to the output buffer.
            vst1q_u8_x4(dst_ptr as *mut u8, encoded);
            // Advance to the next addresses in input and output.
            src_ptr = src_ptr.add(std::mem::size_of::<uint8x16_t>() * 4);
            dst_ptr = dst_ptr.add(std::mem::size_of::<uint8x16_t>() * 4);
            i += std::mem::size_of::<uint8x16_t>() * 4;
        }

        // If an invalid symbol was encountered, recover which one.
        // FIXME: run a vectorize the error search?
        let error64 = vreinterpretq_u64_u8(vorrq_u8(
            vorrq_u8(error.0, error.1),
            vorrq_u8(error.2, error.3),
        ));
        if vgetq_lane_u64(error64, 0) != 0 || vgetq_lane_u64(error64, 1) != 0 {
            for i in 0..l {
                A::Symbol::from_ascii(seq[i])?;
            }
        }

        // Encode the rest of the sequence using the generic implementation.
        g.encode_into(&seq[i..], &mut dst[i..])?;
    }

    Ok(())
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn score_neon<A, C>(
    seq: &StripedSequence<A, C>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<C>,
) where
    A: Alphabet,
    C: StrictlyPositive + Rem<U16> + Div<U16>,
    <C as Rem<U16>>::Output: Zero,
    <C as Div<U16>>::Output: Unsigned,
{
    let zero_u8 = vdupq_n_u8(0);
    let zero_f32 = vdupq_n_f32(0.0);
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..<C as Div<U16>>::Output::USIZE)
        .into_iter()
        .map(|i| i * <Neon as Backend>::LANES::USIZE)
    {
        // process every position of the sequence data
        for i in 0..seq.data.rows() - seq.wrap {
            // reset sums for current position
            let mut s = float32x4x4_t(zero_f32, zero_f32, zero_f32, zero_f32);
            // reset position
            let mut dataptr = seq.data[i].as_ptr().add(offset);
            let mut pssmptr = pssm.weights()[0].as_ptr();
            // advance position in the position weight matrix
            for _ in 0..pssm.len() {
                // load sequence row and broadcast to f32
                let x = vld1q_u8(dataptr as *const u8);
                let z = vzipq_u8(x, zero_u8);
                let lo = vzipq_u8(z.0, zero_u8);
                let hi = vzipq_u8(z.1, zero_u8);
                let x1 = vreinterpretq_u32_u8(lo.0);
                let x2 = vreinterpretq_u32_u8(lo.1);
                let x3 = vreinterpretq_u32_u8(hi.0);
                let x4 = vreinterpretq_u32_u8(hi.1);
                // index lookup table with each bases incrementally
                for k in 0..A::K::USIZE {
                    let sym = vdupq_n_u32(k as u32);
                    let lut = vreinterpretq_u32_f32(vld1q_dup_f32(pssmptr.add(k)));
                    let p1 = vceqq_u32(x1, sym);
                    let p2 = vceqq_u32(x2, sym);
                    let p3 = vceqq_u32(x3, sym);
                    let p4 = vceqq_u32(x4, sym);
                    s.0 = vaddq_f32(s.0, vreinterpretq_f32_u32(vandq_u32(lut, p1)));
                    s.1 = vaddq_f32(s.1, vreinterpretq_f32_u32(vandq_u32(lut, p2)));
                    s.2 = vaddq_f32(s.2, vreinterpretq_f32_u32(vandq_u32(lut, p3)));
                    s.3 = vaddq_f32(s.3, vreinterpretq_f32_u32(vandq_u32(lut, p4)));
                }
                // advance to next row in sequence and PSSM matrices
                dataptr = dataptr.add(seq.data.stride());
                pssmptr = pssmptr.add(pssm.weights().stride());
            }
            // record the score for the current position
            let row = &mut data[i];
            vst1q_f32_x4(row[offset..].as_mut_ptr(), s);
        }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn threshold_neon<C>(scores: &StripedScores<C>, threshold: f32) -> Vec<usize>
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
            let max = vdupq_n_u32(u32::MAX);
            let t = vdupq_n_f32(threshold);
            let ones = vdupq_n_u32(1);
            let mut dst = indices.as_mut_ptr();
            for offset in (0..<C as Div<U16>>::Output::USIZE).map(|i| i * 16) {
                // prepare indices
                let mut v = [0u32; 16];
                for i in 0..16 {
                    v[i] = ((offset + i) * rows) as u32;
                }
                // compute real sequence index for each column of the striped scores
                let mut x = vld1q_u32_x4(v.as_slice().as_ptr());
                // Process rows iteratively
                let mut dataptr = data[0].as_ptr();
                for _ in 0..data.rows() {
                    // load scores for the current row
                    let r = vld1q_f32_x4(dataptr.add(offset));
                    // check whether scores are greater or equal to the threshold
                    let m = uint32x4x4_t(
                        vcltq_f32(t, r.0),
                        vcltq_f32(t, r.1),
                        vcltq_f32(t, r.2),
                        vcltq_f32(t, r.3),
                    );
                    // Mask indices that should be removed
                    let i = uint32x4x4_t(
                        vbslq_u32(m.0, x.0, max),
                        vbslq_u32(m.1, x.1, max),
                        vbslq_u32(m.2, x.2, max),
                        vbslq_u32(m.3, x.3, max),
                    );
                    // Store masked indices into the destination vector
                    vst1q_u32_x4(dst, i);
                    // Advance result buffer to next row
                    dst = dst.add(0x10);
                    // Advance sequence indices to next row
                    x.0 = vaddq_u32(x.0, ones);
                    x.1 = vaddq_u32(x.1, ones);
                    x.2 = vaddq_u32(x.2, ones);
                    x.3 = vaddq_u32(x.3, ones);
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

impl Neon {
    #[allow(unused)]
    pub fn encode_into<A>(seq: &[u8], dst: &mut [A::Symbol]) -> Result<(), InvalidSymbol>
    where
        A: Alphabet,
    {
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            return encode_into_neon::<A>(seq, dst);
        };
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        {
            panic!("attempting to run NEON code on a non-Arm host");
            unreachable!()
        }
    }

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
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            score_neon(seq, pssm, scores);
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }

    #[allow(unused)]
    pub fn threshold<C>(scores: &StripedScores<C>, threshold: f32) -> Vec<usize>
    where
        C: StrictlyPositive + Rem<U16> + Div<U16>,
        <C as Rem<U16>>::Output: Zero,
        <C as Div<U16>>::Output: Unsigned,
    {
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            threshold_neon(scores, threshold)
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }
}
