//! Arm 128-bit vector implementation, for 16 elements column width.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Range;
use std::ops::Rem;

use super::Backend;
use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::err::InvalidSymbol;
use crate::num::consts::U16;
use crate::num::MultipleOf;
use crate::num::StrictlyPositive;
use crate::num::Unsigned;
use crate::num::Zero;
use crate::pli::Encode;
use crate::pli::Pipeline;
use crate::scores::StripedScores;

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
unsafe fn score_neon<A: Alphabet, C: MultipleOf<U16>>(
    pssm: &ScoringMatrix<A>,
    seq: &StripedSequence<A, C>,
    rows: Range<usize>,
    scores: &mut StripedScores<C>,
) {
    let zero_u8 = vdupq_n_u8(0);
    let zero_f32 = vdupq_n_f32(0.0);
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..C::Quotient::USIZE).map(|i| i * <Neon as Backend>::LANES::USIZE) {
        // process every position of the sequence data
        for i in rows.clone() {
            // reset sums for current position
            let mut s = float32x4x4_t(zero_f32, zero_f32, zero_f32, zero_f32);
            // reset position
            let mut dataptr = seq.matrix()[i].as_ptr().add(offset);
            let mut pssmptr = pssm.matrix()[0].as_ptr();
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
                dataptr = dataptr.add(seq.matrix().stride());
                pssmptr = pssmptr.add(pssm.matrix().stride());
            }
            // record the score for the current position
            let row = &mut data[i];
            vst1q_f32_x4(row[offset..].as_mut_ptr(), s);
        }
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
    pub fn score_rows_into<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<C>,
    ) where
        A: Alphabet,
        C: MultipleOf<U16>,
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
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            score_neon(pssm, seq, rows, scores);
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }
}
