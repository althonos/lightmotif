//! Arm 128-bit vector implementation, for 16 elements column width.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Range;
use std::ops::Rem;

use generic_array::ArrayLength;

use super::Backend;
use crate::abc::Alphabet;
use crate::abc::Symbol;
use crate::dense::DenseMatrix;
use crate::err::InvalidSymbol;
use crate::num::IsLessOrEqual;
use crate::num::MultipleOf;
use crate::num::NonZero;
use crate::num::StrictlyPositive;
use crate::num::Unsigned;
use crate::num::Zero;
use crate::num::U16;
use crate::pli::Encode;
use crate::pli::Pipeline;
use crate::pwm::ScoringMatrix;
use crate::scores::StripedScores;
use crate::seq::StripedSequence;

/// A marker type for the SSE2 implementation of the pipeline.
#[derive(Clone, Debug, Default)]
pub struct Neon;

impl Backend for Neon {
    type Lanes = U16;
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
unsafe fn score_f32_neon_vandq<A: Alphabet, C: MultipleOf<U16> + ArrayLength>(
    pssm: &DenseMatrix<f32, A::K>,
    seq: &StripedSequence<A, C>,
    rows: Range<usize>,
    scores: &mut StripedScores<f32, C>,
) {
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..C::Quotient::USIZE).map(|i| i * <Neon as Backend>::Lanes::USIZE) {
        let psmptr = pssm[0].as_ptr();
        let mut rowptr = data[0].as_mut_ptr().add(offset);
        let mut seqptr = seq.matrix()[rows.start].as_ptr().add(offset);

        // process every position of the sequence data
        for _ in 0..rows.len() {
            // reset sums for current position
            let mut s = float32x4x4_t(
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            // reset position
            let mut seqrow = seqptr;
            let mut psmrow = psmptr;
            // advance position in the position weight matrix
            for _ in 0..pssm.rows() {
                // load sequence row
                let x = vld1q_u8(seqrow as *const u8);
                let z = vzipq_u8(x, vdupq_n_u8(0));
                // transform u8 into u32
                let lo = vzipq_u8(z.0, vdupq_n_u8(0));
                let hi = vzipq_u8(z.1, vdupq_n_u8(0));
                let x1 = vreinterpretq_u32_u8(lo.0);
                let x2 = vreinterpretq_u32_u8(lo.1);
                let x3 = vreinterpretq_u32_u8(hi.0);
                let x4 = vreinterpretq_u32_u8(hi.1);
                // index lookup table with each bases incrementally
                for k in 0..A::K::USIZE {
                    let sym = vdupq_n_u32(k as u32);
                    let lut = vreinterpretq_u32_f32(vld1q_dup_f32(psmrow.add(k)));
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
                seqrow = seqrow.add(seq.matrix().stride());
                psmrow = psmrow.add(pssm.stride());
            }
            // record the score for the current position
            vst1q_f32_x4(rowptr, s);
            rowptr = rowptr.add(data.stride());
            seqptr = seqptr.add(seq.matrix().stride());
        }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn score_f32_neon_vqtbl1q<A: Alphabet, C: MultipleOf<U16> + ArrayLength>(
    pssm: &DenseMatrix<f32, A::K>,
    seq: &StripedSequence<A, C>,
    rows: Range<usize>,
    scores: &mut StripedScores<f32, C>,
) {
    #[inline]
    unsafe fn _vuntrans_f32(
        v0: uint8x16_t,
        v1: uint8x16_t,
        v2: uint8x16_t,
        v3: uint8x16_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let b01: uint8x16x2_t = vzipq_u8(v0, v1);
        let b23: uint8x16x2_t = vzipq_u8(v2, v3);
        let f01: uint16x8x2_t = vzipq_u16(vreinterpretq_u16_u8(b01.0), vreinterpretq_u16_u8(b23.0));
        let f23: uint16x8x2_t = vzipq_u16(vreinterpretq_u16_u8(b01.1), vreinterpretq_u16_u8(b23.1));
        (
            vreinterpretq_f32_u16(f01.0),
            vreinterpretq_f32_u16(f01.1),
            vreinterpretq_f32_u16(f23.0),
            vreinterpretq_f32_u16(f23.1),
        )
    }

    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..C::Quotient::USIZE).map(|i| i * <Neon as Backend>::Lanes::USIZE) {
        let psmptr: *const f32 = pssm[0].as_ptr();
        let mut rowptr: *mut f32 = data[0].as_mut_ptr().add(offset);
        let mut seqptr: *const A::Symbol = seq.matrix()[rows.start].as_ptr().add(offset);

        // process every position of the sequence data
        for _ in 0..rows.len() {
            // reset sums for current position
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);
            // reset position
            let mut seqrow = seqptr;
            let mut psmrow = psmptr;
            // advance position in the position weight matrix
            for _ in 0..pssm.rows() {
                // load sequence row
                let x: uint8x16_t = vld1q_u8(seqrow as *const u8);
                // load pssm row with de-interleaving
                let pt = vld4q_u8(psmrow as *const u8);
                // let pt: uint8x16x4_t = _vtrans_f32(p);
                // index each byte
                let b0: uint8x16_t;
                let b1: uint8x16_t;
                let b2: uint8x16_t;
                let b3: uint8x16_t;
                // ARMv8 can perform indexing of uint8x16x4_t by uint8x16_t
                // using the vqtbl1q_u8 instruction, so it's easy
                #[cfg(target_arch = "aarch64")]
                {
                    b0 = vqtbl1q_u8(pt.0, x);
                    b1 = vqtbl1q_u8(pt.1, x);
                    b2 = vqtbl1q_u8(pt.2, x);
                    b3 = vqtbl1q_u8(pt.3, x);
                }
                // ARMv7 only support indexing uint8x8_t by uint8x8_t;
                // we know that we have 8 elements in the LUT (since K <= 8)
                // but we still need to handle both halves of `x`
                #[cfg(target_arch = "arm")]
                {
                    // index LUT with first 8 bytes
                    let l0 = vtbl1_u8(vget_low_u8(pt.0), vget_low_u8(x));
                    let l1 = vtbl1_u8(vget_low_u8(pt.1), vget_low_u8(x));
                    let l2 = vtbl1_u8(vget_low_u8(pt.2), vget_low_u8(x));
                    let l3 = vtbl1_u8(vget_low_u8(pt.3), vget_low_u8(x));
                    // index LUT with next 8 bytes
                    let h0 = vtbl1_u8(vget_low_u8(pt.0), vget_high_u8(x));
                    let h1 = vtbl1_u8(vget_low_u8(pt.1), vget_high_u8(x));
                    let h2 = vtbl1_u8(vget_low_u8(pt.2), vget_high_u8(x));
                    let h3 = vtbl1_u8(vget_low_u8(pt.3), vget_high_u8(x));
                    // combine the results
                    b0 = vcombine_u8(l0, h0);
                    b1 = vcombine_u8(l1, h1);
                    b2 = vcombine_u8(l2, h2);
                    b3 = vcombine_u8(l3, h3);
                }
                // revert the look-up result into a proper float32x4x4_t
                let (xs0, xs1, xs2, xs3) = _vuntrans_f32(b0, b1, b2, b3);
                s0 = vaddq_f32(s0, xs0);
                s1 = vaddq_f32(s1, xs1);
                s2 = vaddq_f32(s2, xs2);
                s3 = vaddq_f32(s3, xs3);
                // advance to next row in sequence and PSSM matrices
                seqrow = seqrow.add(seq.matrix().stride());
                psmrow = psmrow.add(pssm.stride());
            }
            // record the score for the current position
            vst1q_f32(rowptr, s0);
            vst1q_f32(rowptr.add(0x04), s1);
            vst1q_f32(rowptr.add(0x08), s2);
            vst1q_f32(rowptr.add(0x0c), s3);
            rowptr = rowptr.add(data.stride());
            seqptr = seqptr.add(seq.matrix().stride());
        }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn score_u8_neon<A: Alphabet, C: MultipleOf<U16> + ArrayLength>(
    pssm: &DenseMatrix<u8, A::K>,
    seq: &StripedSequence<A, C>,
    rows: Range<usize>,
    scores: &mut StripedScores<u8, C>,
) {
    // process columns of the striped matrix, any multiple of 16 is supported
    let data = scores.matrix_mut();
    for offset in (0..C::Quotient::USIZE).map(|i| i * <Neon as Backend>::Lanes::USIZE) {
        let mut rowptr = data[0].as_mut_ptr().add(offset);
        // process every position of the sequence data
        for i in rows.clone() {
            // reset sums for current position
            let mut s = vdupq_n_u8(0);
            // reset position
            let mut seqptr = seq.matrix()[i].as_ptr().add(offset);
            let mut pssmptr = pssm[0].as_ptr();
            // advance position in the position weight matrix
            for _ in 0..pssm.rows() {
                // load sequence row
                let x = vld1q_u8(seqptr as *const u8);
                // load pssm row
                let t = vld1q_u8(pssmptr as *const u8);
                // shuffle pssm with the sequence characters
                let y: uint8x16_t;
                // ARMv8 can perform indexing of uint8x16_t by uint8x16_t
                // using the vqtbl1q_u8 instruction, so it's easy
                #[cfg(target_arch = "aarch64")]
                {
                    y = vqtbl1q_u8(t, x);
                }
                // ARMv7 only support indexing uint8x8_t by uint8x8_t;
                // we know that we have 8 elements in the LUT (since K <= 8)
                // but we still need to handle both halves of `x`
                #[cfg(target_arch = "arm")]
                {
                    // index LUT with first 8 bytes
                    let lo = vtbl1_u8(vget_low_u8(t), vget_low_u8(x));
                    let hi = vtbl1_u8(vget_low_u8(t), vget_high_u8(x));
                    y = vcombine_u8(lo, hi);
                }
                // add scores to the running sum
                s = vaddq_u8(s, y);
                // advance to next row in PSSM and sequence matrices
                seqptr = seqptr.add(seq.matrix().stride());
                pssmptr = pssmptr.add(pssm.stride());
            }
            // record the score for the current position
            vst1q_u8(rowptr, s);
            rowptr = rowptr.add(data.stride());
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
    pub fn score_f32_rows_into_vandq<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, C>,
    ) where
        A: Alphabet,
        C: MultipleOf<U16> + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
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

        if seq.len() < pssm.rows() || rows.is_empty() {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            score_f32_neon_vandq(pssm, seq, rows, scores);
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }

    #[allow(unused)]
    fn score_f32_rows_into_vtblq<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, C>,
    ) where
        A: Alphabet,
        C: MultipleOf<U16> + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<f32, A::K>>,
    {
        assert!(A::K::USIZE <= 8);

        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.rows() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.rows()
            );
        }

        if seq.len() < pssm.rows() || rows.is_empty() {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(target_arch = "aarch64")]
        unsafe {
            score_f32_neon_vqtbl1q(pssm, seq, rows, scores)
        };
        #[cfg(not(target_arch = "aarch64"))]
        panic!("attempting to run ARMv8 NEON code on a non-ARMv8 host")
    }

    #[allow(unused)]
    pub fn score_f32_rows_into<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<f32, C>,
    ) where
        A: Alphabet,
        C: MultipleOf<U16> + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<f32, A::K>>,
    {
        if A::K::USIZE <= 8 {
            Self::score_f32_rows_into_vtblq(pssm, seq, rows, scores)
        } else {
            Self::score_f32_rows_into_vandq(pssm, seq, rows, scores)
        }
    }

    #[allow(unused)]
    pub fn score_u8_rows_into<A, C, S, M>(
        pssm: M,
        seq: S,
        rows: Range<usize>,
        scores: &mut StripedScores<u8, C>,
    ) where
        <A as Alphabet>::K: IsLessOrEqual<U16>,
        <<A as Alphabet>::K as IsLessOrEqual<U16>>::Output: NonZero,
        A: Alphabet,
        C: MultipleOf<U16> + ArrayLength,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<DenseMatrix<u8, A::K>>,
    {
        // vqtbl1q_u8 limits to K<=16 since the lookup-table is a uint8x16_t,
        // with max 16 elements, but this could be expanded to K<=32 with
        // vqtbl2q_u8 to support protein sequences...
        assert!(A::K::USIZE <= 16);

        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        if seq.wrap() < pssm.rows() - 1 {
            panic!(
                "not enough wrapping rows for motif of length {}",
                pssm.rows()
            );
        }

        if seq.len() < pssm.rows() || rows.is_empty() {
            scores.resize(0, 0);
            return;
        }

        scores.resize(rows.len(), (seq.len() + 1).saturating_sub(pssm.rows()));
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            score_u8_neon(pssm, seq, rows, scores);
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }
}
