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
use crate::num::StrictlyPositive;
use crate::pli::scores::StripedScores;
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
            // advance position in the position weight matrix
            for j in 0..pssm.len() {
                // load sequence row and broadcast to f32
                let x = vld1q_u8(seq.data[i + j].as_ptr().add(offset) as *const u8);
                let z = vzipq_u8(x, zero_u8);
                let lo = vzipq_u8(z.0, zero_u8);
                let hi = vzipq_u8(z.1, zero_u8);
                let x1 = vreinterpretq_u32_u8(lo.0);
                let x2 = vreinterpretq_u32_u8(lo.1);
                let x3 = vreinterpretq_u32_u8(hi.0);
                let x4 = vreinterpretq_u32_u8(hi.1);
                // load row for current weight matrix position
                let row = pssm.weights()[j].as_ptr();
                // index lookup table with each bases incrementally
                for k in 0..A::K::USIZE {
                    let sym = vdupq_n_u32(k as u32);
                    let lut = vreinterpretq_u32_f32(vld1q_dup_f32(row.add(k)));
                    let p1 = vceqq_u32(x1, sym);
                    let p2 = vceqq_u32(x2, sym);
                    let p3 = vceqq_u32(x3, sym);
                    let p4 = vceqq_u32(x4, sym);
                    s.0 = vaddq_f32(s.0, vreinterpretq_f32_u32(vandq_u32(lut, p1)));
                    s.1 = vaddq_f32(s.1, vreinterpretq_f32_u32(vandq_u32(lut, p2)));
                    s.2 = vaddq_f32(s.2, vreinterpretq_f32_u32(vandq_u32(lut, p3)));
                    s.3 = vaddq_f32(s.3, vreinterpretq_f32_u32(vandq_u32(lut, p4)));
                }
            }
            // record the score for the current position
            let row = &mut data[i];
            vst1q_f32_x4(row[offset..].as_mut_ptr(), s);
        }
    }
}

impl Neon {
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

        scores.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        unsafe {
            score_neon(seq, pssm, scores);
        }
        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        panic!("attempting to run NEON code on a non-Arm host")
    }
}
