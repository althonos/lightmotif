#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::Range;

use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

pub use self::vector::Vector;

use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;

// --- Vector ------------------------------------------------------------------

mod vector {
    use typenum::consts::U1;
    use typenum::consts::U16;
    use typenum::consts::U32;
    use typenum::marker_traits::NonZero;
    use typenum::marker_traits::Unsigned;

    mod seal {
        pub trait Sealed {}

        impl Sealed for u8 {}

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl Sealed for std::arch::x86_64::__m128i {}

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl Sealed for std::arch::x86_64::__m256i {}
    }

    /// Sealed trait for concrete vector implementations.
    ///
    /// The trait is defined for the loading vector type, which has `LANES`
    /// lanes of `u8` values. These values are then splat into 4 vectors with
    /// `f32` values to actually compute the scores.
    pub trait Vector: seal::Sealed {
        type LANES: Unsigned + NonZero;
    }

    impl Vector for u8 {
        type LANES = U1;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    impl Vector for std::arch::x86_64::__m128i {
        type LANES = U16;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    impl Vector for std::arch::x86_64::__m256i {
        type LANES = U32;
    }

    #[cfg(target_feature = "avx2")]
    pub type Best = std::arch::x86_64::__m256i;
    #[cfg(all(not(target_feature = "avx2"), target_feature = "ssse3"))]
    pub type Best = std::arch::x86_64::__m128i;
    #[cfg(all(not(target_feature = "avx2"), not(target_feature = "ssse3")))]
    pub type Best = u8;
}

// --- Score -------------------------------------------------------------------

/// Generic trait for computing sequence scores with a PSSM.
pub trait Score<A: Alphabet, V: Vector, C: NonZero + Unsigned = <V as Vector>::LANES> {
    /// Compute the PSSM scores into the given buffer.
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>;

    /// Compute the PSSM scores for every sequence positions.
    fn score<S, M>(seq: S, pssm: M) -> StripedScores<C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let mut scores = StripedScores::new_for(&seq, &pssm);
        Self::score_into(seq, pssm, &mut scores);
        scores
    }

    /// Find the sequence position with the highest score.
    fn best_position(scores: &StripedScores<C>) -> Option<usize> {
        if scores.length == 0 {
            return None;
        }

        let mut best_pos = 0;
        let mut best_score = scores.data[0][0];
        for i in 0..scores.length {
            let col = i / scores.data.rows();
            let row = i % scores.data.rows();
            if scores.data[row][col] > best_score {
                best_score = scores.data[row][col];
                best_pos = i;
            }
        }

        Some(best_pos)
    }
}

// --- Pipeline ----------------------------------------------------------------

/// Wrapper implementing score computation for different platforms.
#[derive(Debug, Default, Clone)]
pub struct Pipeline<A: Alphabet, V: Vector = vector::Best> {
    alphabet: std::marker::PhantomData<A>,
    vector: std::marker::PhantomData<V>,
}

/// Scalar scoring implementation.
impl<A: Alphabet, C: NonZero + Unsigned> Score<A, u8, C> for Pipeline<A, u8> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();

        let seq_rows = seq.data.rows() - seq.wrap;
        let result = &mut scores.data;
        if result.rows() < seq_rows {
            panic!("not enough rows for scores: {}", pssm.len());
        }

        for i in 0..seq.length - pssm.len() + 1 {
            let mut score = 0.0;
            for j in 0..pssm.len() {
                let offset = i + j;
                let col = offset / seq_rows;
                let row = offset % seq_rows;
                score += pssm.weights()[j][seq.data[row][col].as_index()];
            }
            let col = i / result.rows();
            let row = i % result.rows();
            result[row][col] = score;
        }
    }
}

// --- SSSE3 -------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn score_ssse3<A: Alphabet>(
    seq: &StripedSequence<A, <__m128i as Vector>::LANES>,
    pssm: &ScoringMatrix<A>,
    scores: &mut StripedScores<<__m128i as Vector>::LANES>,
) {
    // mask vectors for broadcasting uint8x16_t to uint32x4_t to floatx4_t
    let m1 = _mm_set_epi32(
        0xFFFFFF03u32 as i32,
        0xFFFFFF02u32 as i32,
        0xFFFFFF01u32 as i32,
        0xFFFFFF00u32 as i32,
    );
    let m2 = _mm_set_epi32(
        0xFFFFFF07u32 as i32,
        0xFFFFFF06u32 as i32,
        0xFFFFFF05u32 as i32,
        0xFFFFFF04u32 as i32,
    );
    let m3 = _mm_set_epi32(
        0xFFFFFF0Bu32 as i32,
        0xFFFFFF0Au32 as i32,
        0xFFFFFF09u32 as i32,
        0xFFFFFF08u32 as i32,
    );
    let m4 = _mm_set_epi32(
        0xFFFFFF0Fu32 as i32,
        0xFFFFFF0Eu32 as i32,
        0xFFFFFF0Du32 as i32,
        0xFFFFFF0Cu32 as i32,
    );
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
            let x1 = _mm_shuffle_epi8(x, m1);
            let x2 = _mm_shuffle_epi8(x, m2);
            let x3 = _mm_shuffle_epi8(x, m3);
            let x4 = _mm_shuffle_epi8(x, m4);
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn best_position_ssse3(scores: &StripedScores<<__m128i as Vector>::LANES>) -> Option<usize> {
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
            score_ssse3(seq, pssm, scores);
        }
    }

    fn best_position(scores: &StripedScores<<__m128i as Vector>::LANES>) -> Option<usize> {
        unsafe { best_position_ssse3(scores) }
    }
}

// --- AVX2 --------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn score_avx2(
    seq: &StripedSequence<Dna, <__m256i as Vector>::LANES>,
    pssm: &ScoringMatrix<Dna>,
    scores: &mut StripedScores<<__m256i as Vector>::LANES>,
) {
    // constant vector for comparing unknown bases
    let n = _mm256_set1_epi8(super::Nucleotide::N as i8);
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

// --- StripedScores -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StripedScores<C: Unsigned + NonZero> {
    data: DenseMatrix<f32, C>,
    length: usize,
}

impl<C: Unsigned + NonZero> StripedScores<C> {
    /// Create a new striped score matrix with the given length and rows.
    pub fn new(length: usize, rows: usize) -> Self {
        Self {
            length,
            data: DenseMatrix::new(rows),
        }
    }

    /// Return the number of scored positions.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return a reference to the striped matrix storing the scores.
    pub fn matrix(&self) -> &DenseMatrix<f32, C> {
        &self.data
    }

    /// Return a mutable reference to the striped matrix storing the scores.
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        &mut self.data
    }

    /// Create a new matrix large enough to store the scores of `pssm` applied to `seq`.
    pub fn new_for<S, M, A>(seq: S, pssm: M) -> Self
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        Self::new(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap)
    }

    /// Resize the striped scores storage to the given length and number of rows.
    pub fn resize(&mut self, length: usize, rows: usize) {
        self.length = length;
        self.data.resize(rows);
    }

    /// Resize the striped scores storage to store the scores of `pssm` applied to `seq`.
    pub fn resize_for<S, M, A>(&mut self, seq: S, pssm: M)
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        self.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
    }

    /// Iterate over scores of individual sequence positions.
    pub fn iter(&self) -> Iter<'_, C> {
        Iter::new(&self)
    }

    /// Convert the striped scores to a vector of scores.
    pub fn to_vec(&self) -> Vec<f32> {
        self.iter().cloned().collect()
    }
}

impl<C: Unsigned + NonZero> AsRef<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_ref(&self) -> &DenseMatrix<f32, C> {
        self.matrix()
    }
}

impl<C: Unsigned + NonZero> AsMut<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        self.matrix_mut()
    }
}

impl<C: Unsigned + NonZero> Index<usize> for StripedScores<C> {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        let col = index / self.data.rows();
        let row = index % self.data.rows();
        &self.data[row][col]
    }
}

impl<C: Unsigned + NonZero> From<StripedScores<C>> for Vec<f32> {
    fn from(scores: StripedScores<C>) -> Self {
        scores.iter().cloned().collect()
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, C: Unsigned + NonZero> {
    scores: &'a StripedScores<C>,
    indices: Range<usize>,
}

impl<'a, C: Unsigned + NonZero> Iter<'a, C> {
    fn new(scores: &'a StripedScores<C>) -> Self {
        Self {
            scores,
            indices: 0..scores.length,
        }
    }

    fn get(&self, i: usize) -> &'a f32 {
        let col = i / self.scores.data.rows();
        let row = i % self.scores.data.rows();
        &self.scores.data[row][col]
    }
}

impl<'a, C: Unsigned + NonZero> Iterator for Iter<'a, C> {
    type Item = &'a f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|i| self.get(i))
    }
}

impl<'a, C: Unsigned + NonZero> ExactSizeIterator for Iter<'a, C> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a, C: Unsigned + NonZero> FusedIterator for Iter<'a, C> {}

impl<'a, C: Unsigned + NonZero> DoubleEndedIterator for Iter<'a, C> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.indices.next_back().map(|i| self.get(i))
    }
}
