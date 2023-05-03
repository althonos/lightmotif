#[cfg(any(target_feature = "ssse3", doc))]
use std::arch::x86_64::*;

use std::ops::Index;

use self::seal::Vector;
use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::pwm::ScoringMatrix;
use super::seq::StripedSequence;

// --- Vector ------------------------------------------------------------------

mod seal {
    /// Sealed trait for concrete vector implementations.
    pub trait Vector {}

    impl Vector for f32 {}

    #[cfg(target_feature = "avx2")]
    impl Vector for std::arch::x86_64::__m256 {}

    #[cfg(target_feature = "ssse3")]
    impl Vector for std::arch::x86_64::__m128 {}
}

// --- Score -------------------------------------------------------------------

/// Generic trait for computing sequence scores with a PSSM.
pub trait Score<A: Alphabet, const K: usize, V: Vector, const C: usize> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A, K>>;

    fn score<S, M>(seq: S, pssm: M) -> StripedScores<C>
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A, K>>,
    {
        let mut scores = StripedScores::new_for(&seq, &pssm);
        Self::score_into(seq, pssm, &mut scores);
        scores
    }
}

// --- Pipeline ----------------------------------------------------------------

/// Wrapper implementing score computation for different platforms.
#[derive(Debug, Default, Clone)]
pub struct Pipeline<A: Alphabet, V: Vector> {
    alphabet: std::marker::PhantomData<A>,
    vector: std::marker::PhantomData<V>,
}

/// Scalar scoring implementation, for any column width.
impl<A: Alphabet, const K: usize, const C: usize> Score<A, K, f32, C> for Pipeline<A, f32> {
    fn score_into<S, M>(seq: S, pssm: M, scores: &mut StripedScores<C>)
    where
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A, K>>,
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

/// Intel 128-bit vector implementation, for 16 elements column width.
#[cfg(any(target_feature = "ssse3", doc))]
impl Score<Dna, { Dna::K }, __m128, { std::mem::size_of::<__m128i>() }> for Pipeline<Dna, __m128> {
    fn score_into<S, M>(
        seq: S,
        pssm: M,
        scores: &mut StripedScores<{ std::mem::size_of::<__m128i>() }>,
    ) where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m128i>() }>>,
        M: AsRef<ScoringMatrix<Dna, { Dna::K }>>,
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
            //
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
                    for i in 0..Dna::K {
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
                let row = &mut result[i];
                _mm_store_ps(row[0..].as_mut_ptr(), s1);
                _mm_store_ps(row[4..].as_mut_ptr(), s2);
                _mm_store_ps(row[8..].as_mut_ptr(), s3);
                _mm_store_ps(row[12..].as_mut_ptr(), s4);
            }
        }
    }
}

#[cfg(any(target_feature = "avx2", doc))]
impl Score<Dna, { Dna::K }, __m256, { std::mem::size_of::<__m256i>() }> for Pipeline<Dna, __m256> {
    fn score_into<S, M>(
        seq: S,
        pssm: M,
        scores: &mut StripedScores<{ std::mem::size_of::<__m256i>() }>,
    ) where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m256i>() }>>,
        M: AsRef<ScoringMatrix<Dna, { Dna::K }>>,
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
                let row = &mut result[i];
                _mm256_store_ps(row[0x00..].as_mut_ptr(), r1);
                _mm256_store_ps(row[0x08..].as_mut_ptr(), r2);
                _mm256_store_ps(row[0x10..].as_mut_ptr(), r3);
                _mm256_store_ps(row[0x18..].as_mut_ptr(), r4);
            }
        }
    }
}

// --- StripedScores -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StripedScores<const C: usize = 32> {
    data: DenseMatrix<f32, C>,
    length: usize,
}

impl<const C: usize> StripedScores<C> {
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
    pub fn new_for<S, M, A, const K: usize>(seq: S, pssm: M) -> Self
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A, K>>,
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
    pub fn resize_for<S, M, A, const K: usize>(&mut self, seq: S, pssm: M)
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<ScoringMatrix<A, K>>,
    {
        let seq = seq.as_ref();
        let pssm = pssm.as_ref();
        self.resize(seq.length - pssm.len() + 1, seq.data.rows() - seq.wrap);
    }

    /// Convert the striped scores to a vector of scores.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.length);
        for i in 0..self.length {
            let col = i / self.data.rows();
            let row = i % self.data.rows();
            vec.push(self.data[row][col]);
        }
        vec
    }

    /// Get the index of the highest scoring position, if any.
    pub fn argmax(&self) -> Option<usize> {
        if self.len() > 0 {
            let mut best_pos = 0;
            let mut best_score = self.data[0][0];
            for i in 0..self.length {
                let col = i / self.data.rows();
                let row = i % self.data.rows();
                if self.data[row][col] > best_score {
                    best_pos = i;
                    best_score = self.data[row][col];
                }
            }
            Some(best_pos)
        } else {
            None
        }
    }
}

impl<const C: usize> AsRef<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_ref(&self) -> &DenseMatrix<f32, C> {
        self.matrix()
    }
}

impl<const C: usize> AsMut<DenseMatrix<f32, C>> for StripedScores<C> {
    fn as_mut(&mut self) -> &mut DenseMatrix<f32, C> {
        self.matrix_mut()
    }
}

impl<const C: usize> Index<usize> for StripedScores<C> {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        let col = index / self.data.rows();
        let row = index % self.data.rows();
        &self.data[row][col]
    }
}

impl<const C: usize> From<StripedScores<C>> for Vec<f32> {
    fn from(scores: StripedScores<C>) -> Self {
        let rows = scores.data.rows();
        let mut vec = Vec::with_capacity(scores.length);
        for i in 0..scores.length {
            let col = i / rows;
            let row = i % rows;
            vec.push(scores.data[row][col]);
        }
        vec
    }
}
