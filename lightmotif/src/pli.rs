#[cfg(target_feature = "ssse3")]
use std::arch::x86_64::*;

use self::seal::Vector;
use super::abc::Alphabet;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::pwm::WeightMatrix;
use super::seq::StripedSequence;

mod seal {
    pub trait Vector {}

    impl Vector for f32 {}

    #[cfg(target_feature = "avx2")]
    impl Vector for std::arch::x86_64::__m256 {}

    #[cfg(target_feature = "ssse3")]
    impl Vector for std::arch::x86_64::__m128 {}
}

pub struct Pipeline<A: Alphabet, V: Vector> {
    alphabet: std::marker::PhantomData<A>,
    vector: std::marker::PhantomData<V>,
}

impl<A: Alphabet, V: Vector> Pipeline<A, V> {
    pub fn new() -> Self {
        Self {
            alphabet: std::marker::PhantomData,
            vector: std::marker::PhantomData,
        }
    }
}

impl Pipeline<Dna, f32> {
    pub fn score_into<S, M, const C: usize>(
        &self,
        seq: S,
        pwm: M,
        scores: &mut StripedScores<f32, C>,
    ) where
        S: AsRef<StripedSequence<Dna, C>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let seq = seq.as_ref();
        let pwm = pwm.as_ref();

        let seq_rows = seq.data.rows() - seq.wrap;
        let result = &mut scores.data;
        if result.rows() < seq_rows {
            panic!("not enough rows for scores: {}", pwm.len());
        }

        for i in 0..seq.length - pwm.len() + 1 {
            let mut score = 0.0;
            for j in 0..pwm.len() {
                let offset = i + j;
                let col = offset / seq_rows;
                let row = offset % seq_rows;
                score += pwm.weights()[j][seq.data[row][col].as_index()];
            }
            let col = i / result.rows();
            let row = i % result.rows();
            result[row][col] = score;
        }
    }

    pub fn score<S, M, const C: usize>(&self, seq: S, pwm: M) -> StripedScores<f32, C>
    where
        S: AsRef<StripedSequence<Dna, C>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let mut scores = StripedScores::new_for(&seq, &pwm);
        self.score_into(seq, pwm, &mut scores);
        scores
    }
}

#[cfg(target_feature = "ssse3")]
impl Pipeline<Dna, __m128> {
    pub fn score_into<S, M>(
        &self,
        seq: S,
        pwm: M,
        scores: &mut StripedScores<__m128, { std::mem::size_of::<__m128i>() }>,
    ) where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m128i>() }>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let seq = seq.as_ref();
        let pwm = pwm.as_ref();
        let result = &mut scores.data;

        if seq.wrap < pwm.len() - 1 {
            panic!("not enough wrapping rows for motif of length {}", pwm.len());
        }
        if result.rows() < (seq.data.rows() - seq.wrap) {
            panic!("not enough rows for scores: {}", pwm.len());
        }

        scores.length = seq.length - pwm.len() + 1;
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
                for j in 0..pwm.len() {
                    // load sequence row and broadcast to f32
                    let x = _mm_load_si128(seq.data[i + j].as_ptr() as *const __m128i);
                    let x1 = _mm_shuffle_epi8(x, m1);
                    let x2 = _mm_shuffle_epi8(x, m2);
                    let x3 = _mm_shuffle_epi8(x, m3);
                    let x4 = _mm_shuffle_epi8(x, m4);
                    // load row for current weight matrix position
                    let row = pwm.weights()[j].as_ptr();
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

    pub fn score<S, M>(
        &self,
        seq: S,
        pwm: M,
    ) -> StripedScores<__m128, { std::mem::size_of::<__m128i>() }>
    where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m128i>() }>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let mut scores = StripedScores::new_for(&seq, &pwm);
        self.score_into(seq, pwm, &mut scores);
        scores
    }
}

#[cfg(target_feature = "avx2")]
impl Pipeline<Dna, __m256> {
    pub fn score_into<S, M>(
        &self,
        seq: S,
        pwm: M,
        scores: &mut StripedScores<__m256, { std::mem::size_of::<__m256i>() }>,
    ) where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m256i>() }>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let seq = seq.as_ref();
        let pwm = pwm.as_ref();
        let result = &mut scores.data;

        if seq.wrap < pwm.len() - 1 {
            panic!("not enough wrapping rows for motif of length {}", pwm.len());
        }
        if result.rows() < (seq.data.rows() - seq.wrap) {
            panic!("not enough rows for scores: {}", pwm.len());
        }

        scores.length = seq.length - pwm.len() + 1;
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
                for j in 0..pwm.len() {
                    // load sequence row and broadcast to f32
                    let x = _mm256_load_si256(seq.data[i + j].as_ptr() as *const __m256i);
                    let x1 = _mm256_shuffle_epi8(x, m1);
                    let x2 = _mm256_shuffle_epi8(x, m2);
                    let x3 = _mm256_shuffle_epi8(x, m3);
                    let x4 = _mm256_shuffle_epi8(x, m4);
                    // load row for current weight matrix position
                    let row = pwm.weights()[j].as_ptr();
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
                // record the score for the current position
                let row = &mut result[i];
                _mm256_store_ps(row[0..].as_mut_ptr(), s1);
                _mm256_store_ps(row[8..].as_mut_ptr(), s2);
                _mm256_store_ps(row[16..].as_mut_ptr(), s3);
                _mm256_store_ps(row[24..].as_mut_ptr(), s4);
            }
        }
    }

    pub fn score<S, M>(
        &self,
        seq: S,
        pwm: M,
    ) -> StripedScores<__m256, { std::mem::size_of::<__m256i>() }>
    where
        S: AsRef<StripedSequence<Dna, { std::mem::size_of::<__m256i>() }>>,
        M: AsRef<WeightMatrix<Dna, { Dna::K }>>,
    {
        let mut scores = StripedScores::new_for(&seq, &pwm);
        self.score_into(seq, pwm, &mut scores);
        scores
    }
}

#[derive(Clone, Debug)]
pub struct StripedScores<V: Vector, const C: usize = 32> {
    pub length: usize,
    pub data: DenseMatrix<f32, C>,
    marker: std::marker::PhantomData<V>,
}

impl<V: Vector, const C: usize> StripedScores<V, C> {
    /// Create a new striped score matrix with the given length and rows.
    pub fn new(length: usize, rows: usize) -> Self {
        Self {
            length,
            data: DenseMatrix::new(rows),
            marker: std::marker::PhantomData,
        }
    }

    /// Create a new matrix large enough to store the scores of `pwm` applied to `seq`.
    pub fn new_for<S, M, A, const K: usize>(seq: S, pwm: M) -> Self
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<WeightMatrix<A, K>>,
    {
        let seq = seq.as_ref();
        let pwm = pwm.as_ref();
        Self::new(seq.length - pwm.len() + 1, seq.data.rows() - seq.wrap)
    }

    pub fn resize(&mut self, length: usize, rows: usize) {
        self.length = length;
        self.data.resize(rows);
    }

    pub fn resize_for<S, M, A, const K: usize>(&mut self, seq: S, pwm: M)
    where
        A: Alphabet,
        S: AsRef<StripedSequence<A, C>>,
        M: AsRef<WeightMatrix<A, K>>,
    {
        let seq = seq.as_ref();
        let pwm = pwm.as_ref();
        self.resize(seq.length - pwm.len() + 1, seq.data.rows() - seq.wrap);
    }
}

impl<const C: usize> StripedScores<f32, C> {
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

    /// Get the index of the highest scoring position.
    ///
    /// ## Panic
    /// Panics if the data buffer is empty.
    pub fn argmax(&self) -> usize {
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
        best_pos
    }
}

#[cfg(target_feature = "ssse3")]
impl<const C: usize> StripedScores<__m128, C> {
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

    /// Get the index of the highest scoring position.
    ///
    /// ## Panic
    /// Panics if the data buffer is empty.
    pub fn argmax(&self) -> usize {
        let mut best_pos = 0;
        let mut best_score = self.data[0][0];
        let mut col = 0;
        let mut row = 0;
        for i in 0..self.length {
            if self.data[row][col] > best_score {
                best_pos = i;
                best_score = self.data[row][col];
            }
            row += 1;
            if row == self.data.rows() {
                row = 0;
                col += 1;
            }
        }
        best_pos
    }
}

#[cfg(target_feature = "avx2")]
impl<const C: usize> StripedScores<__m256, C> {
    /// Convert the striped scores to a vector of scores.
    pub fn to_vec(&self) -> Vec<f32> {
        // NOTE(@althonos): Because in AVX2 the __m256 vector is actually
        //                  two independent __m128, the shuffling creates
        //                  intrication in the results.
        #[rustfmt::skip]
        const COLS: &[usize] = &[
            0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
            4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
        ];

        let mut col = 0;
        let mut row = 0;
        let mut vec = Vec::with_capacity(self.length);
        while vec.len() < self.length {
            vec.push(self.data[row][COLS[col]]);
            row += 1;
            if row == self.data.rows() {
                row = 0;
                col += 1;
            }
        }
        vec
    }

    /// Get the index of the highest scoring position.
    ///
    /// ## Panic
    /// Panics if the data buffer is empty.
    pub fn argmax(&self) -> usize {
        let mut best_pos = 0;
        let mut best_score = self.data[0][0];
        let mut col = 0;
        let mut row = 0;
        for i in 0..self.length {
            if self.data[row][col] > best_score {
                best_pos = i;
                best_score = self.data[row][col];
            }
            row += 1;
            if row == self.data.rows() {
                row = 0;
                col += 1;
            }
        }
        best_pos
    }
}
