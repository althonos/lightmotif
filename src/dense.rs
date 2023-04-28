use std::ops::Index;
use std::ops::IndexMut;

/// An aligned dense matrix of with a constant number of columns.
#[derive(Debug, Clone)]
pub struct DenseMatrix<T: Default, const C: usize = 32> {
    data: Vec<T>,
    indices: Vec<usize>,
}

impl<T: Default, const C: usize> DenseMatrix<T, C> {
    /// Create a new matrix
    pub fn new(rows: usize) -> Self {
        // allocate data block
        let mut data = Vec::with_capacity(rows * C + C - 1); // FIXME
        for _ in 0..data.capacity() {
            data.push(T::default());
        }
        // compute offset to have proper alignment
        let offset = unsafe { data.align_to::<[u8; C]>().0.len() };
        // record row coordinates (only really useful when aligned)
        let mut indices = Vec::with_capacity(rows);
        for i in 0..rows {
            indices.push(offset + i * C)
        }
        // finish matrix
        Self { data, indices }
    }

    /// The number of columns of the matrix.
    pub const fn columns(&self) -> usize {
        C
    }

    /// The number of rows of the matrix.
    pub fn rows(&self) -> usize {
        self.indices.len()
    }
}

impl<T: Default, const C: usize> Index<usize> for DenseMatrix<T, C> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        let row = self.indices[index];
        &self.data[row..row + C]
    }
}

impl<T: Default, const C: usize> IndexMut<usize> for DenseMatrix<T, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = self.indices[index];
        &mut self.data[row..row + C]
    }
}
