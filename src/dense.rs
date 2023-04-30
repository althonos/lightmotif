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
        // alway over-allocate columns to avoid alignment issues.
        let c = C.next_power_of_two();
        // allocate data block
        let mut data = Vec::with_capacity( (rows+1) * c ); // FIXME
        data.resize_with(data.capacity(), T::default);
        // compute offset to have proper alignment
        let mut offset = 0;
        while data[offset..].as_ptr() as usize % c > 0 {
            offset += 1
        }
        // record row coordinates (only really useful when aligned)
        let mut indices = Vec::with_capacity(rows);
        for i in 0..rows {
            indices.push(offset + i * c)
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
