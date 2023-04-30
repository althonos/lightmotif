use std::ops::Index;
use std::ops::IndexMut;

/// An aligned dense matrix of with a constant number of columns.
#[derive(Debug, Clone)]
pub struct DenseMatrix<T: Default + Copy, const C: usize = 32> {
    data: Vec<T>,
    indices: Vec<usize>,
}

impl<T: Default + Copy, const C: usize> DenseMatrix<T, C> {
    /// Create a new matrix with the given number of rows.
    pub fn new(rows: usize) -> Self {
        let data = Vec::new();
        let indices = Vec::new();
        let mut matrix = Self { data, indices };
        matrix.resize(rows);
        matrix
    }

    /// The number of columns of the matrix.
    pub const fn columns(&self) -> usize {
        C
    }

    /// The number of rows of the matrix.
    pub fn rows(&self) -> usize {
        self.indices.len()
    }

    /// Change the number of rows of the matrix.
    pub fn resize(&mut self, rows: usize) {
        // alway over-allocate columns to avoid alignment issues.
        let c = C.next_power_of_two();

        //
        let previous_rows = self.rows();
        let previous_offset = if self.rows() == 0 { 0 } else { self.indices[0] };

        // allocate data block
        self.data.resize_with((rows + 1) * c, T::default);

        // compute offset to aligned memory
        let mut offset = 0;
        while self.data[offset..].as_ptr() as usize % c > 0 {
            offset += 1
        }

        // copy data in case alignment offset changed
        if previous_offset != offset {
            self.data.as_mut_slice().copy_within(
                previous_offset..previous_offset + (previous_rows * c),
                offset,
            );
        }

        // record row coordinates
        self.indices.resize(rows, 0);
        for i in 0..rows {
            self.indices[i] = offset + i * c;
        }
    }
}

impl<T: Default + Copy, const C: usize> Index<usize> for DenseMatrix<T, C> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        let row = self.indices[index];
        &self.data[row..row + C]
    }
}

impl<T: Default + Copy, const C: usize> IndexMut<usize> for DenseMatrix<T, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = self.indices[index];
        &mut self.data[row..row + C]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_resize() {
        let mut dense = DenseMatrix::<u64, 32>::new(4);

        for i in 0..4 {
            dense[i][0] = (i + 1) as u64;
        }
        assert_eq!(dense[0][0], 1);
        assert_eq!(dense[1][0], 2);
        assert_eq!(dense[2][0], 3);
        assert_eq!(dense[3][0], 4);
        assert_eq!(dense[0].as_ptr() as usize % 4, 0);

        dense.resize(256);
        assert_eq!(dense[0][0], 1);
        assert_eq!(dense[1][0], 2);
        assert_eq!(dense[2][0], 3);
        assert_eq!(dense[3][0], 4);
        assert_eq!(dense[0].as_ptr() as usize % 4, 0);

        dense.resize(512);
        assert_eq!(dense[0][0], 1);
        assert_eq!(dense[1][0], 2);
        assert_eq!(dense[2][0], 3);
        assert_eq!(dense[3][0], 4);
        assert_eq!(dense[0].as_ptr() as usize % 4, 0);
    }
}
