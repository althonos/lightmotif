use std::ops::Index;
use std::ops::IndexMut;
use std::ptr::NonNull;

use std::iter::ExactSizeIterator;
use std::iter::FusedIterator;

// --- DenseMatrix -------------------------------------------------------------

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
    #[inline]
    pub const fn columns(&self) -> usize {
        C
    }

    /// The number of rows of the matrix.
    #[inline]
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

    /// Iterate over the rows of the matrix.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, C> {
        Iter::new(self)
    }

    /// Returns an iterator that allows modifying each row.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, C> {
        IterMut::new(self)
    }
}

impl<T: Default + Copy, const C: usize> Index<usize> for DenseMatrix<T, C> {
    type Output = [T];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let row = self.indices[index];
        &self.data[row..row + C]
    }
}

impl<T: Default + Copy, const C: usize> IndexMut<usize> for DenseMatrix<T, C> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = self.indices[index];
        &mut self.data[row..row + C]
    }
}

impl<'a, T: Default + Copy, const C: usize> IntoIterator for &'a DenseMatrix<T, C> {
    type Item = &'a [T];
    type IntoIter = Iter<'a, T, C>;
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<'a, T: Default + Copy, const C: usize> IntoIterator for &'a mut DenseMatrix<T, C> {
    type Item = &'a mut [T];
    type IntoIter = IterMut<'a, T, C>;
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, T, const C: usize>
where
    T: 'a + Default + Copy,
{
    indices: std::slice::Iter<'a, usize>,
    data: std::ptr::NonNull<T>,
}

impl<'a, T, const C: usize> Iter<'a, T, C>
where
    T: 'a + Default + Copy,
{
    fn new(matrix: &'a DenseMatrix<T, C>) -> Self {
        let indices = matrix.indices.iter();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_ptr() as *mut T) };
        Self { indices, data }
    }

    fn get(&mut self, i: usize) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(i), C) }
    }
}

// --- IterMut -----------------------------------------------------------------

pub struct IterMut<'a, T, const C: usize>
where
    T: 'a + Default + Copy,
{
    indices: std::slice::Iter<'a, usize>,
    data: std::ptr::NonNull<T>,
}

impl<'a, T, const C: usize> IterMut<'a, T, C>
where
    T: 'a + Default + Copy,
{
    fn new(matrix: &'a mut DenseMatrix<T, C>) -> Self {
        let indices = matrix.indices.iter();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        Self { indices, data }
    }

    fn get(&mut self, i: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr().add(i), C) }
    }
}

// --- iterator ----------------------------------------------------------------

macro_rules! iterator {
    ($t:ident, $T:ident, $($item:tt)*) => {
        impl<'a, $T, const C: usize> Iterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
        {
            type Item = &'a $($item)*;
            fn next(&mut self) -> Option<Self::Item> {
                self.indices.next().map(|&i| self.get(i))
            }
        }

        impl<'a, $T, const C: usize> ExactSizeIterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
        {
            fn len(&self) -> usize {
                self.indices.len()
            }
        }

        impl<'a, $T, const C: usize> FusedIterator for $t<'a, $T, C>
        where
            $T: Default + Copy
        {}

        impl<'a, $T, const C: usize> DoubleEndedIterator for $t<'a, $T, C>
        where
            $T: Default + Copy
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.indices.next_back().map(|&i| self.get(i))
            }
        }
    };
}

iterator!(Iter, T, [T]);
iterator!(IterMut, T, mut [T]);

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

    #[test]
    fn test_iter_mut() {
        let mut dense = DenseMatrix::<u64, 32>::new(4);
        for i in 0..4 {
            dense[i][0] = (i + 1) as u64;
        }
        assert_eq!(dense[0][0], 1);
        assert_eq!(dense[1][0], 2);
        assert_eq!(dense[2][0], 3);
        assert_eq!(dense[3][0], 4);

        for row in dense.iter_mut() {
            row[0] *= 4;
        }
        assert_eq!(dense[0][0], 4);
        assert_eq!(dense[1][0], 8);
        assert_eq!(dense[2][0], 12);
        assert_eq!(dense[3][0], 16);
    }
}
