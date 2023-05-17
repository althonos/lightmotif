//! Dense matrix storage with automatic memory alignment.

use std::iter::ExactSizeIterator;
use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::IndexMut;
use std::ptr::NonNull;

use typenum::marker_traits::Unsigned;

// --- DenseMatrix -------------------------------------------------------------

/// An aligned dense matrix of with a constant number of columns.
#[derive(Debug, Clone)]
pub struct DenseMatrix<T: Default + Copy, C: Unsigned> {
    data: Vec<T>,
    indices: Vec<usize>,
    _columns: std::marker::PhantomData<C>,
}

impl<T: Default + Copy, C: Unsigned> DenseMatrix<T, C> {
    /// Create a new matrix with the given number of rows.
    pub fn new(rows: usize) -> Self {
        let data = Vec::new();
        let indices = Vec::new();
        let mut matrix = Self {
            data,
            indices,
            _columns: std::marker::PhantomData,
        };
        matrix.resize(rows);
        matrix
    }

    /// Create a new *uninitialized* matrix with the given number of rows.
    pub unsafe fn uninitialized(rows: usize) -> Self {
        // alway over-allocate columns to avoid alignment issues.
        let c = C::USIZE.next_power_of_two();

        // NOTE: this is unsafe but given that we require `T` to be
        //       copy, this should be fine, as `Copy` prevents the
        //       type to be `Dorp` as well.
        // reserve the vector without initializing the data
        let mut data = Vec::with_capacity((rows + 1) * c);
        data.set_len((rows + 1) * c);

        // compute offset to aligned memory
        let mut offset = 0;
        while data[offset..].as_ptr() as usize % c > 0 {
            offset += 1
        }

        // record indices to each rows
        let indices = (0..rows).map(|i| offset + i * c).collect();

        Self {
            data,
            indices,
            _columns: std::marker::PhantomData,
        }
    }

    /// The number of columns of the matrix.
    #[inline]
    pub const fn columns(&self) -> usize {
        C::USIZE
    }

    /// The number of rows of the matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        self.indices.len()
    }

    /// Change the number of rows of the matrix.
    pub fn resize(&mut self, rows: usize) {
        // alway over-allocate columns to avoid alignment issues.
        let c = C::USIZE.next_power_of_two();

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

impl<T: Default + Copy, C: Unsigned> Index<usize> for DenseMatrix<T, C> {
    type Output = [T];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let row = self.indices[index];
        &self.data[row..row + C::USIZE]
    }
}

impl<T: Default + Copy, C: Unsigned> IndexMut<usize> for DenseMatrix<T, C> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let row = self.indices[index];
        &mut self.data[row..row + C::USIZE]
    }
}

impl<'a, T: Default + Copy, C: Unsigned> IntoIterator for &'a DenseMatrix<T, C> {
    type Item = &'a [T];
    type IntoIter = Iter<'a, T, C>;
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<'a, T: Default + Copy, C: Unsigned> IntoIterator for &'a mut DenseMatrix<T, C> {
    type Item = &'a mut [T];
    type IntoIter = IterMut<'a, T, C>;
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, T, C>
where
    T: 'a + Default + Copy,
    C: Unsigned,
{
    indices: std::slice::Iter<'a, usize>,
    data: std::ptr::NonNull<T>,
    _columns: std::marker::PhantomData<C>,
}

impl<'a, T, C> Iter<'a, T, C>
where
    T: 'a + Default + Copy,
    C: Unsigned,
{
    fn new(matrix: &'a DenseMatrix<T, C>) -> Self {
        let indices = matrix.indices.iter();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_ptr() as *mut T) };
        Self {
            indices,
            data,
            _columns: std::marker::PhantomData,
        }
    }

    fn get(&mut self, i: usize) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(i), C::USIZE) }
    }
}

// --- IterMut -----------------------------------------------------------------

pub struct IterMut<'a, T, C>
where
    T: 'a + Default + Copy,
    C: Unsigned,
{
    indices: std::slice::Iter<'a, usize>,
    data: std::ptr::NonNull<T>,
    _columns: std::marker::PhantomData<C>,
}

impl<'a, T, C> IterMut<'a, T, C>
where
    T: 'a + Default + Copy,
    C: Unsigned,
{
    fn new(matrix: &'a mut DenseMatrix<T, C>) -> Self {
        let indices = matrix.indices.iter();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        Self {
            indices,
            data,
            _columns: std::marker::PhantomData,
        }
    }

    fn get(&mut self, i: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr().add(i), C::USIZE) }
    }
}

// --- iterator ----------------------------------------------------------------

macro_rules! iterator {
    ($t:ident, $T:ident, $($item:tt)*) => {
        impl<'a, $T, C> Iterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
            C: Unsigned,
        {
            type Item = &'a $($item)*;
            fn next(&mut self) -> Option<Self::Item> {
                self.indices.next().map(|&i| self.get(i))
            }
        }

        impl<'a, $T, C> ExactSizeIterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
            C: Unsigned,
        {
            fn len(&self) -> usize {
                self.indices.len()
            }
        }

        impl<'a, $T, C> FusedIterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
            C: Unsigned,
        {}

        impl<'a, $T, C> DoubleEndedIterator for $t<'a, $T, C>
        where
            $T: Default + Copy,
            C: Unsigned,
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
    use typenum::consts::U32;

    use super::*;

    #[test]
    fn test_resize() {
        let mut dense = DenseMatrix::<u64, U32>::new(4);

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
        let mut dense = DenseMatrix::<u64, U32>::new(4);
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
