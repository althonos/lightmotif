//! Dense matrix storage with automatic memory alignment.

use std::fmt::Debug;
use std::fmt::Error as FmtError;
use std::fmt::Formatter;
use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::IndexMut;

use crate::num::ArrayLength;
use crate::pli::dispatch::Dispatch;
use crate::pli::platform::Backend;

// --- Default Columns ---------------------------------------------------------

/// The default number of columns.
pub type DefaultColumns = <Dispatch as Backend>::Lanes;

// --- MatrixElement -----------------------------------------------------------

/// A marker trait for types allowed as `DenseMatrix` elements.
pub trait MatrixElement: Default + Copy {}

impl<T> MatrixElement for T where T: Default + Copy {}

// --- DenseMatrix -------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MatrixCoordinates {
    pub row: usize,
    pub col: usize,
}

impl MatrixCoordinates {
    #[inline]
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

// --- DenseMatrix -------------------------------------------------------------

#[cfg_attr(target_arch = "x86_64", repr(align(32)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(align(16)))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Row<T: MatrixElement, C: ArrayLength = DefaultColumns> {
    a: generic_array::GenericArray<T, C>,
}

/// A memory-aligned dense matrix with a constant number of columns.
#[derive(Clone, PartialEq, Eq)]
pub struct DenseMatrix<T: MatrixElement, C: ArrayLength = DefaultColumns> {
    data: Vec<Row<T, C>>,
    rows: usize,
}

impl<T: MatrixElement, C: ArrayLength> DenseMatrix<T, C> {
    /// Create a new matrix with the given number of rows.
    pub fn new(rows: usize) -> Self {
        let data = Vec::new();
        let mut matrix = Self { data, rows: 0 };
        matrix.resize(rows);
        matrix
    }

    /// Create a new matrix with the given number of rows and the given capacity.
    pub fn with_capacity(rows: usize, capacity: usize) -> Self {
        let data = Vec::with_capacity(capacity);
        let mut matrix = Self { data, rows: 0 };
        matrix.resize(rows);
        matrix
    }

    /// Create a new *uninitialized* matrix with the given number of rows.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninitialized(rows: usize) -> Self {
        let mut m = Self::new(0);
        m.data.reserve(rows);
        m.data.set_len(rows);
        m.rows = rows;
        m
    }

    /// Create a new dense matrix from an iterable of rows.
    ///
    /// # Panics
    ///
    /// Panics if any of the rows does not have the number of elements
    /// corresponding to the dense matrix columns.
    pub fn from_rows<I>(rows: I) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: AsRef<[T]>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let it = rows.into_iter();
        let mut dense = unsafe { Self::uninitialized(it.len()) };

        for (i, row) in it.enumerate() {
            dense[i].copy_from_slice(row.as_ref());
        }

        dense
    }

    /// The number of columns of the matrix.
    #[inline]
    pub const fn columns(&self) -> usize {
        C::USIZE
    }

    /// The stride of the matrix, as a number of elements.
    ///
    /// This may be different from the number of columns to account for memory
    /// alignment constraints. Multiply by `std::mem::size_of::<T>()` to obtain
    /// the stride in bytes.
    ///
    /// # Example
    /// ```rust
    /// # use lightmotif::num::U43;
    /// # use lightmotif::dense::DenseMatrix;
    /// let d = DenseMatrix::<u8, U43>::new(0);
    /// assert!(d.stride() == 64 || d.stride() == 48);
    /// ```
    #[inline]
    pub const fn stride(&self) -> usize {
        std::mem::size_of::<Row<T, C>>() / std::mem::size_of::<T>()
    }

    /// The row capacity of the matrix.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// The number of rows of the matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Change the number of rows of the matrix.
    #[inline]
    pub fn resize(&mut self, rows: usize) {
        self.data.resize_with(rows, Default::default);
        self.rows = rows;
    }

    /// Reserve a number of rows in the matrix.
    #[inline]
    pub fn reserve(&mut self, capacity: usize) {
        self.data.reserve(capacity);
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

    /// Return the matrix content as a flat slice, including padding.
    #[inline]
    pub unsafe fn ravel(&self) -> &[T] {
        std::slice::from_raw_parts(self.data.as_ptr() as *mut T, self.rows() * self.stride())
    }

    /// Return the matrix content as a flat mutable slice, including padding.
    #[inline]
    pub unsafe fn ravel_mut(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(
            self.data.as_mut_ptr() as *mut T,
            self.rows() * self.stride(),
        )
    }

    /// Fill the entire matrix with a constant value.
    #[inline]
    pub fn fill(&mut self, value: T) {
        unsafe {
            self.ravel_mut().fill(value);
        }
    }
}

impl<T: MatrixElement, C: ArrayLength> AsRef<DenseMatrix<T, C>> for DenseMatrix<T, C> {
    #[inline]
    fn as_ref(&self) -> &DenseMatrix<T, C> {
        self
    }
}

impl<T: MatrixElement + Debug, C: ArrayLength> Debug for DenseMatrix<T, C> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: MatrixElement, C: ArrayLength> Index<usize> for DenseMatrix<T, C> {
    type Output = [T];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.data[index].a.as_slice()
    }
}

impl<T: MatrixElement, C: ArrayLength> IndexMut<usize> for DenseMatrix<T, C> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data[index].a.as_mut_slice()
    }
}

impl<T: MatrixElement, C: ArrayLength> Index<MatrixCoordinates> for DenseMatrix<T, C> {
    type Output = T;
    #[inline]
    fn index(&self, index: MatrixCoordinates) -> &Self::Output {
        &self.data[index.row].a[index.col]
    }
}

impl<T: MatrixElement, C: ArrayLength> IndexMut<MatrixCoordinates> for DenseMatrix<T, C> {
    #[inline]
    fn index_mut(&mut self, index: MatrixCoordinates) -> &mut Self::Output {
        &mut self.data[index.row].a[index.col]
    }
}

impl<'a, T: MatrixElement, C: ArrayLength> IntoIterator for &'a DenseMatrix<T, C> {
    type Item = &'a [T];
    type IntoIter = Iter<'a, T, C>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<'a, T: MatrixElement, C: ArrayLength> IntoIterator for &'a mut DenseMatrix<T, C> {
    type Item = &'a mut [T];
    type IntoIter = IterMut<'a, T, C>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, T, C>
where
    T: 'a + MatrixElement,
    C: ArrayLength,
{
    it: <&'a Vec<Row<T, C>> as IntoIterator>::IntoIter,
}

impl<'a, T, C> Iter<'a, T, C>
where
    T: 'a + MatrixElement,
    C: ArrayLength,
{
    #[inline]
    fn new(matrix: &'a DenseMatrix<T, C>) -> Self {
        Self {
            it: matrix.data.iter(),
        }
    }

    #[inline]
    fn get(row: &'a Row<T, C>) -> &'a [T] {
        row.a.as_slice()
    }
}

// --- IterMut -----------------------------------------------------------------

pub struct IterMut<'a, T, C>
where
    T: 'a + MatrixElement,
    C: ArrayLength,
{
    it: <&'a mut Vec<Row<T, C>> as IntoIterator>::IntoIter,
}

impl<'a, T, C> IterMut<'a, T, C>
where
    T: 'a + MatrixElement,
    C: ArrayLength,
{
    #[inline]
    fn new(matrix: &'a mut DenseMatrix<T, C>) -> Self {
        Self {
            it: matrix.data.iter_mut(),
        }
    }

    #[inline]
    fn get(row: &'a mut Row<T, C>) -> &'a mut [T] {
        row.a.as_mut_slice()
    }
}

// --- iterator ----------------------------------------------------------------

macro_rules! iterator {
    ($t:ident, $T:ident, $($item:tt)*) => {
        impl<'a, $T, C> Iterator for $t<'a, $T, C>
        where
            $T: MatrixElement,
            C: ArrayLength,
        {
            type Item = &'a $($item)*;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.it.next().map(|row| Self::get(row))
            }
        }

        impl<'a, $T, C> ExactSizeIterator for $t<'a, $T, C>
        where
            $T: MatrixElement,
            C: ArrayLength,
        {
            #[inline]
            fn len(&self) -> usize {
                self.it.len()
            }
        }

        impl<'a, $T, C> FusedIterator for $t<'a, $T, C>
        where
            $T: MatrixElement,
            C: ArrayLength,
        {}

        impl<'a, $T, C> DoubleEndedIterator for $t<'a, $T, C>
        where
            $T: MatrixElement,
            C: ArrayLength,
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.it.next_back().map(|row| Self::get(row))
            }
        }
    };
}

iterator!(Iter, T, [T]);
iterator!(IterMut, T, mut [T]);

#[cfg(test)]
mod test {
    use typenum::consts::U16;
    use typenum::consts::U32;
    use typenum::consts::U33;
    use typenum::consts::U8;

    use super::*;

    #[test]
    fn stride() {
        let d1 = DenseMatrix::<u8, U32>::new(0);
        assert_eq!(d1.stride(), 32);

        let d2 = DenseMatrix::<u8, U16>::new(0);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(d2.stride(), 32);
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(d2.stride(), 16);

        let d3 = DenseMatrix::<u32, U32>::new(0);
        assert_eq!(d3.stride(), 32);

        let d4 = DenseMatrix::<u32, U8>::new(0);
        assert_eq!(d4.stride(), 8);

        let d5 = DenseMatrix::<u32, U16>::new(0);
        assert_eq!(d5.stride(), 16);

        let d7 = DenseMatrix::<u8, U33>::new(0);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(d7.stride(), 64);
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(d7.stride(), 48);
    }

    #[test]
    fn resize() {
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
    fn iter_mut() {
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

    #[test]
    fn index_matrix_coordinates() {
        let mut dense = DenseMatrix::<u64, U32>::new(4);
        for i in 0..4 {
            dense[i][0] = (i + 1) as u64;
        }
        assert_eq!(dense[MatrixCoordinates::new(0, 0)], 1);
        assert_eq!(dense[MatrixCoordinates::new(1, 0)], 2);
        assert_eq!(dense[MatrixCoordinates::new(2, 0)], 3);
        assert_eq!(dense[MatrixCoordinates::new(3, 0)], 4);

        for row in dense.iter_mut() {
            row[0] *= 4;
        }
        assert_eq!(dense[MatrixCoordinates::new(0, 0)], 4);
        assert_eq!(dense[MatrixCoordinates::new(1, 0)], 8);
        assert_eq!(dense[MatrixCoordinates::new(2, 0)], 12);
        assert_eq!(dense[MatrixCoordinates::new(3, 0)], 16);
    }
}
