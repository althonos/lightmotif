//! Dense matrix storage with automatic memory alignment.

use std::fmt::Debug;
use std::fmt::Error as FmtError;
use std::fmt::Formatter;
use std::iter::ExactSizeIterator;
use std::iter::FusedIterator;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;
use std::ptr::NonNull;

use typenum::marker_traits::Unsigned;

// --- DefaultAlignment --------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type _DefaultAlignment = typenum::consts::U32;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
type _DefaultAlignment = typenum::consts::U16;

/// The default alignment used in dense matrices.
pub type DefaultAlignment = _DefaultAlignment;

// --- DenseMatrix -------------------------------------------------------------

/// A memory-aligned dense matrix with a constant number of columns.
#[derive(Clone, PartialEq, Eq)]
pub struct DenseMatrix<T: Default + Copy, C: Unsigned, A: Unsigned = DefaultAlignment> {
    data: Vec<T>,
    offset: usize,
    rows: usize,
    _columns: std::marker::PhantomData<C>,
    _alignment: std::marker::PhantomData<A>,
}

impl<T: Default + Copy, C: Unsigned, A: Unsigned> DenseMatrix<T, C, A> {
    /// Create a new matrix with the given number of rows.
    pub fn new(rows: usize) -> Self {
        let data = Vec::new();
        let mut matrix = Self {
            data,
            offset: 0,
            rows: 0,
            _columns: std::marker::PhantomData,
            _alignment: std::marker::PhantomData,
        };
        matrix.resize(rows);
        matrix
    }

    /// Create a new *uninitialized* matrix with the given number of rows.
    pub unsafe fn uninitialized(rows: usize) -> Self {
        // Always over-allocate columns to avoid alignment issues.
        let c = C::USIZE + (A::USIZE - C::USIZE % A::USIZE) * (C::USIZE % A::USIZE > 0) as usize;

        // NOTE: this is unsafe but given that we require `T` to be
        //       copy, this should be fine, as `Copy` prevents the
        //       type to be `Drop` as well.
        // reserve the vector without initializing the data
        let mut data = Vec::with_capacity((rows + 1) * c);
        data.set_len((rows + 1) * c);

        // compute offset to aligned memory
        let mut offset = 0;
        while data[offset..].as_ptr() as usize % c > 0 {
            offset += 1
        }

        Self {
            data,
            offset,
            rows,
            _columns: std::marker::PhantomData,
            _alignment: std::marker::PhantomData,
        }
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
        let mut dense = Self::new(it.len());

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
    /// # use typenum::{U43, U32};
    /// # use lightmotif::dense::DenseMatrix;
    /// let d = DenseMatrix::<u8, U43, U32>::new(0);
    /// assert_eq!(d.stride(), 64);
    /// ```
    #[inline]
    pub const fn stride(&self) -> usize {
        let x = std::mem::size_of::<T>();
        let c = C::USIZE * x;
        let b = c + (A::USIZE - c % A::USIZE) * ((c % A::USIZE) > 0) as usize;
        b / x + ((b % x) > 0) as usize
    }

    /// The number of rows of the matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Change the number of rows of the matrix.
    pub fn resize(&mut self, rows: usize) {
        // Always over-allocate columns to avoid alignment issues.
        let c: usize = self.stride();

        // Cache previous dimensions
        let previous_rows = self.rows;
        let previous_offset = self.offset;

        // Only reallocate if needed
        if previous_rows > rows {
            // Truncate rows
            self.data.truncate((rows + 1) * c);
        } else if previous_rows < rows {
            // Allocate data block
            self.data.resize_with((rows + 1) * c, T::default);
            // Compute offset to aligned memory
            self.offset = 0;
            while self.data[self.offset..].as_ptr() as usize % c > 0 {
                self.offset += 1
            }
            // Copy data in case alignment offset changed
            if previous_offset != self.offset {
                self.data.as_mut_slice().copy_within(
                    previous_offset..previous_offset + (previous_rows * c),
                    self.offset,
                );
            }
        }

        // Update row count
        self.rows = rows;
    }

    /// Iterate over the rows of the matrix.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, C, A> {
        Iter::new(self)
    }

    /// Returns an iterator that allows modifying each row.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, C, A> {
        IterMut::new(self)
    }

    /// Fill the entire matrix with a constant value.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T: Default + Copy + Debug, C: Unsigned, A: Unsigned> Debug for DenseMatrix<T, C, A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Default + Copy, C: Unsigned, A: Unsigned> Index<usize> for DenseMatrix<T, C, A> {
    type Output = [T];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let c = self.stride();
        let row = self.offset + c * index;
        &self.data[row..row + C::USIZE]
    }
}

impl<T: Default + Copy, C: Unsigned, A: Unsigned> IndexMut<usize> for DenseMatrix<T, C, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let c = self.stride();
        let row = self.offset + c * index;
        &mut self.data[row..row + C::USIZE]
    }
}

impl<'a, T: Default + Copy, C: Unsigned, A: Unsigned> IntoIterator for &'a DenseMatrix<T, C, A> {
    type Item = &'a [T];
    type IntoIter = Iter<'a, T, C, A>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<'a, T: Default + Copy, C: Unsigned, A: Unsigned> IntoIterator
    for &'a mut DenseMatrix<T, C, A>
{
    type Item = &'a mut [T];
    type IntoIter = IterMut<'a, T, C, A>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

// --- Iter --------------------------------------------------------------------

pub struct Iter<'a, T, C, A>
where
    T: 'a + Default + Copy,
    C: Unsigned,
    A: Unsigned,
{
    matrix: &'a DenseMatrix<T, C, A>,
    indices: Range<usize>,
    data: std::ptr::NonNull<T>,
}

impl<'a, T, C, A> Iter<'a, T, C, A>
where
    T: 'a + Default + Copy,
    C: Unsigned,
    A: Unsigned,
{
    fn new(matrix: &'a DenseMatrix<T, C, A>) -> Self {
        let indices = 0..matrix.rows();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_ptr() as *mut T) };
        Self {
            indices,
            matrix,
            data,
        }
    }

    #[inline]
    fn get(&mut self, i: usize) -> &'a [T] {
        let c = self.matrix.stride();
        let row = self.matrix.offset + c * i;
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(row), C::USIZE) }
    }
}

// --- IterMut -----------------------------------------------------------------

pub struct IterMut<'a, T, C, A>
where
    T: 'a + Default + Copy,
    C: Unsigned,
    A: Unsigned,
{
    matrix: &'a mut DenseMatrix<T, C, A>,
    indices: Range<usize>,
    data: std::ptr::NonNull<T>,
}

impl<'a, T, C, A> IterMut<'a, T, C, A>
where
    T: 'a + Default + Copy,
    C: Unsigned,
    A: Unsigned,
{
    fn new(matrix: &'a mut DenseMatrix<T, C, A>) -> Self {
        let indices = 0..matrix.rows();
        let data = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        Self {
            indices,
            data,
            matrix,
        }
    }

    #[inline]
    fn get(&mut self, i: usize) -> &'a mut [T] {
        let c = self.matrix.stride();
        let row = self.matrix.offset + c * i;
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr().add(row), C::USIZE) }
    }
}

// --- iterator ----------------------------------------------------------------

macro_rules! iterator {
    ($t:ident, $T:ident, $($item:tt)*) => {
        impl<'a, $T, C, A> Iterator for $t<'a, $T, C, A>
        where
            $T: Default + Copy,
            C: Unsigned,
            A: Unsigned,
        {
            type Item = &'a $($item)*;
            fn next(&mut self) -> Option<Self::Item> {
                self.indices.next().map(|i| self.get(i))
            }
        }

        impl<'a, $T, C, A> ExactSizeIterator for $t<'a, $T, C, A>
        where
            $T: Default + Copy,
            C: Unsigned,
            A: Unsigned,
        {
            #[inline]
            fn len(&self) -> usize {
                self.indices.len()
            }
        }

        impl<'a, $T, C, A> FusedIterator for $t<'a, $T, C, A>
        where
            $T: Default + Copy,
            C: Unsigned,
            A: Unsigned,
        {}

        impl<'a, $T, C, A> DoubleEndedIterator for $t<'a, $T, C, A>
        where
            $T: Default + Copy,
            C: Unsigned,
            A: Unsigned,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.indices.next_back().map(|i| self.get(i))
            }
        }
    };
}

iterator!(Iter, T, [T]);
iterator!(IterMut, T, mut [T]);

#[cfg(test)]
mod test {
    use typenum::consts::U15;
    use typenum::consts::U16;
    use typenum::consts::U3;
    use typenum::consts::U32;
    use typenum::consts::U8;

    use super::*;

    #[test]
    fn test_stride() {
        let d1 = DenseMatrix::<u8, U32, U32>::new(0);
        assert_eq!(d1.stride(), 32);

        let d2 = DenseMatrix::<u8, U16, U32>::new(0);
        assert_eq!(d2.stride(), 32);

        let d3 = DenseMatrix::<u32, U32, U32>::new(0);
        assert_eq!(d3.stride(), 32);

        let d4 = DenseMatrix::<u32, U8, U32>::new(0);
        assert_eq!(d4.stride(), 8);

        let d5 = DenseMatrix::<u32, U16, U32>::new(0);
        assert_eq!(d5.stride(), 16);

        let d6 = DenseMatrix::<u32, U3, U16>::new(0);
        assert_eq!(d6.stride(), 4);

        let d7 = DenseMatrix::<u8, U15, U8>::new(0);
        assert_eq!(d7.stride(), 16);
    }

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
