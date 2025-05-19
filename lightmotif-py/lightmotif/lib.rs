#![doc = include_str!("../README.md")]

extern crate generic_array;
extern crate lightmotif;
#[cfg(feature = "tfmpvalue")]
extern crate lightmotif_tfmpvalue;
extern crate pyo3;

use std::fmt::Display;
use std::fmt::Formatter;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::abc::Nucleotide;
use lightmotif::abc::Protein;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::num::Unsigned;
use lightmotif::pli::Score;
#[cfg(feature = "tfmpvalue")]
use lightmotif_tfmpvalue::TfmPvalue;

use generic_array::GenericArray;
use pyo3::exceptions::PyBufferError;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_ssize_t;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyString;

mod io;
mod pyfile;

// --- Macros ------------------------------------------------------------------

macro_rules! apply_self {
    ($self:expr, |$x:ident| $e:expr) => {
        match &$self {
            Self::Dna($x) => $e,
            Self::Protein($x) => $e,
        }
    };
}

macro_rules! impl_matrix_methods {
    ($datatype:ident) => {
        impl $datatype {
            #[allow(unused)]
            fn rows(&self) -> usize {
                apply_self!(self, |x| x.matrix().rows())
            }

            #[allow(unused)]
            fn columns(&self) -> usize {
                apply_self!(self, |x| x.matrix().columns())
            }

            #[allow(unused)]
            fn stride(&self) -> usize {
                apply_self!(self, |x| x.matrix().stride())
            }
        }
    };
}

macro_rules! alphabet_data_enum {
    (
        #[derive($($t:ident),*)]
        enum $name:ident($($p:ident)::*)
    ) => {

        #[derive($($t),*)]
        enum $name {
            Dna($($p)::*<Dna>),
            Protein($($p)::*<Protein>)
        }

        impl From<$($p)::*<Dna>> for $name {
            fn from(value: $($p)::*<Dna>) -> Self {
                $name::Dna(value)
            }
        }

        impl From<$($p)::*<Protein>> for $name {
            fn from(value: $($p)::*<Protein>) -> Self {
                $name::Protein(value)
            }
        }
    }
}

// --- Helpers -----------------------------------------------------------------

fn dict_to_alphabet_array<'py, A: lightmotif::Alphabet>(
    d: Bound<'py, PyDict>,
) -> PyResult<GenericArray<f32, A::K>> {
    let mut p = std::iter::repeat(0.0)
        .take(A::K::USIZE)
        .collect::<GenericArray<f32, A::K>>();
    for (k, v) in d.iter() {
        let s = k.extract::<Bound<PyString>>()?;
        let key = s.to_cow()?;
        if key.len() != 1 {
            return Err(PyValueError::new_err((
                "Invalid key for pseudocount:",
                s.to_string(),
            )));
        }
        let x = key.chars().next().unwrap();
        let symbol = <A as lightmotif::Alphabet>::Symbol::from_char(x)
            .map_err(|_| PyValueError::new_err(("Invalid key for pseudocount:", x)))?;
        let value = v.extract::<f32>()?;
        p[symbol.as_index()] = value;
    }
    Ok(p)
}

// --- EncodedSequence ---------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug)]
    enum EncodedSequenceData(lightmotif::seq::EncodedSequence)
);

impl EncodedSequenceData {
    unsafe fn as_ptr(&self) -> *const u8 {
        apply_self!(self, |x| {
            let data: &[_] = x.as_ref();
            data.as_ptr() as *const u8
        })
    }

    fn len(&self) -> usize {
        apply_self!(self, |x| x.len())
    }

    fn get(&self, index: usize) -> u8 {
        apply_self!(self, |x| x[index] as u8)
    }

    fn stripe(&self) -> StripedSequenceData {
        apply_self!(self, |x| x.to_striped().into())
    }
}

impl Display for EncodedSequenceData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        apply_self!(self, |x| x.fmt(f))
    }
}

/// A biological sequence encoded as digits.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct EncodedSequence {
    data: EncodedSequenceData,
}

#[pymethods]
impl EncodedSequence {
    /// Encode a sequence.
    #[new]
    #[pyo3(signature = (sequence, protein=false))]
    pub fn __init__<'py>(
        sequence: Bound<'py, PyString>,
        protein: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let seq = sequence.to_cow()?;
        let py = sequence.py();
        let data = py
            .allow_threads(|| {
                if protein {
                    lightmotif::seq::EncodedSequence::<Protein>::encode(&*seq)
                        .map(EncodedSequenceData::from)
                } else {
                    lightmotif::seq::EncodedSequence::<Dna>::encode(&*seq)
                        .map(EncodedSequenceData::from)
                }
            })
            .map_err(|lightmotif::err::InvalidSymbol(x)| {
                PyValueError::new_err(format!("Invalid symbol in input: {}", x))
            })?;
        Ok(Self::from(data).into())
    }

    /// Convert the encoded sequence to a string.
    pub fn __str__(&self) -> String {
        self.data.to_string()
    }

    /// Get the length of the encoded sequence.
    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Get a copy of the encoded sequence.
    pub fn __copy__(&self) -> Self {
        self.copy()
    }

    /// Get an element of the encoded sequence.
    pub fn __getitem__(&self, mut index: Py_ssize_t) -> PyResult<u8> {
        let length = self.data.len();
        if index < 0 {
            index += length as Py_ssize_t;
        }
        if index < 0 || index >= length as Py_ssize_t {
            Err(PyIndexError::new_err("sequence index out of range"))
        } else {
            Ok(self.data.get(index as usize))
        }
    }

    /// Get the underlying memory of the encoded sequence.
    #[cfg(not(feature = "abi3"))]
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        (*view).buf = slf.data.as_ptr() as *mut std::os::raw::c_void;
        (*view).len = slf.data.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<u8>() as isize;

        (*view).obj = slf.as_ptr();
        pyo3::ffi::Py_INCREF((*view).obj);
        let msg = std::ffi::CStr::from_bytes_with_nul(b"B\0").unwrap();
        (*view).format = msg.as_ptr() as *mut _;

        (*view).ndim = 1;
        (*view).shape = std::ptr::null_mut();
        (*view).strides = std::ptr::null_mut();
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    /// `bool`: `True` if the encoded sequence is a protein sequence.
    #[getter]
    pub fn protein(&self) -> bool {
        matches!(self.data, EncodedSequenceData::Protein(_))
    }

    /// Create a copy of this sequence.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Convert this sequence into a striped matrix.
    ///
    /// Returns:
    ///     `~lightmotif.StripedSequence`: The input sequence, stored in a
    ///     column-major matrix.
    ///
    pub fn stripe(&self) -> StripedSequence {
        self.data.stripe().into()
    }
}

impl From<EncodedSequenceData> for EncodedSequence {
    fn from(data: EncodedSequenceData) -> Self {
        Self { data }
    }
}

// --- StripedSequence ---------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug)]
    enum StripedSequenceData(lightmotif::seq::StripedSequence)
);

impl_matrix_methods!(StripedSequenceData);

impl StripedSequenceData {
    unsafe fn as_ptr(&self) -> *const u8 {
        apply_self!(self, |x| x.matrix()[0].as_ptr() as *const u8)
    }
}

/// An encoded biological sequence stored in a column-major matrix.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct StripedSequence {
    data: StripedSequenceData,
    shape: [Py_ssize_t; 2],
    strides: [Py_ssize_t; 2],
}

impl From<StripedSequenceData> for StripedSequence {
    fn from(data: StripedSequenceData) -> Self {
        let cols = data.columns();
        let rows = data.rows();
        let shape = [cols as Py_ssize_t, rows as Py_ssize_t];
        let strides = [
            std::mem::size_of::<u8>() as Py_ssize_t,
            (std::mem::size_of::<u8>() * data.stride()) as Py_ssize_t,
        ];
        Self {
            data,
            shape,
            strides,
        }
    }
}

#[pymethods]
impl StripedSequence {
    /// Get a copy of the striped sequence.
    pub fn __copy__(&self) -> Self {
        self.copy()
    }

    /// `bool`: `True` if the striped sequence is a protein sequence.
    #[getter]
    pub fn protein(&self) -> bool {
        matches!(self.data, StripedSequenceData::Protein(_))
    }

    #[cfg(not(feature = "abi3"))]
    unsafe fn __getbuffer__(
        mut slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let data = slf.data.as_ptr();
        (*view).buf = data as *mut std::os::raw::c_void;
        (*view).len = (slf.data.rows() * slf.data.columns()) as isize;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<Nucleotide>() as isize;

        (*view).obj = slf.as_ptr();
        pyo3::ffi::Py_INCREF((*view).obj);
        let msg = std::ffi::CStr::from_bytes_with_nul(b"B\0").unwrap();
        (*view).format = msg.as_ptr() as *mut _;

        (*view).ndim = 2;
        (*view).shape = slf.shape.as_mut_ptr();
        (*view).strides = slf.strides.as_mut_ptr();
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    /// Create a copy of this sequence.
    pub fn copy(&self) -> Self {
        self.clone()
    }
}

// --- CountMatrix -------------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug, PartialEq)]
    enum CountMatrixData(lightmotif::pwm::CountMatrix)
);

impl_matrix_methods!(CountMatrixData);

impl CountMatrixData {
    fn get(&self, index: usize) -> &[u32] {
        apply_self!(self, |x| &x.matrix()[index])
    }
}

/// A matrix storing the count of a motif letters at each position.
#[pyclass(module = "lightmotif.lib", sequence)]
#[derive(Clone, Debug)]
pub struct CountMatrix {
    data: CountMatrixData,
}

impl CountMatrix {
    fn new<D>(data: D) -> Self
    where
        D: Into<CountMatrixData>,
    {
        Self { data: data.into() }
    }
}

#[pymethods]
impl CountMatrix {
    /// Create a new count matrix.
    #[new]
    #[allow(unused_variables)]
    #[pyo3(signature = (values, *, protein = false))]
    pub fn __init__<'py>(
        values: Bound<'py, PyDict>,
        protein: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        macro_rules! run {
            ($alphabet:ty) => {{
                // Extract values from dictionary into matrix
                let mut data: Option<DenseMatrix<u32, <$alphabet as Alphabet>::K>> = None;
                for s in <$alphabet as Alphabet>::symbols() {
                    let key = String::from(s.as_char());
                    if let Some(res) = values.get_item(&key)? {
                        let column = res;
                        if data.is_none() {
                            data = Some(DenseMatrix::new(column.len()?));
                        }
                        let matrix = data.as_mut().unwrap();
                        if matrix.rows() != column.len()? {
                            return Err(PyValueError::new_err("Invalid number of rows"));
                        }
                        for (i, x) in column.try_iter()?.enumerate() {
                            matrix[i][s.as_index()] = x?.extract::<u32>()?;
                        }
                    }
                }

                match data {
                    None => Err(PyValueError::new_err("Invalid count matrix")),
                    Some(matrix) => match lightmotif::CountMatrix::<$alphabet>::new(matrix) {
                        Ok(counts) => Ok(CountMatrix::new(counts).into()),
                        Err(_) => Err(PyValueError::new_err("Inconsistent rows in count matrix")),
                    },
                }
            }};
        }
        if protein {
            run!(Protein)
        } else {
            run!(Dna)
        }
    }

    pub fn __eq__<'py>(&self, object: Bound<'py, PyAny>) -> PyResult<bool> {
        if let Ok(other) = object.extract::<PyRef<Self>>() {
            Ok(self.data == other.data)
        } else {
            Ok(false)
        }
    }

    pub fn __len__(&self) -> usize {
        self.data.rows()
    }

    pub fn __getitem__<'py>(slf: PyRef<'py, Self>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();

        let mut index_ = index;
        if index_ < 0 {
            index_ += slf.data.rows() as isize;
        }
        if index_ < 0 || (index_ as usize) >= slf.data.rows() {
            return Err(PyIndexError::new_err(index));
        }

        let row = slf.data.get(index as usize);
        row.into_pyobject(py)
    }

    /// `bool`: `True` if the count matrix stores protein counts.
    #[getter]
    pub fn protein(&self) -> bool {
        matches!(self.data, CountMatrixData::Protein(_))
    }

    /// Normalize this count matrix to obtain a position weight matrix.
    ///
    /// This method converts the count matrix to a weight matrix. Each row
    /// from the matrix is normalized so that they sum to ``1.0``. Each element
    /// is then divided by a uniform background probability to obtain
    /// odds-ratio at every position of the motif. Pseudocounts can be given
    /// to prevent zero elements, which may translate into -∞ scores in the
    /// final position-specific scoring matrix.
    ///
    /// Arguments:
    ///     pseudocount (`float`, `dict` or `None`): The pseudocounts to apply
    ///         before normalizing the count matrix. If a `float` is given,
    ///         then a similar pseudocount is applied to every column of the
    ///         matrix (excluding the default symbol). Otherwise, a `dict`
    ///         may be given to map each symbol of the alphabet to a distinct
    ///         pseudocount. If `None` given, no pseudocount is used.
    ///
    #[pyo3(signature = (pseudocount=None))]
    pub fn normalize(&self, pseudocount: Option<PyObject>) -> PyResult<WeightMatrix> {
        macro_rules! run {
            ($data:ident, $alphabet:ty) => {{
                let pseudo = Python::with_gil(|py| {
                    if let Some(obj) = pseudocount {
                        if let Ok(x) = obj.extract::<f32>(py) {
                            Ok(lightmotif::abc::Pseudocounts::from(x))
                        } else if let Ok(d) = obj.extract::<Bound<PyDict>>(py) {
                            let p = dict_to_alphabet_array::<$alphabet>(d)?;
                            Ok(lightmotif::abc::Pseudocounts::from(p))
                        } else {
                            Err(PyTypeError::new_err("Invalid type for pseudocount"))
                        }
                    } else {
                        Ok(lightmotif::abc::Pseudocounts::default())
                    }
                })?;
                let data = $data.to_freq(pseudo).to_weight(None);
                Ok(WeightMatrix::new(data))
            }};
        }
        match &self.data {
            CountMatrixData::Dna(dna) => run!(dna, Dna),
            CountMatrixData::Protein(prot) => run!(prot, Protein),
        }
    }
}

impl From<CountMatrixData> for CountMatrix {
    fn from(data: CountMatrixData) -> Self {
        Self { data }
    }
}

// --- WeightMatrix ------------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug, PartialEq)]
    enum WeightMatrixData(lightmotif::pwm::WeightMatrix)
);

impl_matrix_methods!(WeightMatrixData);

impl WeightMatrixData {
    fn get(&self, index: usize) -> &[f32] {
        apply_self!(self, |x| &x.matrix()[index])
    }
}

/// A matrix storing position-specific odds-ratio for a motif.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    data: WeightMatrixData,
}

impl WeightMatrix {
    fn new<D>(data: D) -> Self
    where
        D: Into<WeightMatrixData>,
    {
        Self { data: data.into() }
    }
}

#[pymethods]
impl WeightMatrix {
    pub fn __eq__<'py>(&self, object: Bound<'py, PyAny>) -> PyResult<bool> {
        if let Ok(other) = object.extract::<PyRef<Self>>() {
            Ok(self.data == other.data)
        } else {
            Ok(false)
        }
    }

    pub fn __len__(&self) -> usize {
        self.data.rows()
    }

    pub fn __getitem__<'py>(slf: PyRef<'py, Self>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();

        let mut index_ = index;
        if index_ < 0 {
            index_ += slf.data.rows() as isize;
        }
        if index_ < 0 || (index_ as usize) >= slf.data.rows() {
            return Err(PyIndexError::new_err(index));
        }

        let row = slf.data.get(index as usize);
        row.into_pyobject(py)
    }

    /// `bool`: `True` if the weight matrix stores protein weights.
    #[getter]
    pub fn protein(&self) -> bool {
        matches!(self.data, WeightMatrixData::Protein(_))
    }

    /// Log-scale this weight matrix to obtain a position-specific scoring matrix.
    ///
    /// Arguments:
    ///     background (`dict` or `None`): The background frequencies to use for
    ///         rescaling the weight matrix before computing log-odds-ratio. If
    ///         `None` given, uniform background frequencies will be used.
    ///
    #[pyo3(signature=(background=None, base=2.0))]
    pub fn log_odds(&self, background: Option<PyObject>, base: f32) -> PyResult<ScoringMatrix> {
        macro_rules! run {
            ($data:ident, $alphabet:ty) => {{
                // extract the background from the method argument
                let bg = Python::with_gil(|py| {
                    if let Some(obj) = background {
                        if let Ok(d) = obj.extract::<Bound<PyDict>>(py) {
                            let p = dict_to_alphabet_array::<$alphabet>(d)?;
                            lightmotif::abc::Background::new(p).map_err(|_| {
                                PyValueError::new_err("Invalid background frequencies")
                            })
                        } else {
                            Err(PyTypeError::new_err("Invalid type for pseudocount"))
                        }
                    } else {
                        Ok(lightmotif::abc::Background::uniform())
                    }
                })?;
                // rescale if backgrounds do not match
                let pwm = match bg.frequencies() != $data.background().frequencies() {
                    false => $data.rescale(bg),
                    true => $data.clone(),
                };
                Ok(ScoringMatrix::new(pwm.to_scoring_with_base(base)))
            }};
        }
        match &self.data {
            WeightMatrixData::Dna(dna) => run!(dna, Dna),
            WeightMatrixData::Protein(protein) => run!(protein, Protein),
        }
    }
}

impl From<WeightMatrixData> for WeightMatrix {
    fn from(data: WeightMatrixData) -> Self {
        Self::new(data)
    }
}

// --- ScoringMatrix -----------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug, PartialEq)]
    enum ScoringMatrixData(lightmotif::pwm::ScoringMatrix)
);

impl_matrix_methods!(ScoringMatrixData);

impl ScoringMatrixData {
    fn get(&self, index: usize) -> &[f32] {
        apply_self!(self, |x| &x[index])
    }

    unsafe fn as_ptr(&self) -> *const f32 {
        apply_self!(self, |x| x.matrix()[0].as_ptr())
    }
}

/// A matrix storing position-specific odds-ratio for a motif.
#[pyclass(module = "lightmotif.lib")]
#[derive(Debug)]
pub struct ScoringMatrix {
    data: ScoringMatrixData,
    distribution: Option<Py<ScoreDistribution>>,
    shape: [Py_ssize_t; 2],
    strides: [Py_ssize_t; 2],
}

impl ScoringMatrix {
    fn new<D>(data: D) -> Self
    where
        D: Into<ScoringMatrixData>,
    {
        let data = data.into();
        let cols = data.columns();
        let rows = data.rows();
        let stride = data.stride();
        let shape = [cols as Py_ssize_t, rows as Py_ssize_t];
        let strides = [
            (stride * std::mem::size_of::<f32>()) as Py_ssize_t,
            std::mem::size_of::<f32>() as Py_ssize_t,
        ];
        Self {
            data,
            shape,
            strides,
            distribution: None,
        }
    }
}

#[pymethods]
impl ScoringMatrix {
    /// Create a new scoring matrix.
    #[new]
    #[pyo3(signature = (values, background = None, *, protein = false))]
    #[allow(unused)]
    pub fn __init__<'py>(
        values: Bound<'py, PyDict>,
        background: Option<PyObject>,
        protein: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        macro_rules! run {
            ($alphabet:ty) => {{
                // extract the background from the method argument
                let bg = Python::with_gil(|py| {
                    if let Some(obj) = background {
                        if let Ok(d) = obj.extract::<Bound<PyDict>>(py) {
                            let p = dict_to_alphabet_array::<$alphabet>(d)?;
                            lightmotif::abc::Background::<$alphabet>::new(p).map_err(|_| {
                                PyValueError::new_err("Invalid background frequencies")
                            })
                        } else {
                            Err(PyTypeError::new_err("Invalid type for pseudocount"))
                        }
                    } else {
                        Ok(lightmotif::abc::Background::uniform())
                    }
                })?;
                // build data
                let mut data: Option<DenseMatrix<f32, <$alphabet as Alphabet>::K>> = None;
                for s in <$alphabet as Alphabet>::symbols() {
                    let key = String::from(s.as_char());
                    if let Some(res) = values.get_item(&key)? {
                        let column = res.downcast::<PyList>()?;
                        if data.is_none() {
                            data = Some(DenseMatrix::new(column.len()));
                        }
                        let matrix = data.as_mut().unwrap();
                        if matrix.rows() != column.len() {
                            return Err(PyValueError::new_err("Invalid number of rows"));
                        }
                        for (i, x) in column.iter().enumerate() {
                            matrix[i][s.as_index()] = x.extract::<f32>()?;
                        }
                    }
                }
                // create matrix
                match data {
                    None => Err(PyValueError::new_err("Invalid count matrix")),
                    Some(matrix) => Ok(Self::new(lightmotif::ScoringMatrix::<$alphabet>::new(
                        bg, matrix,
                    ))
                    .into()),
                }
            }};
        }

        if protein {
            run!(Protein)
        } else {
            run!(Dna)
        }
    }

    pub fn __eq__(&self, object: Bound<PyAny>) -> PyResult<bool> {
        if let Ok(other) = object.extract::<PyRef<Self>>() {
            Ok(self.data == other.data)
        } else {
            Ok(false)
        }
    }

    pub fn __len__(&self) -> usize {
        self.data.rows()
    }

    pub fn __getitem__<'py>(slf: PyRef<'py, Self>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let mut index_ = index;
        if index_ < 0 {
            index_ += slf.data.rows() as isize;
        }
        if index_ < 0 || (index_ as usize) >= slf.data.rows() {
            return Err(PyIndexError::new_err(index));
        }
        slf.data.get(index as usize).into_pyobject(py)
    }

    #[cfg(not(feature = "abi3"))]
    unsafe fn __getbuffer__(
        mut slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let data = slf.data.as_ptr();
        (*view).buf = data as *mut std::os::raw::c_void;
        (*view).len = -1;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<f32>() as isize;

        (*view).obj = slf.as_ptr();
        pyo3::ffi::Py_INCREF((*view).obj);
        let msg = std::ffi::CStr::from_bytes_with_nul(b"f\0").unwrap();
        (*view).format = msg.as_ptr() as *mut _;

        (*view).ndim = 2;
        (*view).shape = slf.shape.as_mut_ptr();
        (*view).strides = slf.strides.as_mut_ptr();
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    #[getter]
    pub fn score_distribution<'py>(slf: Bound<'py, Self>) -> PyResult<Py<ScoreDistribution>> {
        let py = slf.py();
        if slf.borrow().distribution.is_none() {
            let data = match &slf.borrow().data {
                ScoringMatrixData::Dna(dna) => {
                    ScoreDistributionData::from(dna.to_score_distribution())
                }
                ScoringMatrixData::Protein(prot) => {
                    ScoreDistributionData::from(prot.to_score_distribution())
                }
            };
            slf.borrow_mut().distribution = Some(Py::new(py, ScoreDistribution::from(data))?);
        }
        Ok(slf.borrow().distribution.as_ref().unwrap().clone_ref(py))
    }

    /// `bool`: `True` if the scoring matrix stores protein scores.
    #[getter]
    pub fn protein(&self) -> bool {
        matches!(self.data, ScoringMatrixData::Protein(_))
    }

    /// Calculate the PSSM score for all positions of the given sequence.
    ///
    /// Returns:
    ///     `~lightmotif.StripedScores`: The PSSM scores for every position
    ///     of the input sequence, stored into a striped matrix for fast
    ///     vectorized operations.
    ///
    /// Note:
    ///     This method uses the best implementation for the local platform,
    ///     prefering AVX2 if available.
    ///
    pub fn calculate(
        slf: Bound<'_, Self>,
        sequence: &mut StripedSequence,
    ) -> PyResult<StripedScores> {
        let pssm = &slf.borrow().data;
        let seq = &mut sequence.data;
        match (seq, pssm) {
            (StripedSequenceData::Dna(dna), ScoringMatrixData::Dna(pssm)) => {
                let pli = lightmotif::pli::Pipeline::dispatch();
                dna.configure(pssm);
                Ok(slf.py().allow_threads(|| pli.score(pssm, dna)).into())
            }
            (StripedSequenceData::Protein(prot), ScoringMatrixData::Protein(pssm)) => {
                let pli = lightmotif::pli::Pipeline::dispatch();
                prot.configure(pssm);
                Ok(slf.py().allow_threads(|| pli.score(pssm, prot)).into())
            }
            (_, _) => Err(PyValueError::new_err("alphabet mismatch")),
        }
    }

    /// Translate an absolute score to a P-value for this PSSM.
    #[pyo3(signature=(score, method="meme"))]
    pub fn pvalue(slf: Bound<'_, Self>, score: f64, method: &str) -> PyResult<f64> {
        let py = slf.py();
        match method {
            "tfmpvalue" => {
                #[cfg(feature = "tfmpvalue")]
                match &slf.borrow().data {
                    ScoringMatrixData::Dna(dna) => Ok(TfmPvalue::new(&dna).pvalue(score)),
                    ScoringMatrixData::Protein(prot) => Ok(TfmPvalue::new(&prot).pvalue(score)),
                }
                #[cfg(not(feature = "tfmpvalue"))]
                Err(PyRuntimeError::new_err(
                    "package compiled without `lightmotif-tfmpvalue`",
                ))
            }
            "meme" => {
                let dist = Self::score_distribution(slf)?;
                match &dist.bind(py).borrow().data {
                    ScoreDistributionData::Dna(dna) => Ok(dna.pvalue(score as f32)),
                    ScoreDistributionData::Protein(prot) => Ok(prot.pvalue(score as f32)),
                }
            }
            other => Err(PyValueError::new_err(format!(
                "invalid pvalue method: {:?}",
                other
            ))),
        }
    }

    pub fn max_score(slf: PyRef<'_, Self>) -> f32 {
        match &slf.data {
            ScoringMatrixData::Dna(dna) => dna.max_score(),
            ScoringMatrixData::Protein(prot) => prot.max_score(),
        }
    }

    /// Translate a P-value to an absolute score for this PSSM.
    #[pyo3(signature=(pvalue, method="meme"))]
    pub fn score(slf: Bound<'_, Self>, pvalue: f64, method: &str) -> PyResult<f64> {
        let py = slf.py();
        match method {
            "tfmpvalue" => {
                #[cfg(feature = "tfmpvalue")]
                match &slf.borrow().data {
                    ScoringMatrixData::Dna(dna) => Ok(TfmPvalue::new(&dna).score(pvalue)),
                    ScoringMatrixData::Protein(prot) => Ok(TfmPvalue::new(&prot).score(pvalue)),
                }
                #[cfg(not(feature = "tfmpvalue"))]
                Err(PyRuntimeError::new_err(
                    "package compiled without `lightmotif-tfmpvalue`",
                ))
            }
            "meme" => {
                let dist = Self::score_distribution(slf)?;
                match &dist.bind(py).borrow().data {
                    ScoreDistributionData::Dna(dna) => Ok(dna.score(pvalue) as f64),
                    ScoreDistributionData::Protein(prot) => Ok(prot.score(pvalue) as f64),
                }
            }
            other => Err(PyValueError::new_err(format!(
                "invalid pvalue method: {:?}",
                other
            ))),
        }
    }

    /// Compute the reverse complement of this scoring matrix.
    pub fn reverse_complement(slf: PyRef<'_, Self>) -> PyResult<ScoringMatrix> {
        match &slf.data {
            ScoringMatrixData::Dna(dna) => Ok(Self::from(ScoringMatrixData::from(
                dna.reverse_complement(),
            ))),
            ScoringMatrixData::Protein(_) => Err(PyRuntimeError::new_err(
                "cannot complement a protein sequence",
            )),
        }
    }
}

impl From<ScoringMatrixData> for ScoringMatrix {
    fn from(data: ScoringMatrixData) -> Self {
        Self::new(data)
    }
}

// --- ScoreDistribution -------------------------------------------------------

alphabet_data_enum!(
    #[derive(Clone, Debug)]
    enum ScoreDistributionData(lightmotif::pwm::dist::ScoreDistribution)
);

impl ScoreDistributionData {
    fn sf(&self) -> &[f64] {
        apply_self!(self, |x| x.sf())
    }
}

/// A matrix storing position-specific odds-ratio for a motif.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct ScoreDistribution {
    data: ScoreDistributionData,
}

impl From<ScoreDistributionData> for ScoreDistribution {
    fn from(data: ScoreDistributionData) -> Self {
        Self { data }
    }
}

#[pymethods]
impl ScoreDistribution {
    /// Get the underlying memory of the encoded sequence.
    #[cfg(not(feature = "abi3"))]
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let array: &[_] = slf.data.sf();
        (*view).buf = array.as_ptr() as *mut std::os::raw::c_void;
        (*view).len = (array.len() * std::mem::size_of::<f64>()) as isize;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<f64>() as isize;

        (*view).obj = slf.as_ptr();
        pyo3::ffi::Py_INCREF((*view).obj);
        let msg = std::ffi::CStr::from_bytes_with_nul(b"d\0").unwrap();
        (*view).format = msg.as_ptr() as *mut _;

        (*view).ndim = 1;
        (*view).shape = std::ptr::null_mut();
        (*view).strides = std::ptr::null_mut();
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }
}

// --- Scores ------------------------------------------------------------------

/// A striped matrix storing scores obtained with a scoring matrix.
#[pyclass(module = "lightmotif.lib", sequence)]
#[derive(Clone, Debug)]
pub struct StripedScores {
    scores: lightmotif::scores::StripedScores<f32>,
    shape: [Py_ssize_t; 2],
    strides: [Py_ssize_t; 2],
}

#[pymethods]
impl StripedScores {
    fn __len__(&self) -> usize {
        self.scores.max_index()
    }

    fn __getitem__(&self, index: isize) -> PyResult<f32> {
        if index < self.scores.max_index() as isize && index >= 0 {
            Ok(self.scores[index as usize])
        } else {
            Err(PyIndexError::new_err("list index out of range"))
        }
    }

    #[cfg(not(feature = "abi3"))]
    unsafe fn __getbuffer__(
        mut slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let data = slf.scores.matrix()[0].as_ptr();
        (*view).buf = data as *mut std::os::raw::c_void;
        (*view).len = -1;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<f32>() as isize;

        (*view).obj = slf.as_ptr();
        pyo3::ffi::Py_INCREF((*view).obj);
        let msg = std::ffi::CStr::from_bytes_with_nul(b"f\0").unwrap();
        (*view).format = msg.as_ptr() as *mut _;

        (*view).ndim = 2;
        (*view).shape = slf.shape.as_mut_ptr();
        (*view).strides = slf.strides.as_mut_ptr();
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    /// Return all positions with a score greater or equal to the threshold.
    ///
    /// Returns:
    ///     `list` of `int`: The indices of the position with a score greater or
    ///     equal to the given threshold. Note that the indices may or may not
    ///     be sorted, depending on the implementation.
    ///
    /// Note:
    ///     This method uses the best implementation for the local platform,
    ///     prefering AVX2 if available.
    ///
    pub fn threshold(slf: PyRef<'_, Self>, threshold: f32) -> Vec<usize> {
        let scores = &slf.scores;
        slf.py().allow_threads(|| scores.threshold(threshold))
    }

    /// Return the maximum score, if the score matrix is not empty.
    ///
    /// Returns:
    ///     `float` or `None`: The maximum score, if any.
    ///
    /// Note:
    ///     This method uses the best implementation for the local platform,
    ///     prefering AVX2 if available.
    ///
    pub fn max(slf: PyRef<'_, Self>) -> Option<f32> {
        let scores = &slf.scores;
        slf.py().allow_threads(|| scores.max())
    }

    /// Return the position of the maximum score, if the score matrix is not empty.
    ///
    /// Returns:
    ///     `int` or `None`: The position of the maximum score, if any.
    ///
    /// Note:
    ///     This method uses the best implementation for the local platform,
    ///     prefering AVX2 if available.
    ///
    pub fn argmax(slf: PyRef<'_, Self>) -> Option<usize> {
        let scores = &slf.scores;
        slf.py().allow_threads(|| scores.argmax())
    }
}

impl From<lightmotif::scores::StripedScores<f32>> for StripedScores {
    fn from(scores: lightmotif::scores::StripedScores<f32>) -> Self {
        // assert_eq!(scores.range().start, 0);
        // extract the matrix shape
        let cols = scores.matrix().columns();
        let rows = scores.matrix().rows();
        // record the matrix shape as a Fortran buffer
        let shape = [cols as Py_ssize_t, rows as Py_ssize_t];
        let strides = [
            std::mem::size_of::<f32>() as Py_ssize_t,
            (scores.matrix().stride() * std::mem::size_of::<f32>()) as Py_ssize_t,
        ];
        // mask the remaining positions that are outside the sequence length
        // let length = scores.len();
        // let data = scores.matrix_mut();
        // for i in length..rows * cols {
        //     let row = i % rows;
        //     let col = i / rows;
        //     data[row][col] = -f32::INFINITY;
        // }
        // return a Python object implementing the buffer protocol
        Self {
            scores,
            shape,
            strides,
        }
    }
}

// --- Motif -------------------------------------------------------------------

/// A base object storing information about a motif.
#[pyclass(module = "lightmotif.lib", subclass)]
#[derive(Debug)]
pub struct Motif {
    #[pyo3(get)]
    /// `CountMatrix` or `None`: The count matrix for this motif.
    ///
    /// This may be `None` if the motif was loaded from a format that does
    /// not store counts but frequencies, such as the ``uniprobe`` format.
    counts: Option<Py<CountMatrix>>,
    #[pyo3(get)]
    /// `WeightMatrix`: The weight matrix for this motif.
    pwm: Py<WeightMatrix>,
    #[pyo3(get)]
    /// `ScoringMatrix`: The scoring matrix for this motif.
    pssm: Py<ScoringMatrix>,
    #[pyo3(get)]
    /// `str` or `None`: An optional name for the motif.
    name: Option<String>,
}

impl Motif {
    fn from_counts<A>(py: Python, counts: lightmotif::pwm::CountMatrix<A>) -> PyResult<Self>
    where
        A: Alphabet,
        CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let weights = counts.to_freq(0.0).to_weight(None);
        let scoring = weights.to_scoring();
        Ok(Motif {
            counts: Some(Py::new(py, CountMatrix::new(counts))?),
            pwm: Py::new(py, WeightMatrix::new(weights))?,
            pssm: Py::new(py, ScoringMatrix::new(scoring))?,
            name: None,
        })
    }

    fn from_weights<A>(py: Python, weights: lightmotif::pwm::WeightMatrix<A>) -> PyResult<Self>
    where
        A: Alphabet,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let scoring = weights.to_scoring();
        Ok(Motif {
            counts: None,
            pwm: Py::new(py, WeightMatrix::new(weights))?,
            pssm: Py::new(py, ScoringMatrix::new(scoring))?,
            name: None,
        })
    }
}

#[pymethods]
impl Motif {
    /// `bool`: `True` if the motif is a protein motif.
    #[getter]
    pub fn protein<'py>(slf: PyRef<'py, Self>) -> bool {
        let py = slf.py();
        matches!(
            slf.pssm.bind(py).borrow().data,
            ScoringMatrixData::Protein(_)
        )
    }
}

// --- Scanner -----------------------------------------------------------------

/// A fast scanner for identifying high scoring positions in a sequence.
///
/// This class internally uses a discretized version of the matrix to
/// identify candidate positions, and then rescores blocks with the full
/// algorithm only if needed. Using a `Scanner` is likely faster than
/// using the `~ScoringMatrix.calculate` method for rare sites or high
/// thresholds.
///
/// Note:
///     This algorithm is only available for DNA motifs because of
///     implementation requirements.
///
#[pyclass(module = "lightmotif.lib")]
#[derive(Debug)]
pub struct Scanner {
    #[allow(unused)]
    pssm: Py<ScoringMatrix>,
    #[allow(unused)]
    sequence: Py<StripedSequence>,
    data: lightmotif::scan::Scanner<
        'static,
        Dna,
        &'static lightmotif::pwm::ScoringMatrix<Dna>,
        &'static lightmotif::seq::StripedSequence<Dna>,
    >,
}

#[pymethods]
impl Scanner {
    #[new]
    #[pyo3(signature = (pssm, sequence, threshold = 0.0, block_size = 256))]
    fn __init__<'py>(
        pssm: Bound<'py, ScoringMatrix>,
        sequence: Bound<'py, StripedSequence>,
        threshold: f32,
        block_size: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        match (
            &pssm.try_borrow()?.data,
            &mut sequence.try_borrow_mut()?.data,
        ) {
            (ScoringMatrixData::Dna(p), StripedSequenceData::Dna(s)) => {
                s.configure(&p);
                // transmute (!!!!!) the scanner so that its lifetime is 'static
                // (i.e. the reference to the PSSM and sequence never expire),
                // which is possible because we are building a self-referential
                // struct and self.scanner.next() will only be called with
                // the GIL held...
                let scanner = unsafe {
                    let mut scanner = lightmotif::scan::Scanner::<Dna, _, _>::new(p, s);
                    scanner.threshold(threshold);
                    scanner.block_size(block_size);
                    std::mem::transmute(scanner)
                };
                Ok(PyClassInitializer::from(Scanner {
                    data: scanner,
                    pssm: pssm.unbind(),
                    sequence: sequence.unbind(),
                }))
            }
            (ScoringMatrixData::Protein(_), StripedSequenceData::Protein(_)) => {
                Err(PyValueError::new_err("protein scanner is not supported"))
            }
            (_, _) => Err(PyValueError::new_err("alphabet mismatch")),
        }
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<Hit> {
        slf.data.next().map(Hit::from)
    }
}

/// A hit found by a `~lightmotif.Scanner`.
#[pyclass(module = "lightmotif.lib")]
#[derive(Debug)]
pub struct Hit {
    #[pyo3(get)]
    /// `int`: The index of the hit, zero-based.
    position: usize,
    #[pyo3(get)]
    /// `float`: The score of the scoring matrix at the hit position.
    score: f32,
}

impl From<lightmotif::scan::Hit> for Hit {
    fn from(value: lightmotif::scan::Hit) -> Self {
        Hit {
            position: value.position(),
            score: value.score(),
        }
    }
}

// --- Functions ---------------------------------------------------------------

/// Create a new motif from an iterable of sequences.
///
/// All sequences must have the same length, and must contain only valid
/// alphabet symbols (*ATGCN* for nucleotides, *ACDEFGHIKLMNPQRSTVWYX*
/// for proteins).
///
/// Arguments:
///     sequences (iterable of `str`): The sequences to use to build the
///         count matrix for the motif.
///     protein (`bool`): Pass `True` to build a protein motif. Defaults
///         to `False`.
///
/// Example:
///     >>> sequences = ["TATAAT", "TATAAA", "TATATT", "TATAAT"]
///     >>> motif = lightmotif.create(sequences)
///
/// Returns:
///     `~lightmotif.Motif`: The motif corresponding to the given sequences.
///
/// Raises:
///     `ValueError`: When any of the sequences contain an invalid character,
///         or when the sequence lengths are not consistent.
///
#[pyfunction]
#[pyo3(signature = (sequences, *, protein = false, name = None))]
pub fn create(sequences: Bound<PyAny>, protein: bool, name: Option<String>) -> PyResult<Motif> {
    let py = sequences.py();
    macro_rules! run {
        ($alphabet:ty) => {{
            let mut encoded = Vec::new();
            for seq in sequences.try_iter()? {
                let s = seq?.extract::<Bound<PyString>>()?;
                let sequence = s.to_cow()?;
                let x = py
                    .allow_threads(|| lightmotif::EncodedSequence::<$alphabet>::encode(&*sequence))
                    .map_err(|_| PyValueError::new_err("Invalid symbol in sequence"))?;
                encoded.push(x);
            }

            let data = lightmotif::CountMatrix::from_sequences(encoded)
                .map_err(|_| PyValueError::new_err("Inconsistent sequence length"))?;
            let weights = data.to_freq(0.0).to_weight(None);
            let scoring = weights.to_scoring();

            Ok(Motif {
                counts: Some(Py::new(py, CountMatrix::new(data))?),
                pwm: Py::new(py, WeightMatrix::new(weights))?,
                pssm: Py::new(py, ScoringMatrix::new(scoring))?,
                name,
            })
        }};
    }

    if protein {
        run!(Protein)
    } else {
        run!(Dna)
    }
}

/// Encode and stripe a text sequence.
///
/// Arguments:
///     sequence (`str`): The sequence to encode and stripe.
///     protein (`bool`): Pass `True` to treat the sequence as a protein
///         sequence.
///
/// Returns:
///     `~lightmotif.StripedSequence`: A striped sequence containing the
///     sequence data.
///
/// Raises:
///     `ValueError`: When the sequences contains an invalid character.
///
#[pyfunction]
#[pyo3(signature = (sequence, *, protein=false))]
pub fn stripe(sequence: Bound<PyString>, protein: bool) -> PyResult<StripedSequence> {
    let py = sequence.py();
    let encoded = EncodedSequence::__init__(sequence, protein).and_then(|e| Py::new(py, e))?;
    let striped = encoded.borrow(py).stripe();
    Ok(striped)
}

/// Scan a sequence using a fast scanner implementation to identify hits.
///
/// See `~lightmotif.Scanner` for more information.
///
/// Arguments:
///     pssm (`~lightmotif.ScoringMatrix`): The scoring matrix to use to
///         score the sequence with.
///     sequence (`~lightmotif.StripedSequence`): The striped sequence to
///         process with the scanner. Longer sequences benifit more from the
///         scanning implementation.
///     threshold (`float`): The score threshold to use for filtering hits.
///         Use `ScoringMatrix.score` to compute a score threshold from a
///         target *p-value*. A higher threshold will result in less
///         candidate hits and total runtime.
///     
/// Returns:
///     `~lightmotif.Scanner`: The scanner for finding candidate hits in
///     the target sequence.
///  
/// Raises:
///     `ValueError`: When either ``pssm`` or ``sequence`` are not using
///         the DNA alphabet.
///
/// Note:
///     This algorithm is only available for DNA motifs because of
///     implementation requirements.  
///
#[pyfunction]
#[pyo3(signature = (pssm, sequence, *, threshold = 0.0, block_size = 256))]
pub fn scan<'py>(
    pssm: Bound<'py, ScoringMatrix>,
    sequence: Bound<'py, StripedSequence>,
    threshold: f32,
    block_size: usize,
) -> PyResult<Bound<'py, Scanner>> {
    let py = pssm.py();
    Bound::new(
        py,
        Scanner::__init__(pssm, sequence, threshold, block_size)?,
    )
}

// --- Module ------------------------------------------------------------------

/// PyO3 bindings to ``lightmotif``, a library for fast PWM motif scanning.
///
/// The API is similar to the `Bio.motifs` module from Biopython on purpose.
///
/// Note:
///     Arbitrary alphabets cannot be configured, as ``lightmotif`` uses
///     constant definitions of alphabets that cannot be changed at runtime.
///     The different sequences are treated as nucleotide sequences. To use
///     a protein sequence instead, most classes and functions
///     have a ``protein`` keyword `True`::
///
///         >>> sequences = ["PILFFRLK", "KDMLKEYL", "PFRLTHKL"]
///         >>> lightmotif.create(sequences)
///         Traceback (most recent call last):
///           ...
///         ValueError: Invalid symbol in sequence
///         >>> lightmotif.create(sequences, protein=True)
///         <lightmotif.lib.Motif object at ...>
///
#[pymodule]
#[pyo3(name = "lib")]
pub fn init<'py>(_py: Python<'py>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__package__", "lightmotif")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS").replace(':', "\n"))?;

    #[cfg(feature = "abi3")]
    m.add("LIMITED_API", true)?;
    #[cfg(not(feature = "abi3"))]
    m.add("LIMITED_API", false)?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    m.add("AVX2_SUPPORTED", std::is_x86_feature_detected!("avx2"))?;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    m.add("AVX2_SUPPORTED", false)?;

    m.add_class::<EncodedSequence>()?;
    m.add_class::<StripedSequence>()?;

    m.add_class::<CountMatrix>()?;
    m.add_class::<WeightMatrix>()?;
    m.add_class::<ScoringMatrix>()?;

    m.add_class::<ScoreDistribution>()?;

    m.add_class::<StripedScores>()?;

    m.add_class::<Motif>()?;
    m.add_class::<io::TransfacMotif>()?;
    m.add_class::<io::JasparMotif>()?;
    m.add_class::<io::UniprobeMotif>()?;
    m.add_class::<io::MemeMotif>()?;

    m.add_class::<Scanner>()?;
    m.add_class::<Hit>()?;

    m.add_class::<io::Loader>()?;

    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_function(wrap_pyfunction!(stripe, m)?)?;
    m.add_function(wrap_pyfunction!(io::load, m)?)?;

    Ok(())
}
