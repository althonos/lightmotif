#![doc = include_str!("../README.md")]

extern crate generic_array;
extern crate lightmotif;
extern crate pyo3;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::Unsigned;
use lightmotif::pli::platform::Backend;
use lightmotif::pli::Maximum;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::Threshold;

use pyo3::exceptions::PyBufferError;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_ssize_t;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyString;
use pyo3::AsPyPointer;

use generic_array::GenericArray;

// --- Compile-time constants --------------------------------------------------

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
type C = <lightmotif::pli::platform::Neon as Backend>::LANES;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type C = <lightmotif::pli::platform::Avx2 as Backend>::LANES;
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64",
)))]
type C = typenum::consts::U1;

// --- Helpers -----------------------------------------------------------------

fn dict_to_alphabet_array<'py, A: lightmotif::Alphabet>(
    d: &'py PyDict,
) -> PyResult<GenericArray<f32, A::K>> {
    let mut p = std::iter::repeat(0.0)
        .take(A::K::USIZE)
        .collect::<GenericArray<f32, A::K>>();
    for (k, v) in d.iter() {
        let s = k.extract::<&PyString>()?.to_str()?;
        if s.len() != 1 {
            return Err(PyValueError::new_err((
                "Invalid key for pseudocount:",
                s.to_string(),
            )));
        }
        let x = s.chars().next().unwrap();
        let symbol = <A as lightmotif::Alphabet>::Symbol::from_char(x)
            .map_err(|_| PyValueError::new_err(("Invalid key for pseudocount:", x)))?;
        let value = v.extract::<f32>()?;
        p[symbol.as_index()] = value;
    }
    Ok(p)
}

// --- EncodedSequence ---------------------------------------------------------

/// A biological sequence encoded as digits.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct EncodedSequence {
    data: lightmotif::seq::EncodedSequence<lightmotif::Dna>,
}

#[pymethods]
impl EncodedSequence {
    /// Encode a sequence with the given alphabet.
    #[new]
    pub fn __init__(sequence: &PyString) -> PyResult<PyClassInitializer<Self>> {
        let seq = sequence.to_str()?;
        let data = lightmotif::EncodedSequence::encode(&seq).map_err(
            |lightmotif::err::InvalidSymbol(x)| {
                PyValueError::new_err(format!("Invalid symbol in input: {}", x))
            },
        )?;
        Ok(EncodedSequence { data }.into())
    }

    /// Create a copy of this sequence.
    pub fn copy(&self) -> EncodedSequence {
        self.clone()
    }

    /// Convert this sequence into a striped matrix.
    pub fn stripe(&self) -> StripedSequence {
        StripedSequence {
            data: self.data.to_striped(),
        }
    }
}

// --- StripedSequence ---------------------------------------------------------

/// An encoded biological sequence stored in a column-major matrix.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct StripedSequence {
    data: lightmotif::seq::StripedSequence<lightmotif::Dna, C>,
}

// --- CountMatrix -------------------------------------------------------------

/// A matrix storing the count of a motif letters at each position.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct CountMatrix {
    data: lightmotif::CountMatrix<lightmotif::Dna>,
}

#[pymethods]
impl CountMatrix {
    /// Create a new count matrix.
    #[new]
    pub fn __init__(_alphabet: &PyString, values: &PyDict) -> PyResult<PyClassInitializer<Self>> {
        let mut data: Option<DenseMatrix<u32, <lightmotif::Dna as Alphabet>::K>> = None;
        for s in lightmotif::Dna::symbols() {
            let key = String::from(s.as_char());
            if let Some(res) = values.get_item(&key) {
                let column = res.downcast::<PyList>()?;
                if data.is_none() {
                    data = Some(DenseMatrix::new(column.len()));
                }
                let matrix = data.as_mut().unwrap();
                if matrix.rows() != column.len() {
                    return Err(PyValueError::new_err("Invalid number of rows"));
                }
                for (i, x) in column.iter().enumerate() {
                    matrix[i][s.as_index()] = x.extract::<u32>()?;
                }
            }
        }

        match data {
            None => Err(PyValueError::new_err("Invalid count matrix")),
            Some(matrix) => match lightmotif::CountMatrix::new(matrix) {
                Ok(counts) => Ok(Self::from(counts).into()),
                Err(_) => Err(PyValueError::new_err("Inconsistent rows in count matrix")),
            },
        }
    }

    pub fn __eq__(&self, object: &PyAny) -> PyResult<bool> {
        let py = object.py();
        if let Ok(other) = object.extract::<Py<Self>>() {
            Ok(self.data == other.borrow(py).data)
        } else {
            Ok(false)
        }
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
    pub fn normalize(&self, pseudocount: Option<PyObject>) -> PyResult<WeightMatrix> {
        let pseudo = Python::with_gil(|py| {
            if let Some(obj) = pseudocount {
                if let Ok(x) = obj.extract::<f32>(py) {
                    Ok(lightmotif::abc::Pseudocounts::from(x))
                } else if let Ok(d) = obj.extract::<&PyDict>(py) {
                    let p = dict_to_alphabet_array::<lightmotif::Dna>(d)?;
                    Ok(lightmotif::abc::Pseudocounts::from(p))
                } else {
                    Err(PyTypeError::new_err("Invalid type for pseudocount"))
                }
            } else {
                Ok(lightmotif::abc::Pseudocounts::default())
            }
        })?;
        let data = self.data.to_freq(pseudo).to_weight(None);
        Ok(WeightMatrix { data })
    }
}

impl From<lightmotif::CountMatrix<lightmotif::Dna>> for CountMatrix {
    fn from(data: lightmotif::CountMatrix<lightmotif::Dna>) -> Self {
        Self { data }
    }
}

// --- WeightMatrix ------------------------------------------------------------

/// A matrix storing position-specific odds-ratio for a motif.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    data: lightmotif::pwm::WeightMatrix<lightmotif::Dna>,
}

#[pymethods]
impl WeightMatrix {
    pub fn __eq__(&self, object: &PyAny) -> PyResult<bool> {
        let py = object.py();
        if let Ok(other) = object.extract::<Py<Self>>() {
            Ok(self.data == other.borrow(py).data)
        } else {
            Ok(false)
        }
    }

    /// Log-scale this weight matrix to obtain a position-specific scoring matrix.
    ///
    /// Arguments:
    ///     background (`dict` or `None`): The background frequencies to use for
    ///         rescaling the weight matrix before computing log-odds-ratio. If
    ///         `None` given, uniform background frequencies will be used.
    ///
    pub fn log_odds(&self, background: Option<PyObject>) -> PyResult<ScoringMatrix> {
        // extract the background from the method argument
        let bg = Python::with_gil(|py| {
            if let Some(obj) = background {
                if let Ok(d) = obj.extract::<&PyDict>(py) {
                    let p = dict_to_alphabet_array::<lightmotif::Dna>(d)?;
                    lightmotif::abc::Background::new(p)
                        .map_err(|_| PyValueError::new_err("Invalid background frequencies"))
                } else {
                    Err(PyTypeError::new_err("Invalid type for pseudocount"))
                }
            } else {
                Ok(lightmotif::abc::Background::uniform())
            }
        })?;
        // rescale if backgrounds do not match
        let pwm = match bg.frequencies() != self.data.background().frequencies() {
            false => self.data.rescale(bg),
            true => self.data.clone(),
        };
        Ok(ScoringMatrix { data: pwm.into() })
    }
}

impl From<lightmotif::WeightMatrix<lightmotif::Dna>> for WeightMatrix {
    fn from(data: lightmotif::WeightMatrix<lightmotif::Dna>) -> Self {
        Self { data }
    }
}

// --- ScoringMatrix -----------------------------------------------------------

/// A matrix storing position-specific odds-ratio for a motif.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct ScoringMatrix {
    data: lightmotif::ScoringMatrix<lightmotif::Dna>,
}

#[pymethods]
impl ScoringMatrix {
    pub fn __eq__(&self, object: &PyAny) -> PyResult<bool> {
        let py = object.py();
        if let Ok(other) = object.extract::<Py<Self>>() {
            Ok(self.data == other.borrow(py).data)
        } else {
            Ok(false)
        }
    }

    /// Compute the PSSM score for all positions of the given sequence.
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
        slf: PyRef<'_, Self>,
        sequence: &mut StripedSequence,
    ) -> PyResult<StripedScores> {
        let pssm = &slf.data;
        let seq = &mut sequence.data;
        seq.configure(pssm);

        let scores = slf.py().allow_threads(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::avx2() {
                return pli.score(seq, pssm);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::sse2() {
                return pli.score(seq, pssm);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            if let Ok(pli) = Pipeline::neon() {
                return pli.score(seq, pssm);
            }
            Pipeline::generic().score(seq, pssm)
        });

        Ok(StripedScores::from(scores))
    }
}

impl From<lightmotif::ScoringMatrix<lightmotif::Dna>> for ScoringMatrix {
    fn from(data: lightmotif::ScoringMatrix<lightmotif::Dna>) -> Self {
        Self { data }
    }
}

// --- Scores ------------------------------------------------------------------

/// A striped matrix storing scores obtained with a scoring matrix.
#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct StripedScores {
    scores: lightmotif::pli::StripedScores<C>,
    shape: [Py_ssize_t; 2],
    strides: [Py_ssize_t; 2],
}

#[pymethods]
impl StripedScores {
    fn __len__(&self) -> usize {
        self.scores.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<f32> {
        if index < self.scores.len() as isize && index >= 0 {
            Ok(self.scores[index as usize])
        } else {
            Err(PyIndexError::new_err("list index out of range"))
        }
    }

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

        (*view).obj = pyo3::ffi::_Py_NewRef(slf.as_ptr());

        let data = slf.scores.matrix()[0].as_ptr();

        (*view).buf = data as *mut std::os::raw::c_void;
        (*view).len = slf.scores.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<f32>() as isize;

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
    pub fn threshold(slf: PyRef<'_, Self>, threshold: f32) -> PyResult<Vec<usize>> {
        let scores = &slf.scores;
        let indices = slf.py().allow_threads(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::avx2() {
                return pli.threshold(scores, threshold);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::sse2() {
                return pli.threshold(scores, threshold);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::neon() {
                return pli.threshold(scores, threshold);
            }
            Pipeline::<Dna, _>::generic().threshold(scores, threshold)
        });
        Ok(indices)
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
    pub fn max(slf: PyRef<'_, Self>) -> PyResult<Option<f32>> {
        let scores = &slf.scores;
        let indices = slf.py().allow_threads(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::avx2() {
                return pli.max(scores);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::sse2() {
                return pli.max(scores);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::neon() {
                return pli.max(scores);
            }
            Pipeline::<Dna, _>::generic().max(scores)
        });
        Ok(indices)
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
    pub fn argmax(slf: PyRef<'_, Self>) -> PyResult<Option<usize>> {
        let scores = &slf.scores;
        let indices = slf.py().allow_threads(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::avx2() {
                return pli.argmax(scores);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::sse2() {
                return pli.argmax(scores);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            if let Ok(pli) = Pipeline::<Dna, _>::neon() {
                return pli.argmax(scores);
            }
            Pipeline::<Dna, _>::generic().argmax(scores)
        });
        Ok(indices)
    }
}

impl From<lightmotif::pli::StripedScores<C>> for StripedScores {
    fn from(mut scores: lightmotif::pli::StripedScores<C>) -> Self {
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
        let length = scores.len();
        let data = scores.matrix_mut();
        for i in length..rows * cols {
            let row = i % rows;
            let col = i / rows;
            data[row][col] = -f32::INFINITY;
        }
        // return a Python object implementing the buffer protocol
        Self {
            scores,
            shape,
            strides,
        }
    }
}

// --- Motif -------------------------------------------------------------------

#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct Motif {
    #[pyo3(get)]
    counts: Py<CountMatrix>,
    #[pyo3(get)]
    pwm: Py<WeightMatrix>,
    #[pyo3(get)]
    pssm: Py<ScoringMatrix>,
}

// --- Module ------------------------------------------------------------------

/// Create a new motif from an iterable of sequences.
///
/// All sequences must have the same length, and must contain only valid DNA
/// symbols (*A*, *T*, *G*, *C*, or *N* as a wildcard).
///
/// Example:
///     >>> sequences = ["TATAAT", "TATAAA", "TATATT", "TATAAT"]
///     >>> motif = lightmotif.create(sequences)
///
/// Returns:
///     `~lightmotif.Motif`: The motif corresponding to the given sequences.
///
#[pyfunction]
pub fn create<'py>(sequences: &'py PyAny) -> PyResult<Motif> {
    let py = sequences.py();

    let mut encoded = Vec::new();
    for seq in sequences.iter()? {
        let s = seq?.extract::<&PyString>()?.to_str()?;
        let x = py
            .allow_threads(|| lightmotif::EncodedSequence::encode(&s))
            .map_err(|_| PyValueError::new_err("Invalid symbol in sequence"))?;
        encoded.push(x);
    }

    let data = lightmotif::CountMatrix::from_sequences(encoded)
        .map_err(|_| PyValueError::new_err("Inconsistent sequence length"))?;
    let weights = data.to_freq(0.0).to_weight(None);
    let scoring = weights.to_scoring();

    Ok(Motif {
        counts: Py::new(py, CountMatrix::from(data))?,
        pwm: Py::new(py, WeightMatrix::from(weights))?,
        pssm: Py::new(py, ScoringMatrix::from(scoring))?,
    })
}

/// Encode and stripe a text sequence.
#[pyfunction]
pub fn stripe<'py>(sequence: &'py PyAny) -> PyResult<StripedSequence> {
    let py = sequence.py();
    let s = sequence.extract::<&PyString>()?;
    let encoded = EncodedSequence::__init__(s).and_then(|e| Py::new(py, e))?;
    let striped = encoded.borrow(py).stripe();
    Ok(striped)
}

/// PyO3 bindings to ``lightmotif``, a library for fast PWM motif scanning.
///
/// The API is similar to the `Bio.motifs` module from Biopython on purpose.
#[pymodule]
#[pyo3(name = "lib")]
pub fn init(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__package__", "lightmotif")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS").replace(':', "\n"))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    m.add("AVX2_SUPPORTED", std::is_x86_feature_detected!("avx2"))?;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    m.add("AVX2_SUPPORTED", false)?;

    m.add_class::<EncodedSequence>()?;
    m.add_class::<StripedSequence>()?;

    m.add_class::<CountMatrix>()?;
    m.add_class::<WeightMatrix>()?;
    m.add_class::<ScoringMatrix>()?;

    m.add_class::<StripedScores>()?;

    m.add_class::<Motif>()?;

    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(stripe, m)?)?;

    Ok(())
}
