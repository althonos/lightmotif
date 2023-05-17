#![doc = include_str!("../README.md")]

extern crate generic_array;
extern crate lightmotif;
extern crate pyo3;

use lightmotif::abc::Symbol;
use lightmotif::num::Unsigned;
use lightmotif::pli::platform::Backend;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;

use pyo3::exceptions::PyBufferError;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_ssize_t;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct StripedSequence {
    data: lightmotif::seq::StripedSequence<lightmotif::Dna, C>,
}

// --- CountMatrix -------------------------------------------------------------

#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct CountMatrix {
    data: lightmotif::CountMatrix<lightmotif::Dna>,
}

#[pymethods]
impl CountMatrix {
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

// --- FrequencyMatrix ---------------------------------------------------------

#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    data: lightmotif::pwm::WeightMatrix<lightmotif::Dna>,
}

#[pymethods]
impl WeightMatrix {
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

#[pyclass(module = "lightmotif.lib")]
#[derive(Clone, Debug)]
pub struct ScoringMatrix {
    data: lightmotif::ScoringMatrix<lightmotif::Dna>,
}

#[pymethods]
impl ScoringMatrix {
    /// Return the PSSM score for all positions of the given sequence.
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
            (cols.next_power_of_two() * std::mem::size_of::<f32>()) as Py_ssize_t,
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

    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(stripe, m)?)?;

    Ok(())
}
