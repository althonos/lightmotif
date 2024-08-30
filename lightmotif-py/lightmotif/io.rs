use std::io::BufRead;
use std::sync::Arc;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::abc::Protein;
use lightmotif_io::error::Error;

use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use super::CountMatrixData;
use super::Motif;
use super::ScoringMatrixData;
use super::WeightMatrixData;

mod pyfile {
    use std::io::Error as IoError;
    use std::io::Read;
    use std::sync::Mutex;

    use pyo3::exceptions::PyOSError;
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    // ---------------------------------------------------------------------------

    #[macro_export]
    macro_rules! transmute_file_error {
        ($self:ident, $e:ident, $msg:expr, $py:expr) => {{
            // Attempt to transmute the Python OSError to an actual
            // Rust `std::io::Error` using `from_raw_os_error`.
            if $e.is_instance_of::<PyOSError>($py) {
                if let Ok(code) = &$e.value_bound($py).getattr("errno") {
                    if let Ok(n) = code.extract::<i32>() {
                        return Err(IoError::from_raw_os_error(n));
                    }
                }
            }

            // if the conversion is not possible for any reason we fail
            // silently, wrapping the Python error, and returning a
            // generic Rust error instead.
            $e.restore($py);
            Err(IoError::new(std::io::ErrorKind::Other, $msg))
        }};
    }

    // ---------------------------------------------------------------------------

    /// A wrapper for a Python file that can outlive the GIL.
    pub struct PyFileRead {
        file: Mutex<PyObject>,
    }

    impl PyFileRead {
        pub fn from_ref(file: &Bound<PyAny>) -> PyResult<PyFileRead> {
            let res = file.call_method1("read", (0,))?;
            if res.downcast::<PyBytes>().is_ok() {
                Ok(PyFileRead {
                    file: Mutex::new(file.to_object(file.py())),
                })
            } else {
                let ty = res.get_type().name()?.to_string();
                Err(PyTypeError::new_err(format!(
                    "expected bytes, found {}",
                    ty
                )))
            }
        }
    }

    impl Read for PyFileRead {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError> {
            Python::with_gil(|py| {
                let file = self.file.lock().expect("failed to lock file");
                match file
                    .to_object(py)
                    .call_method1(py, pyo3::intern!(py, "read"), (buf.len(),))
                {
                    Ok(obj) => {
                        // Check `fh.read` returned bytes, else raise a `TypeError`.
                        if let Ok(bytes) = obj.downcast_bound::<PyBytes>(py) {
                            let b = bytes.as_bytes();
                            (&mut buf[..b.len()]).copy_from_slice(b);
                            Ok(b.len())
                        } else {
                            let ty = obj.bind(py).get_type().name()?.to_string();
                            let msg = format!("expected bytes, found {}", ty);
                            PyTypeError::new_err(msg).restore(py);
                            Err(IoError::new(
                                std::io::ErrorKind::Other,
                                "fh.read did not return bytes",
                            ))
                        }
                    }
                    Err(e) => transmute_file_error!(self, e, "read method failed", py),
                }
            })
        }
    }
}

fn convert_error(error: Error) -> PyErr {
    match error {
        Error::InvalidData => PyValueError::new_err("invalid data"),
        Error::Io(err) => Arc::into_inner(err)
            .map(PyErr::from)
            .unwrap_or_else(|| PyOSError::new_err("unknown error")),
        Error::Nom(err) => PyValueError::new_err(format!("failed to parse input: {}", err)),
    }
}

fn convert_jaspar(record: Result<lightmotif_io::jaspar::Record, Error>) -> PyResult<Motif> {
    let record = record.map_err(convert_error)?;
    let name = record.id().to_string();
    let counts = record.into();
    let mut motif = Python::with_gil(|py| Motif::from_counts(py, counts))?;
    motif.name = Some(name);
    Ok(motif)
}

fn convert_jaspar16<A>(record: Result<lightmotif_io::jaspar16::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.map_err(convert_error)?;
    let name = record.id().to_string();
    let counts = record.into_matrix();
    let mut motif = Python::with_gil(|py| Motif::from_counts(py, counts))?;
    motif.name = Some(name);
    Ok(motif)
}

fn convert_transfac<A>(record: Result<lightmotif_io::transfac::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.map_err(convert_error)?;
    let name = record.accession().or(record.id()).map(String::from);
    let counts = record
        .to_counts()
        .ok_or_else(|| PyValueError::new_err("invalid count matrix"))?;
    let mut motif = Python::with_gil(|py| Motif::from_counts(py, counts))?;
    motif.name = name;
    Ok(motif)
}

fn convert_uniprobe<A>(record: Result<lightmotif_io::uniprobe::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.map_err(convert_error)?;
    let name = record.id().to_string();
    let freqs = record.into_matrix();
    let weights = freqs.to_weight(None);
    let mut motif = Python::with_gil(|py| Motif::from_weights(py, weights))?;
    motif.name = Some(name);
    Ok(motif)
}

#[pyclass(module = "lightmotif.lib")]
pub struct Loader {
    reader: Box<dyn Iterator<Item = PyResult<Motif>> + Send>,
}

#[pymethods]
impl Loader {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyResult<Motif>> {
        slf.reader.next()
    }
}

/// Load the motifs contained in a file.
#[pyfunction]
#[pyo3(signature = (file, format="jaspar", *, protein=false))]
pub fn load(file: Bound<PyAny>, format: &str, protein: bool) -> PyResult<Loader> {
    let b: Box<dyn BufRead + Send> = if let Ok(s) = file.downcast::<PyString>() {
        std::fs::File::open(s.to_str()?)
            .map(std::io::BufReader::new)
            .map(Box::new)?
    } else {
        pyfile::PyFileRead::from_ref(&file)
            .map(std::io::BufReader::new)
            .map(Box::new)?
    };
    let reader: Box<dyn Iterator<Item = PyResult<Motif>> + Send> = match format {
        "jaspar" if protein => {
            return Err(PyValueError::new_err(
                "cannot read protein motifs from JASPAR format",
            ))
        }
        "jaspar16" if protein => Box::new(
            lightmotif_io::jaspar16::read::<_, Protein>(b).map(convert_jaspar16::<Protein>),
        ),
        "transfac" if protein => Box::new(
            lightmotif_io::transfac::read::<_, Protein>(b).map(convert_transfac::<Protein>),
        ),
        "uniprobe" if protein => Box::new(
            lightmotif_io::uniprobe::read::<_, Protein>(b).map(convert_uniprobe::<Protein>),
        ),
        "jaspar" => Box::new(lightmotif_io::jaspar::read(b).map(convert_jaspar)),
        "jaspar16" => {
            Box::new(lightmotif_io::jaspar16::read::<_, Dna>(b).map(convert_jaspar16::<Dna>))
        }
        "transfac" => {
            Box::new(lightmotif_io::transfac::read::<_, Dna>(b).map(convert_transfac::<Dna>))
        }
        "uniprobe" => {
            Box::new(lightmotif_io::uniprobe::read::<_, Dna>(b).map(convert_uniprobe::<Dna>))
        }
        _ => return Err(PyValueError::new_err(format!("invalid format: {}", format))),
    };
    Ok(Loader { reader })
}
