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

use super::pyfile::PyFileRead;
use super::CountMatrixData;
use super::Motif;
use super::ScoringMatrixData;
use super::WeightMatrixData;

// --- Error handling ----------------------------------------------------------

fn convert_error(error: Error) -> PyErr {
    match error {
        Error::InvalidData(None) => PyValueError::new_err("invalid data"),
        Error::InvalidData(Some(err)) => PyValueError::new_err(format!("invalid data: {}", err)),
        Error::Io(err) => Arc::into_inner(err)
            .map(PyErr::from)
            .unwrap_or_else(|| PyOSError::new_err("unknown error")),
        Error::Nom(err) => PyValueError::new_err(format!("failed to parse input: {}", err)),
    }
}

// --- JASPAR motif ------------------------------------------------------------

/// A motif loaded from a JASPAR or JASPAR16 file.
///
/// The JASPAR database stores motifs with a FASTA-like header line containing
/// the motif name and description, and one line per matrix column prefixed
/// by the alphabet symbol that contains the count matrix.
///
#[pyclass(module = "lightmotif.lib", extends = Motif)]
pub struct JasparMotif {
    #[pyo3(get)]
    /// `str` or `None`: The description of the motif, if any.
    description: Option<String>,
}

impl JasparMotif {
    fn convert(record: Result<lightmotif_io::jaspar::Record, Error>) -> PyResult<PyObject> {
        let record = record.map_err(convert_error)?;
        let name = record.id().to_string();
        let description = record.description().map(String::from);
        let counts = record.into();
        Python::with_gil(|py| {
            let mut motif = Motif::from_counts(py, counts)?;
            motif.name = Some(name);
            let init = PyClassInitializer::from(motif).add_subclass(JasparMotif { description });
            Ok(Py::new(py, init)?.into_any())
        })
    }

    fn convert16<A>(record: Result<lightmotif_io::jaspar16::Record<A>, Error>) -> PyResult<PyObject>
    where
        A: Alphabet,
        CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let record = record.map_err(convert_error)?;
        let name = record.id().to_string();
        let description = record.description().map(String::from);
        let counts = record.into_matrix();
        Python::with_gil(|py| {
            let mut motif = Motif::from_counts(py, counts)?;
            motif.name = Some(name);
            let init = PyClassInitializer::from(motif).add_subclass(JasparMotif { description });
            Ok(Py::new(py, init)?.into_any())
        })
    }
}

// --- UniPROBE motif ----------------------------------------------------------

/// A motif loaded from a UniPROBE file.
#[pyclass(module = "lightmotif.lib", extends = Motif)]
pub struct UniprobeMotif {}

impl UniprobeMotif {
    fn convert<A>(record: Result<lightmotif_io::uniprobe::Record<A>, Error>) -> PyResult<PyObject>
    where
        A: Alphabet,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let record = record.map_err(convert_error)?;
        let name = record.id().to_string();
        let freqs = record.into_matrix();
        let weights = freqs.to_weight(None);
        Python::with_gil(|py| {
            let mut motif = Motif::from_weights(py, weights)?;
            motif.name = Some(name);
            let init = PyClassInitializer::from(motif).add_subclass(UniprobeMotif {});
            Ok(Py::new(py, init)?.into_any())
        })
    }
}

// --- TRANSFAC records --------------------------------------------------------

/// A motif loaded from a TRANSFAC file.
///
/// The TRANSFAC database stores motif information in an EMBL-like file that
/// contains the count matrix for the motif, as well as optional metadata
/// such as accession, description, creation date or bibliography.
///
#[pyclass(module = "lightmotif.lib", extends = Motif)]
pub struct TransfacMotif {
    #[pyo3(get)]
    /// `str` or `None`: The identifier of the motif, if any.
    id: Option<String>,
    #[pyo3(get)]
    /// `str` or `None`: The accession of the motif, if any.
    accession: Option<String>,
    #[pyo3(get)]
    /// `str` or `None`: The description of the motif, if any.
    description: Option<String>,
}

impl TransfacMotif {
    fn convert<A>(record: Result<lightmotif_io::transfac::Record<A>, Error>) -> PyResult<PyObject>
    where
        A: Alphabet,
        CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let record = record.map_err(convert_error)?;

        let name = record.name().map(String::from);
        let description = record.description().map(String::from);
        let id = record.id().map(String::from);
        let accession = record.accession().map(String::from);
        let counts = record
            .to_counts()
            .ok_or_else(|| PyValueError::new_err("invalid count matrix"))?;
        Python::with_gil(|py| {
            let mut motif = Motif::from_counts(py, counts)?;
            motif.name = name;
            let init = PyClassInitializer::from(motif).add_subclass(TransfacMotif {
                description,
                accession,
                id,
            });
            Ok(Py::new(py, init)?.into_any())
        })
    }
}

// --- MEME records ------------------------------------------------------------

/// A motif loaded from a MEME file.
///
/// `MEME format files <https://meme-suite.org/meme/doc/meme-format.html>`_
/// store additional metadata about each motif, such as
#[pyclass(module = "lightmotif.lib", extends = Motif)]
pub struct MemeMotif {
    #[pyo3(get)]
    /// `str` or `None`: The description of the motif, if any.
    description: Option<String>,
    #[pyo3(get)]
    /// `str` or `None`: The URL of the motif, if any.
    url: Option<String>,
}

impl MemeMotif {
    fn convert<A>(record: Result<lightmotif_io::meme::Record<A>, Error>) -> PyResult<PyObject>
    where
        A: Alphabet,
        CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
        WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
        ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
    {
        let record = record.map_err(convert_error)?;

        let name = record.id().into();
        let description = record.name().map(String::from);
        let url = record.url().map(String::from);
        let background = record.background().map(Clone::clone);

        let weights = record.into_matrix().to_weight(background);

        Python::with_gil(|py| {
            let mut motif = Motif::from_weights(py, weights)?;
            motif.name = Some(name);
            let init = PyClassInitializer::from(motif).add_subclass(MemeMotif { description, url });
            Ok(Py::new(py, init)?.into_any())
        })
    }
}

// --- Loader ------------------------------------------------------------------

/// An iterator for loading motifs from a file.
#[pyclass(module = "lightmotif.lib")]
pub struct Loader {
    reader: Box<dyn Iterator<Item = PyResult<PyObject>> + Send + Sync>,
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (file, format="jaspar", *, protein=false))]
    /// Create a new loader from the given parameters.
    ///
    /// Arguments:
    ///     file (`os.PathLike` or file-like object): The file containing the
    ///         motifs to load, as either a path to a filesystem location, or
    ///         a file-like object open in binary mode.
    ///     format (`str`): The format of the motif file. Supported formats
    ///         are ``jaspar``, ``jaspar16``, ``uniprobe`` and ``transfac``.
    ///     protein(`bool`): Set to `True` if the loader should be expecting
    ///         a protein motif rather than a DNA motif.
    ///
    pub fn __init__(
        file: Bound<PyAny>,
        format: &str,
        protein: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = file.py();
        let pathlike = py
            .import(pyo3::intern!(py, "os"))?
            .call_method1(pyo3::intern!(py, "fsdecode"), (&file,));
        let b: Box<dyn BufRead + Send + Sync> = if let Ok(path) = pathlike {
            // NOTE(@althonos): In theory this is safe because `os.fsencode` encodes
            //                  the PathLike object into the OS prefered encoding,
            //                  which is was OsStr wants. In practice, there may be
            //                  some weird bugs if that encoding is incorrect, idk...
            let decoded = path.downcast::<PyString>()?;
            std::fs::File::open(&*decoded.to_cow()?)
                .map(std::io::BufReader::new)
                .map(Box::new)?
        } else {
            PyFileRead::from_ref(&file)
                .map(std::io::BufReader::new)
                .map(Box::new)?
        };
        let reader: Box<dyn Iterator<Item = PyResult<PyObject>> + Send + Sync> = match format {
            "jaspar" if protein => {
                return Err(PyValueError::new_err(
                    "cannot read protein motifs from JASPAR format",
                ))
            }
            "jaspar16" if protein => {
                Box::new(lightmotif_io::jaspar16::read::<_, Protein>(b).map(JasparMotif::convert16))
            }
            "transfac" if protein => {
                Box::new(lightmotif_io::transfac::read::<_, Protein>(b).map(TransfacMotif::convert))
            }
            "uniprobe" if protein => {
                Box::new(lightmotif_io::uniprobe::read::<_, Protein>(b).map(UniprobeMotif::convert))
            }
            "meme" if protein => {
                Box::new(lightmotif_io::meme::read::<_, Protein>(b).map(MemeMotif::convert))
            }
            "jaspar" => Box::new(lightmotif_io::jaspar::read(b).map(JasparMotif::convert)),
            "jaspar16" => {
                Box::new(lightmotif_io::jaspar16::read::<_, Dna>(b).map(JasparMotif::convert16))
            }
            "transfac" => {
                Box::new(lightmotif_io::transfac::read::<_, Dna>(b).map(TransfacMotif::convert))
            }
            "uniprobe" => {
                Box::new(lightmotif_io::uniprobe::read::<_, Dna>(b).map(UniprobeMotif::convert))
            }
            "meme" => Box::new(lightmotif_io::meme::read::<_, Dna>(b).map(MemeMotif::convert)),
            _ => return Err(PyValueError::new_err(format!("invalid format: {}", format))),
        };
        Ok(PyClassInitializer::from(Loader { reader }))
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyResult<PyObject>> {
        slf.reader.next()
    }
}

/// Load the motifs contained in a file.
///
/// Arguments:
///     file (`os.PathLike` or file-like object): The file containing the
///         motifs to load, as either a path to a filesystem location, or
///         a file-like object open in binary mode.
///     format (`str`): The format of the motif file. Supported formats
///         are ``jaspar``, ``jaspar16``, ``uniprobe`` and ``transfac``.
///     protein(`bool`): Set to `True` if the loader should be expecting
///         a protein motif rather than a DNA motif.
///
/// Returns:
///     `~lightmotif.Loader`: A loader configured to load one or more `Motif`
///     from the given file.
///
#[pyfunction]
#[pyo3(signature = (file, format="jaspar", *, protein=false))]
pub fn load<'py>(
    file: Bound<'py, PyAny>,
    format: &str,
    protein: bool,
) -> PyResult<Bound<'py, Loader>> {
    Bound::new(file.py(), Loader::__init__(file, format, protein)?)
}
