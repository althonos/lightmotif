use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::abc::Protein;
use lightmotif_io::error::Error;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::CountMatrixData;
use super::Motif;
use super::ScoringMatrixData;
use super::WeightMatrixData;

fn convert_jaspar(record: Result<lightmotif_io::jaspar::Record, Error>) -> PyResult<Motif> {
    let record = record.unwrap();
    let counts = record.into();
    Python::with_gil(|py| Motif::from_counts(py, counts))
}

fn convert_jaspar16<A>(record: Result<lightmotif_io::jaspar16::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.unwrap();
    let counts = record.into_matrix();
    Python::with_gil(|py| Motif::from_counts(py, counts))
}

fn convert_transfac<A>(record: Result<lightmotif_io::transfac::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    CountMatrixData: From<lightmotif::pwm::CountMatrix<A>>,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.unwrap();
    let counts = record
        .to_counts()
        .ok_or_else(|| PyValueError::new_err("invalid count matrix"))?;
    Python::with_gil(|py| Motif::from_counts(py, counts))
}

fn convert_uniprobe<A>(record: Result<lightmotif_io::uniprobe::Record<A>, Error>) -> PyResult<Motif>
where
    A: Alphabet,
    WeightMatrixData: From<lightmotif::pwm::WeightMatrix<A>>,
    ScoringMatrixData: From<lightmotif::pwm::ScoringMatrix<A>>,
{
    let record = record.unwrap();
    let freqs = record.into_matrix();
    let weights = freqs.to_weight(None);
    Python::with_gil(|py| Motif::from_weights(py, weights))
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

#[pyfunction]
#[pyo3(signature = (path, format="jaspar", protein=false))]
pub fn load(path: &str, format: &str, protein: bool) -> PyResult<Loader> {
    let file = std::fs::File::open(path)
        .map(std::io::BufReader::new)
        .unwrap();
    let reader: Box<dyn Iterator<Item = PyResult<Motif>> + Send> = match format {
        "jaspar" if protein => {
            return Err(PyValueError::new_err(
                "cannot read protein motifs from JASPAR format",
            ))
        }
        "jaspar16" if protein => Box::new(
            lightmotif_io::jaspar16::read::<_, Protein>(file).map(convert_jaspar16::<Protein>),
        ),
        "transfac" if protein => Box::new(
            lightmotif_io::transfac::read::<_, Protein>(file).map(convert_transfac::<Protein>),
        ),
        "uniprobe" if protein => Box::new(
            lightmotif_io::uniprobe::read::<_, Protein>(file).map(convert_uniprobe::<Protein>),
        ),
        "jaspar" => Box::new(lightmotif_io::jaspar::read(file).map(convert_jaspar)),
        "jaspar16" => {
            Box::new(lightmotif_io::jaspar16::read::<_, Dna>(file).map(convert_jaspar16::<Dna>))
        }
        "transfac" => {
            Box::new(lightmotif_io::transfac::read::<_, Dna>(file).map(convert_transfac::<Dna>))
        }
        "uniprobe" => {
            Box::new(lightmotif_io::uniprobe::read::<_, Dna>(file).map(convert_uniprobe::<Dna>))
        }
        _ => return Err(PyValueError::new_err(format!("invalid format: {}", format))),
    };
    Ok(Loader { reader })
}
