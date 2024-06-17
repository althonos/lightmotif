use std::io::BufRead;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Pseudocounts;

use lightmotif::dense::DenseMatrix;
use lightmotif::pwm::CountMatrix;
use lightmotif::pwm::FrequencyMatrix;

mod parse;
mod reader;

pub use self::reader::Reader;

#[derive(Debug, Clone)]
pub struct Record<A: Alphabet> {
    id: Option<String>,
    accession: Option<String>,
    name: Option<String>,
    description: Option<String>,
    data: Option<DenseMatrix<f32, A::K>>,
    dates: Vec<Date>,
    references: Vec<Reference>,
    sites: Vec<String>,
}

impl<A: Alphabet> Record<A> {
    /// The identifier of the record, if any.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The accession of the record, if any.
    pub fn accession(&self) -> Option<&str> {
        self.accession.as_deref()
    }

    /// The name of the record, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The description of the record, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// The raw data found in the matrix.
    pub fn data(&self) -> Option<&DenseMatrix<f32, A::K>> {
        self.data.as_ref()
    }

    /// The raw data found in the matrix.
    pub fn to_counts(&self) -> Option<CountMatrix<A>> {
        if let Some(data) = &self.data {
            let mut counts = DenseMatrix::<u32, A::K>::new(data.rows());
            for (i, row) in data.iter().enumerate() {
                for (j, &x) in row.iter().enumerate() {
                    // check the matrix contains count data
                    if x.round() != x {
                        return None;
                    }
                    counts[i][j] = x.round() as u32
                }
            }
            CountMatrix::new(counts).ok()
        } else {
            None
        }
    }

    pub fn to_freq<P>(&self, pseudo: P) -> Option<FrequencyMatrix<A>>
    where
        P: Into<Pseudocounts<A>>,
    {
        if let Some(data) = &self.data {
            let p = pseudo.into();
            let mut probas = DenseMatrix::<f32, A::K>::new(data.rows());
            for (i, row) in data.iter().enumerate() {
                let src = &data[i];
                let dst = &mut probas[i];
                for (j, &x) in row.iter().enumerate() {
                    dst[j] = x + p.counts()[j];
                }
                let s: f32 = dst.iter().sum();
                for x in dst.iter_mut() {
                    *x /= s;
                }
            }
            FrequencyMatrix::new(probas).ok()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateKind {
    Created,
    Updated,
}

#[derive(Debug, Clone)]
pub struct Date {
    kind: DateKind,
    author: String,
    day: u8,
    month: u8,
    year: u16,
}

#[derive(Clone, Debug)]
pub struct ReferenceNumber {
    local: u32,
    xref: Option<String>,
}

impl ReferenceNumber {
    pub fn new(local: u32) -> Self {
        Self { local, xref: None }
    }

    pub fn with_xref<X>(local: u32, xref: X) -> Self
    where
        X: Into<Option<String>>,
    {
        Self {
            local,
            xref: xref.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Reference {
    number: ReferenceNumber,
    // authors: String,
    title: Option<String>,
    link: Option<String>,
    pmid: Option<String>,
}

pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::reader::Reader<B, A> {
    self::reader::Reader::new(reader)
}
