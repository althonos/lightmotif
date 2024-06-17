//! Parser implementation for the TRANSFAC format.
//!
//! The TRANSFAC matrix format is similar to the EMBL sequence format,
//! using a 2-letter header before each row that is used for metadata.
//! The matrix usually contains counts, but they may be in floating-point
//! format if they were rescaled.
//!
//! ```text
//! AC  M00005
//! XX
//! DT  19.10.1992 (created); ewi.
//! CO  Copyright (C), Biobase GmbH.
//! XX
//! P0      A      C      G      T
//! 01      3      0      0      2      W
//! 02      1      1      3      0      G
//! 03      3      1      1      0      A
//! 04      2      1      2      0      R
//! 05      1      2      0      2      Y
//! 06      0      5      0      0      C
//! 07      5      0      0      0      A
//! XX
//! ```
//!
//! The parser implemented in this module is not complete, and only supports
//! the following metadata:
//!
//! - `ID`: identifier
//! - `AC`: accession
//! - `NA`: name
//! - `DE`: description
//! - `DT`: date (creation or update)
//! - `RE`: references (similar to EMBL in format)
//! - `BS`: binding sites
//! - `P0`: matrix.
//!

use std::io::BufRead;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Pseudocounts;
use lightmotif::dense::DenseMatrix;
use lightmotif::pwm::CountMatrix;
use lightmotif::pwm::FrequencyMatrix;

mod parse;
mod reader;

pub use self::reader::Reader;

/// A TRANSFAC record.
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

    /// The references associated with the record.
    pub fn references(&self) -> &[Reference] {
        &self.references
    }

    /// Get the record matrix as an integer count data.
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

    /// Get the record matrix as a frequency matrix.
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
    /// Create a new reference number with the given number.
    pub fn new(local: u32) -> Self {
        Self::with_xref(local, None)
    }

    /// Create a new reference number with the given number and cross-reference.
    pub fn with_xref<X>(local: u32, xref: X) -> Self
    where
        X: Into<Option<String>>,
    {
        Self {
            local,
            xref: xref.into(),
        }
    }

    /// The local number of the reference number.
    pub fn local(&self) -> u32 {
        self.local
    }

    /// The cross-reference, if any.
    pub fn xref(&self) -> Option<&str> {
        self.xref.as_ref().map(String::as_str)
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

impl Reference {
    /// Create a new reference with the given reference number.
    pub fn new(number: ReferenceNumber) -> Self {
        Self {
            number,
            title: None,
            link: None,
            pmid: None,
        }
    }

    /// The number of the reference.
    pub fn number(&self) -> &ReferenceNumber {
        &self.number
    }

    /// The title of the reference, if any.
    pub fn title(&self) -> Option<&str> {
        self.title.as_ref().map(String::as_str)
    }

    /// A link to the reference, if any.
    pub fn link(&self) -> Option<&str> {
        self.link.as_ref().map(String::as_str)
    }

    /// The PubMed ID of the reference, if any.
    pub fn pmid(&self) -> Option<&str> {
        self.pmid.as_ref().map(String::as_str)
    }
}

pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::reader::Reader<B, A> {
    self::reader::Reader::new(reader)
}
