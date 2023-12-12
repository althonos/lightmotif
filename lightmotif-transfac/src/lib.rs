#![doc = include_str!("../README.md")]
#![allow(unused)]

extern crate memchr;

use std::io::BufRead;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::pwm::FrequencyMatrix;

pub mod error;
#[doc(hidden)]
pub mod parse;
pub mod reader;

#[derive(Clone)]
pub struct Record<A: Alphabet> {
    id: Option<String>,
    accession: Option<String>,
    name: Option<String>,
    description: Option<String>,
    matrix: DenseMatrix<f32, A::K>,
    dates: Vec<Date>,
    references: Vec<Reference>,
    sites: Vec<String>,
}

impl<A: Alphabet> Record<A> {
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn accession(&self) -> Option<&str> {
        self.accession.as_deref()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl<A: Alphabet> AsRef<DenseMatrix<f32, A::K>> for Record<A> {
    fn as_ref(&self) -> &DenseMatrix<f32, A::K> {
        &self.matrix
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
