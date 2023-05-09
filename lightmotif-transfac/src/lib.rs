#![doc = include_str!("../README.md")]
#![allow(unused)]

extern crate memchr;

use std::io::BufRead;

use lightmotif::Alphabet;
use lightmotif::CountMatrix;
use lightmotif::DenseMatrix;
use lightmotif::Symbol;

pub mod error;
mod parse;
pub mod reader;

#[derive(Clone, Debug)]
pub struct Matrix<A: Alphabet, const K: usize> {
    id: Option<String>,
    accession: Option<String>,
    name: Option<String>,
    counts: CountMatrix<A, K>,
    dates: Vec<Date>,
    references: Vec<Reference>,
    sites: Vec<String>,
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

pub fn read<B: BufRead, A: Alphabet, const K: usize>(reader: B) -> self::reader::Reader<B, A, K> {
    self::reader::Reader::new(reader)
}
