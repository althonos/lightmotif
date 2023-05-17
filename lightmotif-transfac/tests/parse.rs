extern crate lightmotif;
extern crate lightmotif_transfac;

use std::fs::File;

use std::io::BufReader;

use lightmotif::Dna;
use lightmotif_transfac::Matrix;

#[test]
fn parse_prodoric() {
    let mut reader = File::open("tests/MX000001.transfac")
        .map(BufReader::new)
        .map(lightmotif_transfac::reader::Reader::new)
        .unwrap();
    let matrix: Matrix<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(matrix.id(), Some("prodoric_MX000001"));
}

#[test]
fn parse_transfac() {
    let mut reader = File::open("tests/M00005.transfac")
        .map(BufReader::new)
        .map(lightmotif_transfac::reader::Reader::new)
        .unwrap();
    let matrix: Matrix<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(matrix.id(), Some("V$AP4_01"));
    assert_eq!(matrix.accession(), Some("M00005"));
}

#[test]
fn parse_jaspar() {
    let mut reader = File::open("tests/MA0001.2.transfac")
        .map(BufReader::new)
        .map(lightmotif_transfac::reader::Reader::new)
        .unwrap();
    let matrix: Matrix<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(matrix.id(), Some("AGL3"));
    assert_eq!(matrix.accession(), Some("MA0001.2"));
}
