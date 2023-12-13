extern crate lightmotif;
extern crate lightmotif_io;

use std::fs::File;
use std::io::BufReader;

use lightmotif::Dna;
use lightmotif_io::transfac::Record;

#[test]
fn parse_prodoric() {
    let mut reader = File::open("tests/MX000001.transfac")
        .map(BufReader::new)
        .map(lightmotif_io::transfac::Reader::new)
        .unwrap();
    let record: Record<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(record.id(), Some("prodoric_MX000001"));
}

#[test]
fn parse_transfac() {
    let mut reader = File::open("tests/M00005.transfac")
        .map(BufReader::new)
        .map(lightmotif_io::transfac::Reader::new)
        .unwrap();
    let record: Record<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(record.id(), Some("V$AP4_01"));
    assert_eq!(record.accession(), Some("M00005"));
}

#[test]
fn parse_jaspar() {
    let mut reader = File::open("tests/MA0001.2.transfac")
        .map(BufReader::new)
        .map(lightmotif_io::transfac::Reader::new)
        .unwrap();
    let record: Record<Dna> = reader.next().unwrap().unwrap();
    assert_eq!(record.id(), Some("AGL3"));
    assert_eq!(record.accession(), Some("MA0001.2"));
}
