#![feature(test)]

extern crate lightmotif;
extern crate lightmotif_io;
extern crate test;

use lightmotif::abc::Dna;
use lightmotif_io::transfac::Reader;

#[bench]
fn bench_reader(bencher: &mut test::Bencher) {
    let prodoric = include_str!("prodoric.transfac");
    bencher.bytes = prodoric.as_bytes().len() as u64;
    bencher.iter(|| {
        test::black_box(Reader::<_, Dna>::new(std::io::Cursor::new(prodoric)).collect::<Vec<_>>());
    })
}
