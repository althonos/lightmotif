#![feature(test)]

extern crate lightmotif;
extern crate lightmotif_io;
extern crate test;

use lightmotif::abc::Dna;
use lightmotif_io::meme::Reader;

#[bench]
fn bench_reader(bencher: &mut test::Bencher) {
    let jaspar = include_str!("JASPAR2024.meme");
    bencher.bytes = jaspar.as_bytes().len() as u64;
    bencher.iter(|| {
        test::black_box(Reader::<_, Dna>::new(std::io::Cursor::new(jaspar)).collect::<Vec<_>>());
    })
}
