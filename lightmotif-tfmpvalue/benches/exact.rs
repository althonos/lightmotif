#![feature(test)]

extern crate lightmotif;
extern crate lightmotif_tfmpvalue;
extern crate test;

use lightmotif::abc::Dna;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;
use lightmotif_tfmpvalue::TfmPvalue;

#[bench]
fn bench_pvalue(bencher: &mut test::Bencher) {
    let pssm = CountMatrix::<Dna>::from_sequences([
        EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap()
    .to_freq(0.25)
    .to_scoring(None);
    let mut tfmp = TfmPvalue::new(&pssm);
    bencher.iter(|| {
        test::black_box(tfmp.pvalue(8.0));
    });
}

#[bench]
fn bench_score(bencher: &mut test::Bencher) {
    let pssm = CountMatrix::<Dna>::from_sequences([
        EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap()
    .to_freq(0.25)
    .to_scoring(None);
    let mut tfmp = TfmPvalue::new(&pssm);
    bencher.iter(|| {
        test::black_box(tfmp.score(1e-3));
    });
}
