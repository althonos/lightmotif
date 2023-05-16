#![feature(test)]

extern crate lightmotif;
extern crate test;

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::__m256i;
use std::str::FromStr;

use lightmotif::CountMatrix;
use lightmotif::Dna;
use lightmotif::EncodedSequence;
use lightmotif::Pipeline;
use lightmotif::Score;
use lightmotif::StripedScores;
use typenum::consts::U32;

const SEQUENCE: &'static str = include_str!("ecoli.txt");

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped::<U32>();

    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(None);

    striped.configure(&pssm);
    let pli = Pipeline::generic();
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pssm, &mut scores)));
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_sse2(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(None);

    striped.configure(&pssm);
    let pli = Pipeline::sse2().unwrap();
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pssm, &mut scores)));
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(None);

    striped.configure(&pssm);
    let pli = Pipeline::avx2().unwrap();
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pssm, &mut scores)));
}
