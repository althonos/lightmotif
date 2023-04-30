#![feature(test)]

extern crate lightmotif;
extern crate test;

#[cfg(target_feature = "ssse3")]
use std::arch::x86_64::__m128;

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::__m256;

use lightmotif::Alphabet;
use lightmotif::Background;
use lightmotif::CountMatrix;
use lightmotif::DnaAlphabet;
use lightmotif::EncodedSequence;
use lightmotif::Pipeline;
use lightmotif::StripedScores;

const SEQUENCE: &'static str = include_str!("ecoli.txt");

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped::<32>();

    let bg = Background::<DnaAlphabet, { DnaAlphabet::K }>::uniform();
    let cm = CountMatrix::<DnaAlphabet, { DnaAlphabet::K }>::from_sequences(&[
        EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_text("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_probability(0.1);
    let pwm = pbm.to_weight(bg);

    striped.configure(&pwm);
    let pli = Pipeline::<_, f32>::new();
    let mut scores = StripedScores::new_for(&striped, &pwm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pwm, &mut scores)));
}

#[cfg(target_feature = "ssse3")]
#[bench]
fn bench_ssse3(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::uniform();
    let cm = CountMatrix::from_sequences(&[
        EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_text("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_probability(0.1);
    let pwm = pbm.to_weight(bg);

    striped.configure(&pwm);
    let pli = Pipeline::<_, __m128>::new();

    let mut scores = StripedScores::new_for(&striped, &pwm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pwm, &mut scores)));
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::uniform();
    let cm = CountMatrix::from_sequences(&[
        EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_text("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_probability(0.1);
    let pwm = pbm.to_weight(bg);

    striped.configure(&pwm);
    let pli = Pipeline::<_, __m256>::new();

    let mut scores = StripedScores::new_for(&striped, &pwm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pwm, &mut scores)));
}
