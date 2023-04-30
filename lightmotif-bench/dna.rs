#![feature(test)]

#[cfg(feature = "bio")]
extern crate bio;
extern crate lightmotif;
extern crate test;

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::__m256;

#[cfg(target_feature = "sse2")]
use std::arch::x86_64::__m128;

use lightmotif::Alphabet;
use lightmotif::Background;
use lightmotif::CountMatrix;
use lightmotif::DnaAlphabet;
use lightmotif::EncodedSequence;
use lightmotif::Pipeline;
use lightmotif::StripedScores;

const SEQUENCE: &'static str = include_str!("../lightmotif/benches/ecoli.txt");

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(seq).unwrap();
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
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        pli.score_into(&striped, &pwm, &mut scores);
        test::black_box(scores.argmax());
    });
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_sse2(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(seq).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::<DnaAlphabet, { DnaAlphabet::K }>::uniform();
    let cm = CountMatrix::<DnaAlphabet, { DnaAlphabet::K }>::from_sequences(&[
        EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_text("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_probability(0.1);
    let pwm = pbm.to_weight(bg);

    striped.configure(&pwm);
    let pli = Pipeline::<_, __m128>::new();

    let mut scores = StripedScores::new_for(&striped, &pwm);
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        pli.score_into(&striped, &pwm, &mut scores);
        test::black_box(scores.argmax());
    });
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<DnaAlphabet>::from_text(seq).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::<DnaAlphabet, { DnaAlphabet::K }>::uniform();
    let cm = CountMatrix::<DnaAlphabet, { DnaAlphabet::K }>::from_sequences(&[
        EncodedSequence::from_text("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_text("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_probability(0.1);
    let pwm = pbm.to_weight(bg);

    striped.configure(&pwm);
    let pli = Pipeline::<_, __m256>::new();

    let mut scores = StripedScores::new_for(&striped, &pwm);
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        pli.score_into(&striped, &pwm, &mut scores);
        test::black_box(scores.argmax());
    });
}

#[bench]
fn bench_bio(bencher: &mut test::Bencher) {
    use bio::pattern_matching::pssm::DNAMotif;
    use bio::pattern_matching::pssm::Motif;

    let seq = &SEQUENCE[..10000];

    let pssm = DNAMotif::from_seqs(
        vec![b"GTTGACCTTATCAAC".to_vec(), b"GTTGATCCAGTCAAC".to_vec()].as_ref(),
        None,
    )
    .unwrap();

    bencher.bytes = seq.len() as u64;
    bencher.iter(|| test::black_box(pssm.score(seq.as_bytes()).unwrap()));
}
