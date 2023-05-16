#![feature(test)]

#[cfg(feature = "bio")]
extern crate bio;
extern crate lightmotif;
extern crate test;

#[cfg(target_feature = "sse2")]
use std::arch::x86_64::__m128i;
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::__m256i;
use std::str::FromStr;

use lightmotif::Background;
use lightmotif::CountMatrix;
use lightmotif::Dna;
use lightmotif::EncodedSequence;
use lightmotif::Pipeline;
use lightmotif::Score;
use lightmotif::StripedScores;
use typenum::consts::U1;

const SEQUENCE: &'static str = include_str!("../lightmotif/benches/ecoli.txt");

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<Dna>::from_str(seq).unwrap();
    let mut striped = encoded.to_striped::<U1>();

    let bg = Background::<Dna>::uniform();
    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(bg);

    striped.configure(&pssm);
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        Pipeline::<Dna, u8>::score_into(&striped, &pssm, &mut scores);
        test::black_box(Pipeline::<Dna, u8>::best_position(&scores));
    });
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_sse2(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<Dna>::from_str(seq).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::<Dna>::uniform();
    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(bg);

    striped.configure(&pssm);
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        Pipeline::<_, __m128i>::score_into(&striped, &pssm, &mut scores);
        test::black_box(Pipeline::<Dna, __m128i>::best_position(&scores).unwrap());
    });
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<Dna>::from_str(seq).unwrap();
    let mut striped = encoded.to_striped();

    let bg = Background::<Dna>::uniform();
    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(bg);

    striped.configure(&pssm);
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        Pipeline::<_, __m256i>::score_into(&striped, &pssm, &mut scores);
        test::black_box(Pipeline::<Dna, __m256i>::best_position(&scores).unwrap());
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
