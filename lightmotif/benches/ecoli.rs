#![feature(test)]

extern crate lightmotif;
extern crate test;

use std::str::FromStr;

use lightmotif::abc::Dna;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::StripedScores;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;
use typenum::consts::U32;
use typenum::marker_traits::NonZero;
use typenum::marker_traits::Unsigned;

const SEQUENCE: &'static str = include_str!("ecoli.txt");

fn bench<C: Unsigned + NonZero, P: Score<Dna, C>>(bencher: &mut test::Bencher, pli: &P) {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped::<C>();

    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::from_str("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::from_str("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(None);

    striped.configure(&pssm);
    let mut scores = StripedScores::new_for(&striped, &pssm);
    bencher.bytes = SEQUENCE.len() as u64;
    bencher.iter(|| test::black_box(pli.score_into(&striped, &pssm, &mut scores)));
}

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let pli = Pipeline::generic();
    bench::<U32, _>(bencher, &pli);
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_sse2(bencher: &mut test::Bencher) {
    let pli = Pipeline::sse2().unwrap();
    bench(bencher, &pli);
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let pli = Pipeline::avx2().unwrap();
    bench(bencher, &pli);
}
