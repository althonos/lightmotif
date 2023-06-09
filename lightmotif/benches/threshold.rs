#![feature(test)]

extern crate lightmotif;
extern crate test;

use lightmotif::abc::Dna;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::StripedScores;
use lightmotif::pli::Threshold;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;
use lightmotif::seq::StripedSequence;

const SEQUENCE: &str = include_str!("ecoli.txt");

fn bench<C: StrictlyPositive, P: Score<Dna, C> + Threshold<C>>(
    bencher: &mut test::Bencher,
    pli: &P,
) {
    let mut striped = StripedSequence::<Dna, C>::encode(SEQUENCE).unwrap();

    let cm = CountMatrix::<Dna>::from_sequences(&[
        EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(None);

    striped.configure(&pssm);
    let mut scores = StripedScores::empty();
    pli.score_into(&striped, &pssm, &mut scores);

    bencher.bytes = (std::mem::size_of::<f32>() * scores.len()) as u64;
    bencher.iter(|| {
        test::black_box(pli.threshold(&scores, 0.0));
    });
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
    bench::<U16, _>(bencher, &pli);
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_sse2_32(bencher: &mut test::Bencher) {
    let pli = Pipeline::sse2().unwrap();
    bench::<U32, _>(bencher, &pli);
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let pli = Pipeline::avx2().unwrap();
    bench(bencher, &pli);
}
