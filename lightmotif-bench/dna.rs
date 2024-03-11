#![feature(test)]

#[cfg(feature = "bio")]
extern crate bio;
extern crate lightmotif;
extern crate test;

use lightmotif::abc::Background;
use lightmotif::abc::Dna;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::U1;
use lightmotif::num::U16;
use lightmotif::pli::Maximum;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::Stripe;
use lightmotif::pli::StripedScores;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;

const SEQUENCE: &str = include_str!("../lightmotif/benches/ecoli.txt");

fn bench_lightmotif<C: StrictlyPositive, P: Score<Dna, C> + Maximum<C>>(
    bencher: &mut test::Bencher,
    pli: &P,
) {
    let seq = &SEQUENCE[..10000];
    let encoded = EncodedSequence::<Dna>::encode(seq).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let bg = Background::<Dna>::uniform();
    let cm = CountMatrix::<Dna>::from_sequences([
        EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
        EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
    ])
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(bg);

    striped.configure(&pssm);
    let mut scores = StripedScores::empty();
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        test::black_box(pli.score_into(&pssm, &striped, &mut scores));
        test::black_box(pli.argmax(&scores));
    });
}

#[bench]
fn bench_generic(bencher: &mut test::Bencher) {
    let pli = Pipeline::generic();
    bench_lightmotif::<U1, _>(bencher, &pli);
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_sse2(bencher: &mut test::Bencher) {
    let pli = Pipeline::sse2().unwrap();
    bench_lightmotif::<U16, _>(bencher, &pli);
}

#[bench]
fn bench_dispatch(bencher: &mut test::Bencher) {
    let pli = Pipeline::dispatch();
    bench_lightmotif(bencher, &pli);
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_avx2(bencher: &mut test::Bencher) {
    let pli = Pipeline::avx2().unwrap();
    bench_lightmotif(bencher, &pli);
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
