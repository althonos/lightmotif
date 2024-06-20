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
use lightmotif::pwm::CountMatrix;
use lightmotif::scan::Scanner;
use lightmotif::scores::StripedScores;
use lightmotif::seq::EncodedSequence;

const SEQUENCE: &str = include_str!("../lightmotif/benches/ecoli.txt");
const N: usize = SEQUENCE.len() / 10;

#[bench]
fn bench_scanner_max_by(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..N];
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
    let mut best = 0;
    bencher.iter(|| {
        best = Scanner::new(&pssm, &striped)
            .max_by(|x, y| x.score.partial_cmp(&y.score).unwrap())
            .unwrap()
            .position;
    });
    bencher.bytes = seq.len() as u64;

    println!("best: {:?}", best);
}

#[bench]
fn bench_scanner_best(bencher: &mut test::Bencher) {
    let seq = &SEQUENCE[..N];
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
    let mut best = 0;
    bencher.iter(|| best = Scanner::new(&pssm, &striped).max().unwrap().position);
    bencher.bytes = seq.len() as u64;

    println!("best: {:?}", best);
}

fn bench_lightmotif<C: StrictlyPositive, P: Score<f32, Dna, C> + Maximum<f32, C>>(
    bencher: &mut test::Bencher,
    pli: &P,
) {
    let seq = &SEQUENCE[..N];
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
    scores.resize(striped.matrix().rows(), striped.len());
    bencher.bytes = seq.len() as u64;
    let mut best = 0;
    bencher.iter(|| {
        test::black_box(pli.score_into(&pssm, &striped, &mut scores));
        best = scores.offset(test::black_box(pli.argmax(&scores).unwrap()));
    });

    println!("best: {:?}", best);
}

fn bench_lightmotif_discrete<C: StrictlyPositive, P: Score<u8, Dna, C> + Maximum<u8, C>>(
    bencher: &mut test::Bencher,
    pli: &P,
) {
    let seq = &SEQUENCE[..N];
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
    let dm = pssm.to_discrete();

    striped.configure(&pssm);
    let mut scores = StripedScores::empty();
    let mut best = 0;
    scores.resize(striped.matrix().rows(), striped.len());
    bencher.bytes = seq.len() as u64;
    bencher.iter(|| {
        test::black_box(pli.score_into(&dm, &striped, &mut scores));
        best = scores.offset(test::black_box(pli.argmax(&scores).unwrap()));
    });

    println!("best: {:?}", best);
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
fn bench_discrete_generic(bencher: &mut test::Bencher) {
    let pli = Pipeline::generic();
    bench_lightmotif_discrete::<U1, _>(bencher, &pli);
}

#[cfg(target_feature = "sse2")]
#[bench]
fn bench_discrete_sse2(bencher: &mut test::Bencher) {
    let pli = Pipeline::sse2().unwrap();
    bench_lightmotif_discrete::<U16, _>(bencher, &pli);
}

#[bench]
fn bench_discrete_dispatch(bencher: &mut test::Bencher) {
    let pli = Pipeline::dispatch();
    bench_lightmotif_discrete(bencher, &pli);
}

#[cfg(target_feature = "avx2")]
#[bench]
fn bench_discrete_avx2(bencher: &mut test::Bencher) {
    let pli = Pipeline::avx2().unwrap();
    bench_lightmotif_discrete(bencher, &pli);
}

#[bench]
fn bench_bio(bencher: &mut test::Bencher) {
    use bio::pattern_matching::pssm::DNAMotif;
    use bio::pattern_matching::pssm::Motif;

    let seq = &SEQUENCE[..N];

    let pssm = DNAMotif::from_seqs(
        vec![b"GTTGACCTTATCAAC".to_vec(), b"GTTGATCCAGTCAAC".to_vec()].as_ref(),
        None,
    )
    .unwrap();

    bencher.bytes = seq.len() as u64;
    let mut best = 0;
    bencher.iter(|| best = test::black_box(pssm.score(seq.as_bytes()).unwrap()).loc);

    println!("best: {:?}", best);
}
