extern crate lightmotif;
extern crate typenum;

use lightmotif::abc::Dna;
use lightmotif::abc::Nucleotide;
use lightmotif::abc::Nucleotide::*;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::U1;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::BestPosition;
use lightmotif::pli::Encode;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::Threshold;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;
use lightmotif::seq::StripedSequence;

const SEQUENCE: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";

// scores computed with Bio.motifs
#[rustfmt::skip]
const EXPECTED: &[Nucleotide] = &[
    A, T, G, T, C, C, C, A, A, C, A, A, C, G, A, T, A, C, C, C, C, G, A, G, C, C, C, A, T, C, G, C, C, G, T, C, A, T, C, G, G, C, T, C, G, G, C, A, T, G, C, A, G, A, T, T, C, C, C, A, G, G, C, G 
];

fn test_encode<P: Encode<Dna>>(pli: &P) {
    let mut encoded = pli.encode(SEQUENCE).unwrap();
    assert_eq!(encoded, EXPECTED);

    // let mut striped = StripedSequence::<Dna, C>::encode(SEQUENCE).unwrap();

    // let cm = CountMatrix::<Dna>::from_sequences(
    //     PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    // )
    // .unwrap();
    // let pbm = cm.to_freq(0.1);
    // let pwm = pbm.to_weight(None);
    // let pssm = pwm.into();

    // striped.configure(&pssm);
    // let result = pli.score(&striped, &pssm);
    // let scores = result.to_vec();

    // assert_eq!(scores.len(), EXPECTED.len());
    // for i in 0..scores.len() {
    //     assert!(
    //         (scores[i] - EXPECTED[i]).abs() < 1e-5,
    //         "{} != {} at position {}",
    //         scores[i],
    //         EXPECTED[i],
    //         i
    //     );
    // }
}

#[test]
fn test_encode_generic() {
    let pli = Pipeline::generic();
    test_encode(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_encode_sse2() {
    let pli = Pipeline::sse2().unwrap();
    test_encode(&pli);
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_encode_avx2() {
    let pli = Pipeline::avx2().unwrap();
    test_encode(&pli);
}

#[cfg(target_feature = "neon")]
#[test]
fn test_encode_neon() {
    let pli = Pipeline::neon().unwrap();
    test_encode(&pli);
}
