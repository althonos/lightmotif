extern crate lightmotif;

#[cfg(target_feature = "ssse3")]
use std::arch::x86_64::__m128;
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::__m256;
use std::str::FromStr;

use lightmotif::Alphabet;
use lightmotif::CountMatrix;
use lightmotif::Dna;
use lightmotif::EncodedSequence;
use lightmotif::Pipeline;
use lightmotif::Score;

const SEQUENCE: &'static str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
const PATTERNS: &[&'static str] = &["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"];

// scores computed with Bio.motifs
#[rustfmt::skip]
const EXPECTED: &[f32] = &[
    -23.07094  , -18.678621 , -15.219191 , -17.745737 , 
    -18.678621 , -23.07094  , -17.745737 , -19.611507 , 
    -27.463257 , -29.989803 , -14.286304 , -26.53037  , 
    -15.219191 , -10.826873 , -10.826873 , -22.138054 , 
    -38.774437 , -30.922688 ,  -5.50167  , -24.003826 ,
    -18.678621 , -15.219191 , -35.315006 , -17.745737 , 
    -10.826873 , -30.922688 , -23.07094  ,  -6.4345555, 
    -31.855574 , -23.07094  , -15.219191 , -31.855574 ,  
    -8.961102  , -26.53037  , -27.463257 , -14.286304 , 
    -15.219191 , -26.53037  , -23.07094  , -18.678621 ,
    -14.286304 , -18.678621 , -26.53037  , -16.152077 , 
    -17.745737 , -18.678621 , -17.745737 , -14.286304 , 
    -30.922688 , -18.678621 
];

#[test]
fn test_score_generic() {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped::<2>();

    let cm = CountMatrix::<Dna, { Dna::K }>::from_sequences(
        PATTERNS
            .iter()
            .map(|x| EncodedSequence::from_str(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq([0.1, 0.1, 0.1, 0.1, 0.0]);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = Pipeline::<Dna, f32>::score(&striped, &pssm);
    let scores = result.to_vec();

    assert_eq!(scores.len(), EXPECTED.len());
    for i in 0..scores.len() {
        assert!(
            (scores[i] - EXPECTED[i]).abs() < 1e-5,
            "{} != {} at position {}",
            scores[i],
            EXPECTED[i],
            i
        );
    }
}

#[cfg(target_feature = "ssse3")]
#[test]
fn test_score_ssse3() {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::from_sequences(
        PATTERNS
            .iter()
            .map(|x| EncodedSequence::from_str(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq([0.1, 0.1, 0.1, 0.1, 0.0]);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = Pipeline::<_, __m128>::score(&striped, &pssm);

    // for i in 0..result.data.rows() {
    //     println!("i={} {:?}", i, &result.data[i]);
    // }

    let scores = result.to_vec();
    assert_eq!(scores.len(), EXPECTED.len());
    for i in 0..EXPECTED.len() {
        assert!(
            (scores[i] - EXPECTED[i]).abs() < 1e-5,
            "{} != {} at position {}",
            scores[i],
            EXPECTED[i],
            i
        );
    }
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_score_avx2() {
    let encoded = EncodedSequence::<Dna>::from_str(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped::<{ std::mem::size_of::<__m256>() }>();

    let cm = CountMatrix::<Dna, { Dna::K }>::from_sequences(
        PATTERNS
            .iter()
            .map(|x| EncodedSequence::from_str(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq([0.1, 0.1, 0.1, 0.1, 0.0]);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = Pipeline::<_, __m256>::score(&striped, &pssm);
    let scores = result.to_vec();

    // for i in 0..result.data.rows() {
    //     println!("{:?}", &result.data[i]);
    // }

    assert_eq!(scores.len(), EXPECTED.len());
    for i in 0..EXPECTED.len() {
        assert!(
            (scores[i] - EXPECTED[i]).abs() < 1e-5,
            "{} != {} at position {}",
            scores[i],
            EXPECTED[i],
            i
        );
    }
}
