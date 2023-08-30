extern crate lightmotif;
extern crate typenum;

use lightmotif::abc::Dna;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::U1;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::Maximum;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::Threshold;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;

const SEQUENCE: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
const PATTERNS: &[&str] = &["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"];

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

fn test_score<C: StrictlyPositive, P: Score<Dna, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&striped, &pssm);
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

fn test_argmax<C: StrictlyPositive, P: Score<Dna, C> + Maximum<C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&striped, &pssm);
    assert_eq!(pli.argmax(&result), Some(18));
}

fn test_threshold<C: StrictlyPositive, P: Score<Dna, C> + Threshold<C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&striped, &pssm);

    let mut positions = pli.threshold(&result, -10.0);
    positions.sort_unstable();
    assert_eq!(positions, vec![18, 27, 32]);

    let mut positions = pli.threshold(&result, -15.0);
    positions.sort_unstable();
    assert_eq!(positions, vec![10, 13, 14, 18, 24, 27, 32, 35, 40, 47]);
}

#[test]
fn test_score_generic() {
    let pli = Pipeline::generic();
    test_score::<U32, _>(&pli);
    test_score::<U1, _>(&pli);
}

#[test]
fn test_argmax_generic() {
    let pli = Pipeline::generic();
    test_argmax::<U32, _>(&pli);
    test_argmax::<U1, _>(&pli);
}

#[test]
fn test_threshold_generic() {
    let pli = Pipeline::generic();
    test_threshold::<U32, _>(&pli);
}

#[test]
fn test_score_dispatch() {
    let pli = Pipeline::dispatch();
    test_score(&pli);
}

#[test]
fn test_argmax_dispatch() {
    let pli = Pipeline::dispatch();
    test_argmax(&pli);
}

#[test]
fn test_threshold_dispatch() {
    let pli = Pipeline::dispatch();
    test_threshold(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_score_sse2() {
    let pli = Pipeline::sse2().unwrap();
    test_score::<U16, _>(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_argmax_sse2() {
    let pli = Pipeline::sse2().unwrap();
    test_argmax::<U16, _>(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_threshold_sse2() {
    let pli = Pipeline::sse2().unwrap();
    test_threshold::<U16, _>(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_score_sse2_32() {
    let pli = Pipeline::sse2().unwrap();
    test_score::<U32, _>(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_argmax_sse2_32() {
    let pli = Pipeline::sse2().unwrap();
    test_argmax::<U32, _>(&pli);
}

#[cfg(target_feature = "sse2")]
#[test]
fn test_threshold_sse2_32() {
    let pli = Pipeline::sse2().unwrap();
    test_threshold::<U32, _>(&pli);
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_score_avx2() {
    let pli = Pipeline::avx2().unwrap();
    test_score(&pli);
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_argmax_avx2() {
    let pli = Pipeline::avx2().unwrap();
    test_argmax(&pli);
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_threshold_avx2() {
    let pli = Pipeline::avx2().unwrap();
    test_threshold::<U32, _>(&pli);
}

#[cfg(target_feature = "neon")]
#[test]
fn test_score_neon() {
    let pli = Pipeline::neon().unwrap();
    test_score::<U16, _>(&pli);
}

#[cfg(target_feature = "neon")]
#[test]
fn test_argmax_neon() {
    let pli = Pipeline::neon().unwrap();
    test_argmax::<U16, _>(&pli);
}

#[cfg(target_feature = "neon")]
#[test]
fn test_threshold_neon() {
    let pli = Pipeline::neon().unwrap();
    test_threshold::<U16, _>(&pli);
}
