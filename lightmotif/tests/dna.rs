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
use lightmotif::pli::Stripe;
use lightmotif::pli::Threshold;
use lightmotif::pwm::CountMatrix;
use lightmotif::scores::StripedScores;
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

fn test_score_rows<C: StrictlyPositive, P: Score<f32, Dna, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let mut scores = StripedScores::empty();

    pli.score_rows_into(&pssm, &striped, 0..2, &mut scores);
    assert_eq!(scores.matrix().rows(), 2);
    assert_eq!(scores.matrix()[0][0], EXPECTED[0]);
    assert_eq!(scores.matrix()[1][0], EXPECTED[1]);

    pli.score_rows_into(&pssm, &striped, 1..2, &mut scores);
    assert_eq!(scores.matrix().rows(), 1);
    assert_eq!(scores.matrix()[0][0], EXPECTED[1]);
}

fn test_score<C: StrictlyPositive, P: Score<f32, Dna, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&pssm, &striped);
    let scores = result.unstripe();

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

fn test_score_discrete<C: StrictlyPositive, P: Score<u8, Dna, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.to_scoring();
    let dm = pssm.to_discrete();

    striped.configure(&pssm);
    let result = pli.score(&dm, &striped);
    let scores = result.unstripe();

    assert_eq!(scores.len(), EXPECTED.len());
    for i in 0..scores.len() {
        assert!(
            dm.unscale(scores[i]) >= EXPECTED[i],
            "{} != {} at position {}",
            dm.unscale(scores[i]),
            EXPECTED[i],
            i
        );
    }
}

fn test_argmax<C: StrictlyPositive, P: Score<f32, Dna, C> + Maximum<f32, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&pssm, &striped);
    assert_eq!(pli.argmax(&result).map(|c| result.offset(c)), Some(18));
}

fn test_threshold<C: StrictlyPositive, P: Score<f32, Dna, C> + Threshold<f32, C>>(pli: &P) {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = Pipeline::generic().stripe(encoded);

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.into();

    striped.configure(&pssm);
    let result = pli.score(&pssm, &striped);

    let positions = pli.threshold(&result, -10.0);
    let mut indices = positions
        .into_iter()
        .map(|c| result.offset(c))
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(indices, vec![18, 27, 32]);

    let positions = pli.threshold(&result, -15.0);
    let mut indices = positions
        .into_iter()
        .map(|c| result.offset(c))
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(indices, vec![10, 13, 14, 18, 24, 27, 32, 35, 40, 47]);
}

#[test]
fn test_score_position() {
    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();

    let cm = CountMatrix::<Dna>::from_sequences(
        PATTERNS.iter().map(|x| EncodedSequence::encode(x).unwrap()),
    )
    .unwrap();
    let pbm = cm.to_freq(0.1);
    let pwm = pbm.to_weight(None);
    let pssm = pwm.to_scoring();

    striped.configure(&pssm);
    for i in 0..encoded.len() - pssm.len() + 1 {
        let score = pssm.score_position(&striped, i);
        assert!(
            (score - EXPECTED[i]).abs() < 1e-5,
            "{} != {} at position {}",
            score,
            EXPECTED[i],
            i
        );
    }
}

mod generic {
    use super::*;

    #[test]
    fn score() {
        let pli = Pipeline::generic();
        super::test_score::<U32, _>(&pli);
        super::test_score::<U16, _>(&pli);
        super::test_score::<U1, _>(&pli);
    }

    #[test]
    fn score_rows() {
        let pli = Pipeline::generic();
        super::test_score_rows::<U32, _>(&pli);
        super::test_score_rows::<U16, _>(&pli);
        super::test_score_rows::<U1, _>(&pli);
    }

    #[test]
    fn score_discrete() {
        let pli = Pipeline::generic();
        super::test_score_discrete::<U32, _>(&pli);
        super::test_score_discrete::<U16, _>(&pli);
        super::test_score_discrete::<U1, _>(&pli);
    }

    #[test]
    fn argmax() {
        let pli = Pipeline::generic();
        super::test_argmax::<U32, _>(&pli);
        super::test_argmax::<U1, _>(&pli);
    }

    #[test]
    fn threshold() {
        let pli = Pipeline::generic();
        super::test_threshold::<U32, _>(&pli);
    }
}

mod dispatch {
    use super::*;

    #[test]
    fn score() {
        let pli = Pipeline::dispatch();
        super::test_score(&pli);
    }

    #[test]
    fn score_rows() {
        let pli = Pipeline::dispatch();
        super::test_score_rows(&pli);
    }

    #[test]
    fn score_discrete() {
        let pli = Pipeline::dispatch();
        super::test_score_discrete(&pli);
    }

    #[test]
    fn argmax() {
        let pli = Pipeline::dispatch();
        super::test_argmax(&pli);
    }

    #[test]
    fn threshold() {
        let pli = Pipeline::dispatch();
        super::test_threshold(&pli);
    }
}

#[cfg(target_feature = "sse2")]
mod sse2 {
    use super::*;

    #[test]
    fn score() {
        let pli = Pipeline::sse2().unwrap();
        super::test_score::<U32, _>(&pli);
        super::test_score::<U16, _>(&pli);
    }

    #[test]
    fn score_rows() {
        let pli = Pipeline::sse2().unwrap();
        super::test_score_rows::<U32, _>(&pli);
        super::test_score_rows::<U16, _>(&pli);
    }

    // #[test]
    // fn score_discrete() {
    //     let pli = Pipeline::sse2().unwrap();
    //     super::test_score_discrete(&pli);
    // }

    #[test]
    fn argmax() {
        let pli = Pipeline::sse2().unwrap();
        super::test_argmax::<U32, _>(&pli);
        super::test_argmax::<U16, _>(&pli);
    }

    #[test]
    fn threshold() {
        let pli = Pipeline::sse2().unwrap();
        super::test_threshold::<U32, _>(&pli);
        super::test_threshold::<U16, _>(&pli);
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::*;

    #[test]
    fn score() {
        let pli = Pipeline::avx2().unwrap();
        super::test_score::<U32, _>(&pli);
    }

    #[test]
    fn score_rows() {
        let pli = Pipeline::avx2().unwrap();
        super::test_score_rows::<U32, _>(&pli);
    }

    #[test]
    fn score_discrete() {
        let pli = Pipeline::avx2().unwrap();
        super::test_score_discrete(&pli);
    }

    #[test]
    fn argmax() {
        let pli = Pipeline::avx2().unwrap();
        super::test_argmax::<U32, _>(&pli);
    }

    #[test]
    fn threshold() {
        let pli = Pipeline::avx2().unwrap();
        super::test_threshold::<U32, _>(&pli);
    }
}

#[cfg(target_feature = "neon")]
mod neon {
    use super::*;

    #[test]
    fn score() {
        let pli = Pipeline::neon().unwrap();
        super::test_score::<U16, _>(&pli);
    }

    #[test]
    fn score_rows() {
        let pli = Pipeline::neon().unwrap();
        super::test_score_rows::<U16, _>(&pli);
    }

    #[test]
    fn score_discrete() {
        let pli = Pipeline::neon().unwrap();
        super::test_score_discrete::<U32, _>(&pli);
        super::test_score_discrete::<U16, _>(&pli);
    }

    #[test]
    fn argmax() {
        let pli = Pipeline::neon().unwrap();
        super::test_argmax::<U16, _>(&pli);
    }

    #[test]
    fn threshold() {
        let pli = Pipeline::neon().unwrap();
        super::test_threshold::<U16, _>(&pli);
    }
}
