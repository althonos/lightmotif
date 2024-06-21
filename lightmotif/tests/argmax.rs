extern crate lightmotif;

use lightmotif::abc::Background;
use lightmotif::abc::Dna;
use lightmotif::num::StrictlyPositive;
use lightmotif::num::U1;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::Maximum;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::Stripe;
use lightmotif::pwm::CountMatrix;
use lightmotif::scan::Scanner;
use lightmotif::scores::StripedScores;
use lightmotif::seq::EncodedSequence;

const SEQUENCE: &str = include_str!("../benches/ecoli.txt");
const PATTERNS: &[&str] = &["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"];
const N: usize = SEQUENCE.len() / 10;

fn test_argmax_f32<C: StrictlyPositive, P: Maximum<f32, C>>(pli: &P) {
    let generic = Pipeline::generic();

    let seq = &SEQUENCE[..N];
    let encoded = EncodedSequence::<Dna>::encode(seq).unwrap();
    let mut striped = generic.stripe(encoded);

    let bg = Background::<Dna>::uniform();
    let cm = PATTERNS
        .iter()
        .map(EncodedSequence::encode)
        .map(Result::unwrap)
        .collect::<Result<CountMatrix<Dna>, _>>()
        .unwrap();
    let pbm = cm.to_freq(0.1);
    let pssm = pbm.to_scoring(bg);

    striped.configure(&pssm);
    let scores: StripedScores<f32, C> = generic.score(pssm, striped);

    let best = scores
        .unstripe()
        .iter()
        .cloned()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .unwrap();
    let m = pli.argmax(&scores).unwrap();
    assert_eq!(scores.offset(m), best.0);
    assert_eq!(scores.matrix()[m], best.1);
}

mod generic {
    use super::*;

    #[test]
    fn argmax_f32() {
        let pli = Pipeline::<Dna, _>::generic();
        super::test_argmax_f32::<U1, _>(&pli);
        super::test_argmax_f32::<U16, _>(&pli);
        super::test_argmax_f32::<U32, _>(&pli);
    }
}

mod dispatch {
    use super::*;

    #[test]
    fn argmax_f32() {
        let pli = Pipeline::<Dna, _>::dispatch();
        super::test_argmax_f32(&pli);
    }

    #[test]
    fn scanner_max() {
        let generic = Pipeline::generic();

        let seq = &SEQUENCE[..N];
        let encoded = EncodedSequence::<Dna>::encode(seq).unwrap();
        let mut striped = generic.stripe(encoded);

        let bg = Background::<Dna>::uniform();
        let cm = PATTERNS
            .iter()
            .map(EncodedSequence::encode)
            .map(Result::unwrap)
            .collect::<Result<CountMatrix<Dna>, _>>()
            .unwrap();
        let pbm = cm.to_freq(0.1);
        let pssm = pbm.to_scoring(bg);

        striped.configure(&pssm);
        let scores: StripedScores<f32, _> = generic.score(&pssm, &striped);
        let best = scores
            .unstripe()
            .iter()
            .cloned()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();

        let m = Scanner::new(&pssm, &striped).max().unwrap();
        assert_eq!(m.position, best.0);
        assert_eq!(m.score, best.1);
    }
}

#[cfg(target_feature = "sse2")]
mod sse2 {
    use super::*;

    #[test]
    fn argmax_f32() {
        let pli = Pipeline::<Dna, _>::sse2().unwrap();
        super::test_argmax_f32::<U16, _>(&pli);
        super::test_argmax_f32::<U32, _>(&pli);
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::*;

    #[test]
    fn argmax_f32() {
        let pli = Pipeline::<Dna, _>::avx2().unwrap();
        super::test_argmax_f32(&pli);
    }
}

#[cfg(target_feature = "neon")]
mod neon {
    use super::*;

    #[test]
    fn argmax_f32() {
        let pli = Pipeline::<Dna, _>::neon().unwrap();
        super::test_argmax_f32::<U16, _>(&pli);
        super::test_argmax_f32::<U32, _>(&pli);
    }
}
