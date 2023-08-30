#![feature(test)]

extern crate lightmotif;
extern crate test;

use lightmotif::num::StrictlyPositive;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Score;
use lightmotif::pli::StripedScores;
use lightmotif::pwm::CountMatrix;
use lightmotif::seq::EncodedSequence;

mod dna {

    use super::*;
    use lightmotif::abc::Dna;

    const SEQUENCE: &str = include_str!("ecoli.txt");

    fn bench<C: StrictlyPositive, P: Score<Dna, C>>(bencher: &mut test::Bencher, pli: &P) {
        let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
        let mut striped = encoded.to_striped();

        let cm = CountMatrix::<Dna>::from_sequences(&[
            EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
            EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
        ])
        .unwrap();
        let pbm = cm.to_freq(0.1);
        let pssm = pbm.to_scoring(None);

        striped.configure(&pssm);
        let mut scores = StripedScores::empty();
        bencher.bytes = SEQUENCE.len() as u64;
        bencher.iter(|| {
            pli.score_into(&striped, &pssm, &mut scores);
            test::black_box(())
        });
    }

    #[bench]
    fn bench_generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench::<U32, _>(bencher, &pli);
    }

    #[bench]
    fn bench_dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench(bencher, &pli);
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

    #[cfg(target_feature = "neon")]
    #[bench]
    fn bench_neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench::<U16, _>(bencher, &pli);
    }
}

mod protein {

    use super::*;
    use lightmotif::abc::Protein;

    const SEQUENCE: &str = include_str!("abyB1.txt");

    fn bench<C: StrictlyPositive, P: Score<Protein, C>>(bencher: &mut test::Bencher, pli: &P) {
        let encoded = EncodedSequence::<Protein>::encode(SEQUENCE).unwrap();
        let mut striped = encoded.to_striped();

        let cm = CountMatrix::<Protein>::from_sequences(&[
            EncodedSequence::encode("SFKELGFDSLTAVELRNRLAAAT").unwrap(),
            EncodedSequence::encode("AFKELGFDSLAAIQLRNRLLADV").unwrap(),
            EncodedSequence::encode("PSRRLGFDSLTAVELRNQLAAST").unwrap(),
            EncodedSequence::encode("AFREIGFDSLTAVELRNRLGAAA").unwrap(),
            EncodedSequence::encode("SLMEEGLDSLAAVELGGTLQRDT").unwrap(),
            EncodedSequence::encode("GFFDLGMDSLMAVELRRRIEQGV").unwrap(),
        ])
        .unwrap();
        let pbm = cm.to_freq(0.1);
        let pssm = pbm.to_scoring(None);

        striped.configure(&pssm);
        let mut scores = StripedScores::empty();
        bencher.bytes = SEQUENCE.len() as u64;
        bencher.iter(|| {
            pli.score_into(&striped, &pssm, &mut scores);
            test::black_box(())
        });
    }

    #[bench]
    fn bench_generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench::<U32, _>(bencher, &pli);
    }

    #[bench]
    fn bench_dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench(bencher, &pli);
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

    #[cfg(target_feature = "neon")]
    #[bench]
    fn bench_neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench::<U16, _>(bencher, &pli);
    }
}
