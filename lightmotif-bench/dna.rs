#![feature(test)]

extern crate bio;
extern crate lightmotif;
extern crate test;

use lightmotif::abc::Background;
use lightmotif::abc::Dna;
use lightmotif::num::ArrayLength;
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

mod scanner {
    use super::*;

    /// Bench how long `Scanner::max_by` takes to find the highest hit using
    /// the score as a comparison key.
    #[bench]
    fn max_by(bencher: &mut test::Bencher) {
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
        bencher.iter(|| {
            Scanner::new(&pssm, &striped).max_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        });
        bencher.bytes = seq.len() as u64;
    }

    /// Bench how long `Scanner::max` takes to find the highest hit using
    /// the custom implementation of `Iterator::max` sorting hits by score
    /// and position keys.
    #[bench]
    fn max(bencher: &mut test::Bencher) {
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
        bencher.iter(|| Scanner::new(&pssm, &striped).max().unwrap());
        bencher.bytes = seq.len() as u64;
    }
}

mod f32 {
    use super::*;

    /// Bench how long `Pipeline::score` and `Pipeline::argmax` take for
    /// an arbitrary pipeline.
    fn bench_lightmotif<
        C: StrictlyPositive + ArrayLength,
        P: Score<f32, Dna, C> + Maximum<f32, C>,
    >(
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
        bencher.iter(|| {
            test::black_box(pli.score_into(&pssm, &striped, &mut scores));
            scores.offset(test::black_box(pli.argmax(&scores).unwrap()));
        });
    }

    /// Bench how long `Pipeline::<_, Generic>` takes.
    #[bench]
    fn generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench_lightmotif::<U1, _>(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Sse2>` takes.
    #[cfg(target_feature = "sse2")]
    #[bench]
    fn sse2(bencher: &mut test::Bencher) {
        let pli = Pipeline::sse2().unwrap();
        bench_lightmotif::<U16, _>(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Dispatch>` takes.
    #[bench]
    fn dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench_lightmotif(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Avx2>` takes.
    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench_lightmotif(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Neon>` takes.
    #[cfg(target_feature = "neon")]
    #[bench]
    fn neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench_lightmotif(bencher, &pli);
    }
}

mod u8 {
    use super::*;

    /// Bench how long `Pipeline::score` and `Pipeline::argmax` take for
    /// an arbitrary pipeline using discrete scores.
    fn bench_lightmotif_discrete<
        C: StrictlyPositive + ArrayLength,
        P: Score<u8, Dna, C> + Maximum<u8, C>,
    >(
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
        scores.resize(striped.matrix().rows(), striped.len());
        bencher.bytes = seq.len() as u64;
        bencher.iter(|| {
            test::black_box(pli.score_into(&dm, &striped, &mut scores));
            scores.offset(test::black_box(pli.argmax(&scores).unwrap()));
        });
    }

    /// Bench how long `Pipeline::<_, Generic>` takes using `u8` scores.
    #[bench]
    fn generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench_lightmotif_discrete::<U1, _>(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Sse2>` takes using `u8` scores.
    #[cfg(target_feature = "sse2")]
    #[bench]
    fn sse2(bencher: &mut test::Bencher) {
        let pli = Pipeline::sse2().unwrap();
        bench_lightmotif_discrete::<U16, _>(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Generic>` takes using `u8` scores.
    #[bench]
    fn dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench_lightmotif_discrete(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Generic>` takes using `u8` scores.
    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench_lightmotif_discrete(bencher, &pli);
    }

    /// Bench how long `Pipeline::<_, Neon>` takes using `u8` scores.
    #[cfg(target_feature = "neon")]
    #[bench]
    fn neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench_lightmotif_discrete(bencher, &pli);
    }
}

mod external {

    use super::*;

    #[bench]
    fn bio(bencher: &mut test::Bencher) {
        use bio::pattern_matching::pssm::DNAMotif;
        use bio::pattern_matching::pssm::Motif;

        let seq = &SEQUENCE[..N];

        let pssm = DNAMotif::from_seqs(
            vec![b"GTTGACCTTATCAAC".to_vec(), b"GTTGATCCAGTCAAC".to_vec()].as_ref(),
            None,
        )
        .unwrap();

        let mut best = 0;
        bencher.bytes = seq.len() as u64;
        bencher.iter(|| best = test::black_box(pssm.score(seq.as_bytes()).unwrap()).loc);

        println!("best: {:?}", best);
        assert_eq!(best, 391677);
    }

    #[bench]
    fn naive(bencher: &mut test::Bencher) {
        let seq = &SEQUENCE[..N];
        let encoded = EncodedSequence::<Dna>::encode(seq).unwrap();

        let bg = Background::<Dna>::uniform();
        let cm = CountMatrix::<Dna>::from_sequences([
            EncodedSequence::encode("GTTGACCTTATCAAC").unwrap(),
            EncodedSequence::encode("GTTGATCCAGTCAAC").unwrap(),
        ])
        .unwrap();
        let pbm = cm.to_freq(0.1);
        let pssm = pbm.to_scoring(bg);

        let mut best = 0;
        bencher.bytes = seq.len() as u64;
        bencher.iter(|| {
            best = 0;
            let mut score_best = -f32::INFINITY;
            for i in 0..encoded.len() - pssm.len() + 1 {
                let mut score = 0.0;
                for j in 0..pssm.len() {
                    score += pssm.matrix()[j][encoded[i + j] as usize];
                }
                if score > score_best {
                    score_best = score;
                    best = i;
                }
            }
        });

        assert_eq!(best, 391677);
    }
}
