#![feature(test)]

extern crate lightmotif;
extern crate test;

use lightmotif::num::U32;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Stripe;
use lightmotif::scan::DefaultColumns;
use lightmotif::seq::EncodedSequence;

mod dna {

    use super::*;
    use lightmotif::abc::Dna;

    const SEQUENCE: &str = include_str!("ecoli.txt");

    fn bench<P: Stripe<Dna, DefaultColumns>>(bencher: &mut test::Bencher, pli: &P) {
        let seq = EncodedSequence::encode(SEQUENCE).unwrap();
        let mut dst = seq.to_striped();

        bencher.iter(|| {
            pli.stripe_into(&seq, &mut dst);
            test::black_box(());
        });
        bencher.bytes = SEQUENCE.as_bytes().len() as u64;
    }

    #[bench]
    fn generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench(bencher, &pli);
    }

    #[bench]
    fn dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench(bencher, &pli);
    }

    // #[cfg(target_feature = "sse2")]
    // #[bench]
    // fn sse2(bencher: &mut test::Bencher) {
    //     let pli = Pipeline::sse2().unwrap();
    //     bench(bencher, &pli);
    // }

    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench(bencher, &pli);
    }

    // #[cfg(target_feature = "neon")]
    // #[bench]
    // fn neon(bencher: &mut test::Bencher) {
    //     let pli = Pipeline::neon().unwrap();
    //     bench(bencher, &pli);
    // }
}

mod protein {

    use super::*;
    use lightmotif::abc::Protein;

    const SEQUENCE: &str = include_str!("abyB1.txt");

    fn bench<P: Stripe<Protein, U32>>(bencher: &mut test::Bencher, pli: &P) {
        let seq = EncodedSequence::encode(SEQUENCE).unwrap();
        let mut dst = seq.to_striped();

        bencher.iter(|| {
            pli.stripe_into(&seq, &mut dst);
            test::black_box(());
        });
        bencher.bytes = SEQUENCE.as_bytes().len() as u64;
    }

    #[bench]
    fn generic(bencher: &mut test::Bencher) {
        let pli = Pipeline::generic();
        bench(bencher, &pli);
    }

    #[bench]
    fn dispatch(bencher: &mut test::Bencher) {
        let pli = Pipeline::dispatch();
        bench(bencher, &pli);
    }

    // #[cfg(target_feature = "sse2")]
    // #[bench]
    // fn sse2(bencher: &mut test::Bencher) {
    //     let pli = Pipeline::sse2().unwrap();
    //     bench(bencher, &pli);
    // }

    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench(bencher, &pli);
    }

    // #[cfg(target_feature = "neon")]
    // #[bench]
    // fn neon(bencher: &mut test::Bencher) {
    //     let pli = Pipeline::neon().unwrap();
    //     bench(bencher, &pli);
    // }
}
