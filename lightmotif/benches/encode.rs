#![feature(test)]

extern crate lightmotif;
extern crate test;

use lightmotif::abc::Alphabet;
use lightmotif::pli::Encode;
use lightmotif::pli::Pipeline;

mod dna {

    use super::*;
    use lightmotif::abc::Dna;

    const SEQUENCE: &str = include_str!("ecoli.txt");

    fn bench<P: Encode<Dna>>(bencher: &mut test::Bencher, pli: &P) {
        let mut dst = vec![<Dna as Alphabet>::Symbol::default(); SEQUENCE.len()];
        bencher.iter(|| {
            test::black_box(pli.encode_into(SEQUENCE, &mut dst).unwrap());
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

    #[cfg(target_feature = "sse2")]
    #[bench]
    fn sse2(bencher: &mut test::Bencher) {
        let pli = Pipeline::sse2().unwrap();
        bench(bencher, &pli);
    }

    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench(bencher, &pli);
    }

    #[cfg(target_feature = "neon")]
    #[bench]
    fn neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench(bencher, &pli);
    }
}

mod protein {

    use super::*;
    use lightmotif::abc::Protein;

    const SEQUENCE: &str = include_str!("abyB1.txt");

    fn bench<P: Encode<Protein>>(bencher: &mut test::Bencher, pli: &P) {
        let mut dst = vec![<Protein as Alphabet>::Symbol::default(); SEQUENCE.len()];
        bencher.iter(|| {
            test::black_box(pli.encode_into(SEQUENCE, &mut dst).unwrap());
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

    #[cfg(target_feature = "sse2")]
    #[bench]
    fn sse2(bencher: &mut test::Bencher) {
        let pli = Pipeline::sse2().unwrap();
        bench(bencher, &pli);
    }

    #[cfg(target_feature = "avx2")]
    #[bench]
    fn avx2(bencher: &mut test::Bencher) {
        let pli = Pipeline::avx2().unwrap();
        bench(bencher, &pli);
    }

    #[cfg(target_feature = "neon")]
    #[bench]
    fn neon(bencher: &mut test::Bencher) {
        let pli = Pipeline::neon().unwrap();
        bench(bencher, &pli);
    }
}
