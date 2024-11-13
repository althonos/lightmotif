#![feature(test)]

extern crate lightmotif;
extern crate test;

mod dna {

    use super::*;
    use lightmotif::abc::Dna;

    const SEQUENCE: &str = include_str!("ecoli.txt");

    mod single {
        use super::*;

        use lightmotif::abc::Nucleotide;
        use lightmotif::seq::EncodedSequence;
        use lightmotif::seq::SymbolCount;

        #[bench]
        fn seq(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Dna>::encode(&SEQUENCE).unwrap();
            bencher.iter(|| {
                test::black_box(seq.count_symbol(Nucleotide::C));
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }

        #[bench]
        fn striped(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Dna>::encode(&SEQUENCE).unwrap();
            let striped = seq.to_striped();
            bencher.iter(|| {
                test::black_box(striped.count_symbol(Nucleotide::C));
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }
    }

    mod all {
        use super::*;

        use lightmotif::seq::EncodedSequence;
        use lightmotif::seq::SymbolCount;

        #[bench]
        fn seq(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Dna>::encode(&SEQUENCE).unwrap();
            bencher.iter(|| {
                test::black_box(seq.count_symbols());
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }

        #[bench]
        fn striped(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Dna>::encode(&SEQUENCE).unwrap();
            let striped = seq.to_striped();
            bencher.iter(|| {
                test::black_box(striped.count_symbols());
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }
    }
}

mod protein {

    use super::*;
    use lightmotif::abc::Protein;

    const SEQUENCE: &str = include_str!("abyB1.txt");

    mod single {
        use super::*;

        use lightmotif::abc::AminoAcid;
        use lightmotif::seq::EncodedSequence;
        use lightmotif::seq::SymbolCount;

        #[bench]
        fn seq(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Protein>::encode(&SEQUENCE).unwrap();
            bencher.iter(|| {
                test::black_box(seq.count_symbol(AminoAcid::E));
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }

        #[bench]
        fn striped(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Protein>::encode(&SEQUENCE).unwrap();
            let striped = seq.to_striped();
            bencher.iter(|| {
                test::black_box(striped.count_symbol(AminoAcid::E));
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }
    }

    mod all {
        use super::*;

        use lightmotif::abc::Protein;
        use lightmotif::seq::EncodedSequence;
        use lightmotif::seq::SymbolCount;

        #[bench]
        fn seq(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Protein>::encode(&SEQUENCE).unwrap();
            bencher.iter(|| {
                test::black_box(seq.count_symbols());
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }

        #[bench]
        fn striped(bencher: &mut test::Bencher) {
            let seq = EncodedSequence::<Protein>::encode(&SEQUENCE).unwrap();
            let striped = seq.to_striped();
            bencher.iter(|| {
                test::black_box(striped.count_symbols());
            });
            bencher.bytes = SEQUENCE.as_bytes().len() as u64;
        }
    }
}
