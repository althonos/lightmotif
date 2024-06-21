extern crate lightmotif;
extern crate typenum;

use lightmotif::abc::Dna;
use lightmotif::abc::Nucleotide;
use lightmotif::abc::Nucleotide::*;
use lightmotif::pli::Encode;
use lightmotif::pli::Pipeline;

const SEQUENCE: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";
const UNKNOWNS: &str = "ATGTCCCAACAACGATACCNN..................NNNNNNNNATGCAGATTCCCAGGCG";

#[rustfmt::skip]
const EXPECTED: &[Nucleotide] = &[
    A, T, G, T, C, C, C, A, A, C, A, A, C, G, A, T, A, C, C, C, C, G, A, G, C, C, C, A, T, C, G, C, C, G, T, C, A, T, C, G, G, C, T, C, G, G, C, A, T, G, C, A, G, A, T, T, C, C, C, A, G, G, C, G 
];

fn test_encode_sequence<P: Encode<Dna>>(pli: &P) {
    let encoded = pli.encode(SEQUENCE).unwrap();
    assert_eq!(encoded, EXPECTED);
}

fn test_encode_unknown<P: Encode<Dna>>(pli: &P) {
    let err = pli.encode(UNKNOWNS).unwrap_err();
    assert_eq!(err.0, '.');
}

mod generic {
    use super::*;

    #[test]
    fn sequence() {
        let pli = Pipeline::generic();
        test_encode_sequence(&pli);
    }

    #[test]
    fn unknown() {
        let pli = Pipeline::generic();
        test_encode_unknown(&pli);
    }
}

mod dispatch {
    use super::*;

    #[test]
    fn sequence() {
        let pli = Pipeline::dispatch();
        test_encode_sequence(&pli);
    }

    #[test]
    fn unknown() {
        let pli = Pipeline::dispatch();
        test_encode_unknown(&pli);
    }
}

#[cfg(target_feature = "sse2")]
mod sse2 {
    use super::*;

    #[test]
    fn sequence() {
        let pli = Pipeline::sse2().unwrap();
        test_encode_sequence(&pli);
    }

    #[test]
    fn unknown() {
        let pli = Pipeline::sse2().unwrap();
        test_encode_unknown(&pli);
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::*;

    #[test]
    fn sequence() {
        let pli = Pipeline::avx2().unwrap();
        test_encode_sequence(&pli);
    }

    #[test]
    fn unknown() {
        let pli = Pipeline::avx2().unwrap();
        test_encode_unknown(&pli);
    }
}

#[cfg(target_feature = "neon")]
mod neon {
    use super::*;

    #[test]
    fn sequence() {
        let pli = Pipeline::neon().unwrap();
        test_encode_sequence(&pli);
    }

    #[test]
    fn unknown() {
        let pli = Pipeline::neon().unwrap();
        test_encode_unknown(&pli);
    }
}
