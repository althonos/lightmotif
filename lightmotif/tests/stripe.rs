extern crate lightmotif;
extern crate typenum;

use lightmotif::abc::Dna;
use lightmotif::abc::Nucleotide;
use lightmotif::num::NonZero;
use lightmotif::num::Unsigned;
use lightmotif::num::U16;
use lightmotif::num::U32;
use lightmotif::pli::Pipeline;
use lightmotif::pli::Stripe;
use lightmotif::seq::EncodedSequence;

const S1: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCGTTATTAT";
const S2: &str = "ATGTCCCAACAACGATACCCCGAGCCCATCGCCGTCATCGGCTCGGCATGCAGATTCCCAGGCG";

fn test_stripe<C: Unsigned + NonZero, P: Stripe<Dna, C>>(pli: &P, sequence: &str) {
    let encoded = EncodedSequence::<Dna>::encode(sequence).unwrap();
    let striped = pli.stripe(&encoded);

    let matrix = striped.matrix();
    if matrix.rows() > 0 {
        assert_eq!(matrix[0][0], Nucleotide::A);
    }
    if matrix.rows() > 1 {
        assert_eq!(matrix[1][0], Nucleotide::T);
    }
    if matrix.rows() > 2 {
        assert_eq!(matrix[2][0], Nucleotide::G);
    }
    if matrix.rows() > 3 {
        assert_eq!(matrix[3][0], Nucleotide::T);
    }

    for (i, &c) in encoded.iter().enumerate() {
        assert_eq!(matrix[i % matrix.rows()][i / matrix.rows()], c);
    }

    for i in sequence.len()..matrix.rows() * matrix.columns() {
        assert_eq!(
            matrix[i % matrix.rows()][i / matrix.rows()],
            Nucleotide::default(),
        )
    }
}

mod generic {
    use super::*;

    #[test]
    fn s1_c32() {
        let pli = Pipeline::generic();
        super::test_stripe::<U32, _>(&pli, S1);
    }

    #[test]
    fn s2_c32() {
        let pli = Pipeline::generic();
        super::test_stripe::<U32, _>(&pli, S2);
    }

    #[test]
    fn s1_c16() {
        let pli = Pipeline::generic();
        super::test_stripe::<U16, _>(&pli, S1);
    }

    #[test]
    fn s2_c16() {
        let pli = Pipeline::generic();
        super::test_stripe::<U16, _>(&pli, S2);
    }
}

mod dispatch {
    use super::*;

    #[test]
    fn s1_c32() {
        let pli = Pipeline::dispatch();
        super::test_stripe(&pli, S1);
    }

    #[test]
    fn s2_c32() {
        let pli = Pipeline::dispatch();
        super::test_stripe(&pli, S2);
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::*;

    #[test]
    fn s1_c32() {
        let pli = Pipeline::avx2().unwrap();
        super::test_stripe(&pli, S1);
    }

    #[test]
    fn s2_c32() {
        let pli = Pipeline::avx2().unwrap();
        super::test_stripe(&pli, S2);
    }
}
