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

    if striped.data.rows() > 0 {
        assert_eq!(striped.data[0][0], Nucleotide::A);
    }
    if striped.data.rows() > 1 {
        assert_eq!(striped.data[1][0], Nucleotide::T);
    }
    if striped.data.rows() > 2 {
        assert_eq!(striped.data[2][0], Nucleotide::G);
    }
    if striped.data.rows() > 3 {
        assert_eq!(striped.data[3][0], Nucleotide::T);
    }

    for (i, &c) in encoded.iter().into_iter().enumerate() {
        assert_eq!(
            striped.data[i % striped.data.rows()][i / striped.data.rows()],
            c
        );
    }

    for i in sequence.len()..striped.data.rows() * striped.data.columns() {
        assert_eq!(
            striped.data[i % striped.data.rows()][i / striped.data.rows()],
            Nucleotide::default(),
        )
    }
}

#[test]
fn test_stripe_generic() {
    let pli = Pipeline::generic();
    test_stripe::<U32, _>(&pli, S1);
    test_stripe::<U32, _>(&pli, S2);
    test_stripe::<U16, _>(&pli, S1);
    test_stripe::<U16, _>(&pli, S2);
}

#[test]
fn test_stripe_dispatch() {
    let pli = Pipeline::dispatch();
    test_stripe(&pli, S1);
    test_stripe(&pli, S2);
}

// #[cfg(target_feature = "sse2")]
// #[test]
// fn test_stripe_sse2() {
//     let pli = Pipeline::sse2().unwrap();
//     test_stripe(&pli);
// }

#[cfg(target_feature = "avx2")]
#[test]
fn test_stripe_avx2() {
    let pli = Pipeline::avx2().unwrap();
    test_stripe(&pli, S1);
    test_stripe(&pli, S2);
}

// #[cfg(target_feature = "neon")]
// #[test]
// fn test_stripe_neon() {
//     let pli = Pipeline::neon().unwrap();
//     test_stripe(&pli);
// }
