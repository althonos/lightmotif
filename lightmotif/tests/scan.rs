use lightmotif::abc::Dna;
use lightmotif::scan::Scanner;
use lightmotif::seq::EncodedSequence;
use lightmotif::CountMatrix;

const SEQUENCE: &str = include_str!("../benches/ecoli.txt");
const PATTERNS: &[&str] = &["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC", "GTTGATCCAGTAAAT"];

#[test]
fn scanner_consistency() {
    const THRESHOLD: f32 = 5.0;

    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();
    striped.configure_wrap(PATTERNS[0].len());

    let pssm = PATTERNS
        .iter()
        .map(EncodedSequence::<Dna>::encode_lossy)
        .collect::<Result<CountMatrix<Dna>, _>>()
        .unwrap()
        .to_freq(0.1)
        .to_scoring(None);

    let scores_brute: Vec<f32> = pssm.score(&striped).into();
    let tresholded = scores_brute
        .iter()
        .enumerate()
        .filter(|(_, x)| **x >= THRESHOLD)
        .collect::<Vec<_>>();

    for block_size in [1, 16, 32, 64, 128, 256] {
        let mut scanner = Scanner::new(&pssm, &striped);
        scanner.block_size(block_size).threshold(THRESHOLD);

        let mut scanner_hits = scanner.collect::<Vec<_>>();
        scanner_hits.sort_by_key(|hit| hit.position());

        assert_eq!(scanner_hits.len(), tresholded.len());
        for (hit, (pos, score)) in scanner_hits.iter().zip(tresholded.iter()) {
            assert_eq!(hit.position(), *pos);
            assert_eq!(hit.score(), **score);
        }
    }
}

#[test]
fn scanner_consistency_infscores() {
    const THRESHOLD: f32 = 5.0;

    let encoded = EncodedSequence::<Dna>::encode(SEQUENCE).unwrap();
    let mut striped = encoded.to_striped();
    striped.configure_wrap(PATTERNS[0].len());

    let pssm = PATTERNS
        .iter()
        .map(EncodedSequence::<Dna>::encode_lossy)
        .collect::<Result<CountMatrix<Dna>, _>>()
        .unwrap()
        .to_freq(0.0)
        .to_scoring(None);

    let scores_brute: Vec<f32> = pssm.score(&striped).into();
    let tresholded = scores_brute
        .iter()
        .enumerate()
        .filter(|(_, x)| **x >= THRESHOLD)
        .collect::<Vec<_>>();

    for block_size in [1, 16, 32, 64, 128, 256] {
        let mut scanner = Scanner::new(&pssm, &striped);
        scanner.block_size(block_size).threshold(THRESHOLD);

        let mut scanner_hits = scanner.collect::<Vec<_>>();
        scanner_hits.sort_by_key(|hit| hit.position());

        assert_eq!(scanner_hits.len(), tresholded.len());
        for (hit, (pos, score)) in scanner_hits.iter().zip(tresholded.iter()) {
            assert_eq!(hit.position(), *pos);
            assert_eq!(hit.score(), **score);
        }
    }
}
