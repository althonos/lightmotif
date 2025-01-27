#![allow(unused)]

use std::str::FromStr;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::err::InvalidData;
use lightmotif::num::ArrayLength;
use lightmotif::num::PowerOfTwo;
use lightmotif::num::Unsigned;
use lightmotif::pwm::FrequencyMatrix;
use nom::bytes::complete::take_while;
use nom::character::complete::anychar;
use nom::character::complete::line_ending;
use nom::character::complete::not_line_ending;
use nom::character::complete::tab;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::multi::many1;
use nom::number::complete::float;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::separated_pair;
use nom::sequence::terminated;
use nom::IResult;
use nom::Parser;

pub fn symbol<A: Alphabet>(input: &str) -> IResult<&str, A::Symbol> {
    map_res(anychar, A::Symbol::from_char).parse(input)
}

pub fn frequencies(input: &str) -> IResult<&str, Vec<f32>> {
    many1(preceded(tab, float)).parse(input)
}

pub fn matrix_column<A: Alphabet>(input: &str) -> IResult<&str, (A::Symbol, Vec<f32>)> {
    terminated(
        separated_pair(
            symbol::<A>,
            nom::character::complete::char(':'),
            frequencies,
        ),
        line_ending,
    )
    .parse(input)
}

pub fn build_matrix<A: Alphabet>(
    input: Vec<(A::Symbol, Vec<f32>)>,
) -> Result<DenseMatrix<f32, A::K>, InvalidData> {
    let mut done = vec![false; A::K::USIZE];
    let mut matrix = DenseMatrix::new(input[0].1.len());

    for (s, frequencies) in input {
        // check that symbol does not appear in duplicate
        if done[s.as_index()] {
            return Err(InvalidData);
        }
        // check that array length is consistent
        if frequencies.len() != matrix.rows() {
            return Err(InvalidData);
        }
        // copy frequencies into matrix
        for (i, x) in frequencies.into_iter().enumerate() {
            matrix[i][s.as_index()] = x
        }
        done[s.as_index()] = true;
    }

    Ok(matrix)
}

pub fn matrix<A: Alphabet>(input: &str) -> IResult<&str, DenseMatrix<f32, A::K>> {
    map_res(nom::multi::many1(matrix_column::<A>), build_matrix::<A>).parse(input)
}

pub fn id(input: &str) -> IResult<&str, &str> {
    map(terminated(not_line_ending, line_ending), |s: &str| s.trim()).parse(input)
}

pub fn record<A: Alphabet>(input: &str) -> IResult<&str, (&str, FrequencyMatrix<A>)> {
    terminated(
        pair(id, map_res(matrix::<A>, FrequencyMatrix::<A>::new)),
        line_ending,
    )
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;

    #[test]
    fn test_matrix_column() {
        let (rest, (symbol, freqs)) = super::matrix_column::<Dna>("T:	0.100	0.200	0.03\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(symbol, Nucleotide::T);
        assert_eq!(freqs, vec![0.100, 0.200, 0.03]);
    }

    #[test]
    fn test_matrix() {
        let (_rest, matrix) = super::matrix::<Dna>(concat!(
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
        ))
        .unwrap();
        assert_eq!(&matrix[0][..4], &[0.179, 0.268, 0.170, 0.383]);
    }

    #[test]
    fn test_record() {
        let (_rest, (name, matrix)) = super::record::<Dna>(concat!(
            "test\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
            "\n",
        ))
        .unwrap();
        assert_eq!(name, "test");
        assert_eq!(&matrix[0][..4], &[0.179, 0.268, 0.170, 0.383]);
    }
}
