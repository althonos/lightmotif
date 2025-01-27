//! Parser implementation for matrices in JASPAR (raw) format.
//! The [JASPAR database](https://jaspar.elixir.no/docs/) stores manually
//! curated DNA-binding sites as count matrices.
//!
//! The JASPAR files contains a FASTA-like header line for each record,
//! followed by one line per symbol storing tab-separated counts at each
//! position. The raw version only allows 4 lines per motif, which represent
//! the counts for the A, C, G and T nucleotides:
//! ```text
//! >MA1104.2 GATA6
//! 22320 20858 35360  5912 4535  2560  5044 76686  1507  1096 13149 18911 22172
//! 16229 14161 13347 11831 62936 1439  1393   815   852 75930  3228 19054 17969
//! 13432 11894 10394  7066 6459   580   615   819   456   712  1810 18153 11605
//! 27463 32531 20343 54635 5514 74865 72392  1124 76629  1706 61257 23326 27698
//! ```
//!
#![allow(unused)]

use std::str::FromStr;

use generic_array::GenericArray;
use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::abc::Nucleotide;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::err::InvalidData;
use lightmotif::num::PowerOfTwo;
use lightmotif::num::Unsigned;
use lightmotif::num::U4;
use lightmotif::pwm::CountMatrix;
use lightmotif::pwm::FrequencyMatrix;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_while;
use nom::character::complete::anychar;
use nom::character::complete::line_ending;
use nom::character::complete::not_line_ending;
use nom::character::complete::space0;
use nom::character::complete::space1;
use nom::character::complete::tab;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::combinator::opt;
use nom::multi::many1;
use nom::multi::separated_list0;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::separated_pair;
use nom::sequence::terminated;
use nom::IResult;
use nom::Parser;

use super::Record;

pub fn counts(input: &str) -> IResult<&str, Vec<u32>> {
    preceded(
        opt(space1),
        separated_list0(space1, nom::character::complete::u32),
    )
    .parse(input)
}

pub fn matrix_column(input: &str) -> IResult<&str, Vec<u32>> {
    terminated(counts, line_ending).parse(input)
}

pub fn build_matrix(
    input: GenericArray<Vec<u32>, U4>,
    symbols: &[<Dna as Alphabet>::Symbol],
) -> Result<DenseMatrix<u32, <Dna as Alphabet>::K>, InvalidData> {
    let mut matrix = DenseMatrix::new(input[0].len());
    for (counts, s) in input.as_slice().into_iter().zip(symbols) {
        // check that array length is consistent
        if counts.len() != matrix.rows() {
            return Err(InvalidData);
        }
        // copy frequencies into matrix
        for (i, x) in counts.into_iter().enumerate() {
            matrix[i][s.as_index()] = *x
        }
    }
    Ok(matrix)
}

pub fn matrix(input: &str) -> IResult<&str, DenseMatrix<u32, <Dna as Alphabet>::K>> {
    let (input, a) = matrix_column(input)?;
    let (input, c) = matrix_column(input)?;
    let (input, g) = matrix_column(input)?;
    let (input, t) = matrix_column(input)?;

    let len = a.len();
    let g = GenericArray::from([a, c, g, t]);
    let symbols = &[Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T];

    match build_matrix(g, symbols) {
        Ok(x) => Ok((input, x)),
        Err(e) => unimplemented!(),
    }
}

pub fn header(input: &str) -> IResult<&str, (&str, Option<&str>)> {
    let (input, id) = preceded(
        tag(">"),
        nom::bytes::complete::take_while(|c: char| !c.is_ascii_whitespace()),
    )
    .parse(input)?;

    let (input, accession) = nom::bytes::complete::take_until("\n")(input)?;
    let (input, _) = line_ending(input)?;

    let accession = accession.trim();
    if accession.is_empty() {
        Ok((input, (id, None)))
    } else {
        Ok((input, (id, Some(accession))))
    }
}

pub fn record(input: &str) -> IResult<&str, Record> {
    let (input, (id, description)) = header(input)?;
    let (input, matrix) = map_res(matrix, CountMatrix::new).parse(input)?;

    Ok((
        input,
        Record {
            id: id.to_string(),
            description: description.map(String::from),
            matrix,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;

    #[test]
    fn test_matrix_column() {
        let (rest, counts) = super::matrix_column("10 12  4  1  2  2  0  0  0  8 13\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(counts, vec![10, 12, 4, 1, 2, 2, 0, 0, 0, 8, 13]);
    }

    #[test]
    fn test_matrix() {
        let (_rest, matrix) = super::matrix(concat!(
            "10 12  4  1  2  2  0  0  0  8 13\n",
            " 2  2  7  1  0  8  0  0  1  2  2\n",
            " 3  1  1  0 23  0 26 26  0  0  4\n",
            "11 11 14 24  1 16  0  0 25 16  7\n",
        ))
        .unwrap();
        assert_eq!(&matrix[0][..4], &[10, 2, 11, 3]);
    }

    #[test]
    fn test_record() {
        let (_rest, record) = super::record(concat!(
            ">MA0002.1 RUNX1\n",
            "10 12  4  1  2  2  0  0  0  8 13\n",
            " 2  2  7  1  0  8  0  0  1  2  2\n",
            " 3  1  1  0 23  0 26 26  0  0  4\n",
            "11 11 14 24  1 16  0  0 25 16  7\n",
        ))
        .unwrap();
        assert_eq!(&record.id, "MA0002.1");
        assert_eq!(record.description.as_deref(), Some("RUNX1"));
        assert_eq!(&record.matrix[0][..4], &[10, 2, 11, 3]);
    }
}
