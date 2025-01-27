#![allow(unused)]

use std::str::FromStr;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::err::InvalidData;
use lightmotif::num::ArrayLength;
use lightmotif::num::PowerOfTwo;
use lightmotif::num::Unsigned;
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

pub fn symbol<A: Alphabet>(input: &str) -> IResult<&str, A::Symbol> {
    map_res(anychar, A::Symbol::from_char).parse(input)
}

pub fn counts(input: &str) -> IResult<&str, Vec<u32>> {
    delimited(
        delimited(space0, tag("["), space0),
        separated_list0(space1, nom::character::complete::u32),
        delimited(space0, tag("]"), space0),
    )
    .parse(input)
}

pub fn matrix_column<A: Alphabet>(input: &str) -> IResult<&str, (A::Symbol, Vec<u32>)> {
    terminated(separated_pair(symbol::<A>, space1, counts), line_ending).parse(input)
}

pub fn build_matrix<A: Alphabet>(
    input: Vec<(A::Symbol, Vec<u32>)>,
) -> Result<DenseMatrix<u32, A::K>, InvalidData> {
    let mut done = vec![false; A::K::USIZE];
    let mut matrix = DenseMatrix::new(input[0].1.len());

    for (s, counts) in input {
        // check that symbol does not appear in duplicate
        if done[s.as_index()] {
            return Err(InvalidData);
        }
        // check that array length is consistent
        if counts.len() != matrix.rows() {
            return Err(InvalidData);
        }
        // copy frequencies into matrix
        for (i, x) in counts.into_iter().enumerate() {
            matrix[i][s.as_index()] = x
        }
        done[s.as_index()] = true;
    }

    Ok(matrix)
}

pub fn matrix<A: Alphabet>(input: &str) -> IResult<&str, DenseMatrix<u32, A::K>> {
    map_res(nom::multi::many1(matrix_column::<A>), build_matrix::<A>).parse(input)
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

    // preceded( tag(">"), nom::bytes::complete::take_while(|c| !c.is_ascii_whitespace())  );
    // map(terminated(not_line_ending, line_ending), |s: &str| s.trim())(input)
    // unimplemented!()
}

pub fn record<A: Alphabet>(input: &str) -> IResult<&str, Record<A>> {
    let (input, (id, description)) = header(input)?;
    let (input, matrix) = map_res(matrix::<A>, CountMatrix::<A>::new).parse(input)?;

    Ok((
        input,
        Record {
            id: id.to_string(),
            description: description.map(String::from),
            matrix,
        },
    ))

    // terminated(
    //     pair(
    //         header,
    //         map_res(matrix::<A, DefaultAlignment>, CountMatrix::<A>::new),
    //     ),
    //     nom::combinator::opt(line_ending),
    // )(input)
    // unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;

    #[test]
    fn test_matrix_column() {
        let (rest, (symbol, counts)) =
            super::matrix_column::<Dna>("T [10 12  4  1  2  2  0  0  0  8 13 ]\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(symbol, Nucleotide::T);
        assert_eq!(counts, vec![10, 12, 4, 1, 2, 2, 0, 0, 0, 8, 13]);
    }

    #[test]
    fn test_matrix() {
        let (_rest, matrix) = super::matrix::<Dna>(concat!(
            "A [10 12  4  1  2  2  0  0  0  8 13 ]\n",
            "C [ 2  2  7  1  0  8  0  0  1  2  2 ]\n",
            "G [ 3  1  1  0 23  0 26 26  0  0  4 ]\n",
            "T [11 11 14 24  1 16  0  0 25 16  7 ]\n",
        ))
        .unwrap();
        assert_eq!(&matrix[0][..4], &[10, 2, 11, 3]);
    }

    #[test]
    fn test_record() {
        let (_rest, record) = super::record::<Dna>(concat!(
            ">MA0002.1 RUNX1\n",
            "A [10 12  4  1  2  2  0  0  0  8 13 ]\n",
            "C [ 2  2  7  1  0  8  0  0  1  2  2 ]\n",
            "G [ 3  1  1  0 23  0 26 26  0  0  4 ]\n",
            "T [11 11 14 24  1 16  0  0 25 16  7 ]\n",
        ))
        .unwrap();
        assert_eq!(&record.id, "MA0002.1");
        assert_eq!(record.description.as_deref(), Some("RUNX1"));
        assert_eq!(&record.matrix[0][..4], &[10, 2, 11, 3]);
    }
}
