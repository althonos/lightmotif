#![doc = include_str!("../README.md")]
#![allow(unused)]

extern crate memchr;

use nom::branch::alt;
use nom::bytes::complete::is_a;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_till;
use nom::bytes::complete::take_until;
use nom::bytes::complete::take_while;
use nom::bytes::complete::take_while1;
use nom::character::complete::anychar;
use nom::character::complete::line_ending;
use nom::character::complete::not_line_ending;
use nom::character::complete::space0;
use nom::character::streaming::space1;
use nom::combinator::eof;
use nom::combinator::map_res;
use nom::error::Error;
use nom::error::ErrorKind;
use nom::multi::count;
use nom::multi::many1;
use nom::multi::many_till;
use nom::multi::separated_list1;
use nom::sequence::delimited;
use nom::sequence::preceded;
use nom::sequence::terminated;
use nom::IResult;

use lightmotif::Alphabet;
use lightmotif::CountMatrix;
use lightmotif::DenseMatrix;
use lightmotif::Symbol;

pub struct TransfacMatrix<A: Alphabet, const K: usize> {
    counts: CountMatrix<A, K>,
}

fn parse_line(input: &str) -> IResult<&str, &str> {
    match memchr::memchr(b'\n', input.as_bytes()) {
        None => Err(nom::Err::Error(Error::new(input, ErrorKind::Verify))),
        Some(i) if i == input.len() => Ok(("", input)),
        Some(i) => {
            let (line, rest) = input.split_at(i + 1);
            Ok((rest, line))
        }
    }
}

fn parse_ac(input: &str) -> IResult<&str, &str> {
    let (input, line) = preceded(tag("AC"), parse_line)(input)?;
    Ok((input, line.trim()))
}

fn parse_id(input: &str) -> IResult<&str, &str> {
    let (input, line) = preceded(tag("ID"), parse_line)(input)?;
    Ok((input, line.trim()))
}

fn parse_bf(input: &str) -> IResult<&str, &str> {
    let (input, line) = preceded(tag("BF"), parse_line)(input)?;
    Ok((input, line.trim()))
}

fn parse_alphabet<S: Symbol>(input: &str) -> IResult<&str, Vec<S>> {
    delimited(
        alt((tag("PO"), tag("P0"))),
        preceded(
            space1,
            separated_list1(space1, map_res(anychar, S::from_char)),
        ),
        line_ending,
    )(input)
}

fn parse_row(input: &str, k: usize) -> IResult<&str, Vec<u32>> {
    delimited(
        nom::character::complete::u32,
        count(delimited(space0, nom::character::complete::u32, space0), k),
        parse_line,
    )(input)
}

fn parse_tag(input: &str) -> IResult<&str, &str> {
    nom::branch::alt((
        tag("BF"),
        tag("ID"),
        tag("XX"),
        tag("P0"),
        tag("PO"),
        tag("//"),
    ))(input)
}

pub fn parse_matrix<A: Alphabet, const K: usize>(
    mut input: &str,
) -> IResult<&str, CountMatrix<A, K>> {
    let mut id = None;
    let mut bf = None;
    let mut countmatrix = None;

    loop {
        match parse_tag(input)?.1 {
            "XX" => {
                let (rest, _) = parse_line(input)?;
                input = rest;
            }
            "ID" => {
                let (rest, line) = parse_id(input)?;
                id = Some(line.trim());
                input = rest;
            }
            "BF" => {
                let (rest, line) = parse_bf(input)?;
                bf = Some(line.trim());
                input = rest;
            }
            "P0" | "PO" => {
                // parse alphabet and count lines
                let (rest, symbols) = parse_alphabet::<A::Symbol>(input)?;
                let (rest, counts) = many1(|l| parse_row(l, symbols.len()))(rest)?;
                input = rest;
                // parse
                let mut data = DenseMatrix::<u32, K>::new(counts.len());
                for (i, count) in counts.iter().enumerate() {
                    for (s, &c) in symbols.iter().zip(count.iter()) {
                        data[i][s.as_index()] = c;
                    }
                }
                countmatrix = Some(data);
            }
            "//" => {
                input = preceded(tag("//"), parse_line)(input)?.0;
                break;
            }
            _ => unreachable!(),
        }
    }

    let matrix = CountMatrix::new(countmatrix.unwrap()).unwrap();
    Ok((input, matrix))
}

pub fn parse_matrices<A: Alphabet, const K: usize>(
    input: &str,
) -> IResult<&str, Vec<CountMatrix<A, K>>> {
    let (input, (matrices, _)) = many_till(parse_matrix, eof)(input)?;
    Ok((input, matrices))
}

#[cfg(test)]
mod test {

    use lightmotif::Alphabet;
    use lightmotif::Dna;
    use lightmotif::Nucleotide;
    use lightmotif::Symbol;

    #[test]
    fn test_parse_id() {
        let line = "ID prodoric_MX000001\n";
        let res = super::parse_id(line).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(res.1, "prodoric_MX000001")
    }

    #[test]
    fn test_parse_bf() {
        let line = "BF Pseudomonas aeruginosa\n";
        let res = super::parse_bf(line).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(res.1, "Pseudomonas aeruginosa");
    }

    #[test]
    fn test_parse_alphabet() {
        let line = "P0      A      T      G      C\n";
        let res = super::parse_alphabet::<Nucleotide>(line).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(
            res.1,
            vec![Nucleotide::A, Nucleotide::T, Nucleotide::G, Nucleotide::C]
        );

        let line = "P0  A   T\n";
        let res = super::parse_alphabet::<Nucleotide>(line).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(res.1, vec![Nucleotide::A, Nucleotide::T]);

        let line = "P0  X Y Z\n";
        let res = super::parse_alphabet::<Nucleotide>(line);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_count() {
        let line = "00      0      0      2      0      G\n";
        let res = super::parse_row(line, 4).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(res.1, vec![0, 0, 2, 0]);
    }

    #[test]
    fn test_parse_matrix() {
        let text = concat!(
            "ID prodoric_MX000001\n",
            "BF Pseudomonas aeruginosa\n",
            "P0      A      T      G      C\n",
            "00      0      0      2      0      G\n",
            "01      0      2      0      0      T\n",
            "02      0      2      0      0      T\n",
            "03      0      0      2      0      G\n",
            "04      2      0      0      0      A\n",
            "05      0      1      0      1      y\n",
            "06      0      0      0      2      C\n",
            "XX\n",
            "//\n",
        );
        let res = super::parse_matrix::<Dna, { Dna::K }>(text).unwrap();
        assert_eq!(res.0, "");

        let matrix = res.1;
        // assert_eq!(matrix.name, "prodoric_MX000001");
        assert_eq!(matrix.counts().rows(), 7);
        assert_eq!(matrix.counts()[0][Nucleotide::A.as_index()], 0);
        assert_eq!(matrix.counts()[0][Nucleotide::T.as_index()], 0);
        assert_eq!(matrix.counts()[0][Nucleotide::G.as_index()], 2);
        assert_eq!(matrix.counts()[0][Nucleotide::C.as_index()], 0);
        assert_eq!(matrix.counts()[0][Nucleotide::N.as_index()], 0);
        assert_eq!(matrix.counts()[5][Nucleotide::A.as_index()], 0);
        assert_eq!(matrix.counts()[5][Nucleotide::T.as_index()], 1);
        assert_eq!(matrix.counts()[5][Nucleotide::G.as_index()], 0);
        assert_eq!(matrix.counts()[5][Nucleotide::C.as_index()], 1);
        assert_eq!(matrix.counts()[5][Nucleotide::N.as_index()], 0);
    }
}
