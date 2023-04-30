use std::str::FromStr;

use nom::bytes::complete::is_a;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_till;
use nom::bytes::complete::take_while;
use nom::bytes::complete::take_while1;
use nom::combinator::eof;
use nom::combinator::map_res;
use nom::error::Error;
use nom::error::ErrorKind;
use nom::multi::count;
use nom::multi::many_till;
use nom::sequence::delimited;
use nom::IResult;

use lightmotif::Alphabet;
use lightmotif::CountMatrix;
use lightmotif::DenseMatrix;
use lightmotif::Symbol;

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_space(c: char) -> bool {
    c == '\t' || c == ' '
}

fn is_digit(c: char) -> bool {
    c.is_digit(10)
}

fn parse_integer<N: FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while1(is_digit), N::from_str)(input)
}

fn parse_id(input: &str) -> IResult<&str, &str> {
    let (input, _) = tag("ID")(input)?;
    let (input, _) = take_while1(is_space)(input)?;
    let (input, id) = take_till(is_newline)(input)?;
    let (input, _) = take_while1(is_newline)(input)?;
    Ok((input, id))
}

fn parse_species(input: &str) -> IResult<&str, &str> {
    let (input, _) = tag("BF")(input)?;
    let (input, _) = take_while1(is_space)(input)?;
    let (input, species) = take_till(is_newline)(input)?;
    let (input, _) = take_while1(is_newline)(input)?;
    Ok((input, species))
}

fn parse_symbol<S: Symbol>(input: &str) -> IResult<&str, S> {
    if let Some(c) = input.chars().nth(0) {
        match S::try_from(c) {
            Ok(s) => Ok((&input[1..], s)),
            Err(_) => Err(nom::Err::Failure(Error::new(input, ErrorKind::MapRes))),
        }
    } else {
        Err(nom::Err::Error(Error::new(input, ErrorKind::Eof)))
    }
}

fn parse_alphabet<S: Symbol>(input: &str) -> IResult<&str, Vec<S>> {
    let (input, _) = tag("P0")(input)?;
    let (input, _) = take_while1(is_space)(input)?;
    let (input, (symbols, _)) = many_till(
        delimited(take_while(is_space), parse_symbol, take_while(is_space)),
        is_a("\n\r"),
    )(input)?;
    let (input, _) = take_while(is_newline)(input)?;
    Ok((input, symbols))
}

fn parse_row(input: &str, k: usize) -> IResult<&str, Vec<u32>> {
    let (input, _) = take_while1(char::is_numeric)(input)?;
    let (input, _) = take_while1(char::is_whitespace)(input)?;
    let (input, counts) = count(
        delimited(
            take_while(is_space),
            parse_integer::<u32>,
            take_while(is_space),
        ),
        k,
    )(input)?;
    let (input, _) = take_till(is_newline)(input)?;
    let (input, _) = take_while1(is_newline)(input)?;
    Ok((input, counts))
}

pub fn parse_matrix<A: Alphabet, const K: usize>(input: &str) -> IResult<&str, CountMatrix<A, K>> {
    let (input, _id) = parse_id(input)?;
    let (input, _) = parse_species(input)?;
    let (input, symbols) = parse_alphabet::<A::Symbol>(input)?;
    let (input, (counts, _)) = many_till(|i| parse_row(i, symbols.len()), tag("XX"))(input)?;

    let (input, _) = take_while1(char::is_whitespace)(input)?;
    let (input, _) = tag("//")(input)?;
    let (input, _) = take_while1(char::is_whitespace)(input)?;

    let mut data = DenseMatrix::<u32, K>::new(counts.len());
    for (i, count) in counts.iter().enumerate() {
        for (s, &c) in symbols.iter().zip(count.iter()) {
            data[i][s.as_index()] = c;
        }
    }

    let matrix = CountMatrix::new(data).unwrap(); // FIXME
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
    fn test_parse_species() {
        let line = "BF Pseudomonas aeruginosa\n";
        let res = super::parse_species(line).unwrap();
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
