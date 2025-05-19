#![allow(unused)]

use std::str::FromStr;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Background;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::err::InvalidData;
use lightmotif::num::ArrayLength;
use lightmotif::num::PowerOfTwo;
use lightmotif::num::Unsigned;
use lightmotif::pwm::CountMatrix;
use lightmotif::pwm::FrequencyMatrix;

use generic_array::GenericArray;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_while;
use nom::character::complete::anychar;
use nom::character::complete::line_ending;
use nom::character::complete::multispace1;
use nom::character::complete::not_line_ending;
use nom::character::complete::space0;
use nom::character::complete::space1;
use nom::character::complete::tab;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::combinator::not;
use nom::combinator::opt;
use nom::combinator::peek;
use nom::error::Error;
use nom::multi::many;
use nom::multi::many1;
use nom::multi::separated_list0;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::separated_pair;
use nom::sequence::terminated;
use nom::AsChar;
use nom::IResult;
use nom::Parser;

use super::Record;

pub fn symbol<A: Alphabet>(input: &str) -> IResult<&str, A::Symbol> {
    map_res(anychar, A::Symbol::from_char).parse(input)
}

pub fn meme_version(input: &str) -> IResult<&str, &str> {
    preceded(
        tag("MEME version "),
        nom::bytes::complete::take_while(|c: char| !c.is_newline()),
    )
    .parse(input)
}

pub fn alphabet<A: Alphabet>(input: &str) -> IResult<&str, Vec<A::Symbol>> {
    delimited(tag("ALPHABET= "), many1(symbol::<A>), line_ending).parse(input)
}

pub fn strands(input: &str) -> IResult<&str, &str> {
    preceded(
        tag("strands: "),
        nom::bytes::complete::take_while(|c: char| !c.is_newline()),
    )
    .parse(input)
}

pub fn background_data<A: Alphabet>(input: &str) -> IResult<&str, Background<A>> {
    let (input, pairs): (_, Vec<(A::Symbol, f32)>) = nom::multi::separated_list1(
        multispace1,
        separated_pair(symbol::<A>, multispace1, nom::number::float()),
    )
    .parse(input)?;

    let mut data = GenericArray::<f32, A::K>::default();
    for (symbol, freq) in pairs {
        data[symbol.as_index()] = freq;
    }

    unsafe { Ok((input, Background::new_unchecked(data))) }
}

pub fn background<A: Alphabet>(input: &str) -> IResult<&str, Background<A>> {
    let (input, _) = tag("Background letter frequencies").parse(input)?;
    let (input, _) = nom::bytes::complete::take_while(|c: char| !c.is_newline()).parse(input)?;
    let (input, _) = line_ending(input)?;
    let (input, data) = background_data(input)?;
    let (input, _) = nom::bytes::complete::take_while(|c: char| c.is_newline()).parse(input)?;
    Ok((input, data))
}

pub fn motif_row<'a, 'i, A: Alphabet>(
    input: &'i str,
    symbols: &'a [A::Symbol],
) -> IResult<&'i str, GenericArray<f32, A::K>> {
    let mut buffer = GenericArray::<f32, A::K>::default();
    let (i, _) = terminated(
        nom::multi::fill(
            preceded(space0, nom::number::float()),
            &mut buffer.as_mut_slice()[..A::K::USIZE - 1],
        ),
        preceded(space0, line_ending),
    )
    .parse(input)?;

    let mut row = GenericArray::<f32, A::K>::default();
    for (i, symbol) in symbols.iter().enumerate() {
        row[symbol.as_index()] = buffer[i];
    }

    Ok((i, row))
}

pub fn motif_matrix<'a, 'i, A: Alphabet>(
    input: &'i str,
    symbols: &'a [A::Symbol],
    rows: Option<usize>,
) -> IResult<&'i str, DenseMatrix<f32, A::K>> {
    let (input, rows) = match rows {
        Some(x) => many(x, |x| motif_row::<A>(x, symbols)).parse(input)?,
        None => terminated(
            many1(|x| motif_row::<A>(x, symbols)),
            not(peek(|x| motif_row::<A>(x, symbols))),
        )
        .parse(input)?,
    };
    Ok((input, DenseMatrix::from_rows(rows)))
}

pub fn letter_probability_matrix<'a, 'i, A: Alphabet>(
    input: &'i str,
    symbols: &'a [A::Symbol],
) -> IResult<&'i str, FrequencyMatrix<A>> {
    let (input, _) = tag("letter-probability matrix:").parse(input)?;
    let (input, _) = terminated(
        nom::bytes::complete::take_while(|c: char| !c.is_newline()),
        line_ending,
    )
    .parse(input)?;
    let (input, matrix) = motif_matrix::<A>(input, symbols, None)?;
    Ok((input, FrequencyMatrix::new_unchecked(matrix)))
}

pub fn url(input: &str) -> IResult<&str, &str> {
    let (input, _) = terminated(tag("URL"), space1).parse(input)?;
    terminated(
        nom::bytes::complete::take_while(|c: char| !c.is_newline()),
        line_ending,
    )
    .parse(input)
}

pub fn name(input: &str) -> IResult<&str, &str> {
    nom::bytes::complete::take_while(|c: char| !c.is_whitespace()).parse(input)
}

pub fn motif(input: &str) -> IResult<&str, (&str, Option<&str>)> {
    let (input, _) = tag("MOTIF").parse(input)?;
    let (input, names): (&str, Vec<&str>) =
        nom::multi::many(1..=2, preceded(space1, name)).parse(input)?;
    let (input, _) = terminated(
        nom::bytes::complete::take_while(|c: char| !c.is_newline()),
        line_ending,
    )
    .parse(input)?;
    match names.len() {
        1 => Ok((input, (names[0], None))),
        2 => Ok((input, (names[0], Some(names[1])))),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lightmotif::abc::AminoAcid;
    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;
    use lightmotif::abc::Protein;

    #[test]
    fn meme_version() {
        let (rest, version) = super::meme_version("MEME version 4").unwrap();
        assert_eq!(version, "4");
    }

    #[test]
    fn background() {
        let (rest, bg) =
            super::background::<Protein>("Background letter frequencies\nA 0.0313 C 0.224 D 4.35e-05 E 0.00439 F 0.0322 G 0.114 H 0.0166 I 0.0479 K 0.0192 L 0.0296 M 0.000913 N 0.0435 P 0.007 Q 0.0105 R 0.213 S 0.00787 T 0.0192 V 0.0905 W 0.027 Y 0.0609\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(bg[AminoAcid::A], 0.0313);
    }

    #[test]
    fn background_source() {
        let (rest, bg) =
            super::background::<Protein>("Background letter frequencies (from lipocalin.S):\nA 0.071 C 0.029 D 0.069 E 0.077 F 0.043 G 0.057 H 0.026 I 0.048 K 0.085\nL 0.087 M 0.018 N 0.053 P 0.032 Q 0.029 R 0.031 S 0.058 T 0.048 V 0.069\nW 0.017 Y 0.050\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(bg[AminoAcid::A], 0.071);
    }

    #[test]
    fn alphabet() {
        let (rest, alphabet) =
            super::alphabet::<Protein>("ALPHABET= ACDEFGHIKLMNPQRSTVWY\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(
            alphabet,
            Protein::symbols()[..<Protein as Alphabet>::K::USIZE - 1]
        );
    }

    #[test]
    fn motif_row() {
        let symbols = &[Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T];
        let (rest, row) =
            super::motif_row::<Dna>("0.611111  0.000000  0.055556  0.333333\n", symbols).unwrap();
        assert_eq!(rest, "");
        assert_eq!(row.as_ref(), [0.611111, 0.000000, 0.333333, 0.055556, 0.0]);
    }

    #[test]
    fn motif_matrix() {
        let symbols = &[Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T];
        let (rest, row) = super::motif_matrix::<Dna>(" 0.002850 0.929490 0.000004 0.000399\n0.002850 0.020399 0.000004 0.000399\n0.002850 0.020399 0.000004 0.000399\n\n", symbols, None).unwrap();
        assert_eq!(rest, "\n");
    }

    #[test]
    fn motif_name1() {
        let (rest, (name, accession)) = super::motif("MOTIF crp\n").unwrap();
        assert_eq!(name, "crp");
        assert_eq!(accession, None);
    }

    #[test]
    fn motif_name2() {
        let (rest, (name, accession)) = super::motif("MOTIF TACTGTATATAHAHMCAG MEME-1\n").unwrap();
        assert_eq!(name, "TACTGTATATAHAHMCAG");
        assert_eq!(accession, Some("MEME-1"));

        let (rest, (name, accession)) = super::motif("MOTIF MA0002.1 RUNX1\n").unwrap();
        assert_eq!(name, "MA0002.1");
        assert_eq!(accession, Some("RUNX1"));
    }

    #[test]
    fn url() {
        let (rest, url) = super::url("URL http://jaspar.genereg.net/matrix/MA0004.1\n").unwrap();
        assert_eq!(rest, "");
        assert_eq!(url, "http://jaspar.genereg.net/matrix/MA0004.1");
    }
}
