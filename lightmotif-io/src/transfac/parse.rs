use lightmotif::abc::Alphabet;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_till;
use nom::character::complete::anychar;
use nom::character::complete::char;
use nom::character::complete::line_ending;
use nom::character::complete::space0;
use nom::character::streaming::space1;
use nom::combinator::eof;
use nom::combinator::map_res;
use nom::combinator::opt;
use nom::error::Error;
use nom::error::ErrorKind;
use nom::multi::count;
use nom::multi::many1;
use nom::multi::separated_list1;
use nom::sequence::delimited;
use nom::sequence::preceded;
use nom::sequence::terminated;
use nom::IResult;
use nom::Parser;

use super::Date;
use super::DateKind;
use super::Record;
use super::Reference;
use super::ReferenceNumber;

pub fn parse_version(input: &str) -> IResult<&str, &str> {
    preceded(tag("VV"), parse_line).parse(input)
}

pub fn parse_line(input: &str) -> IResult<&str, &str> {
    match memchr::memchr(b'\n', input.as_bytes()) {
        None => Err(nom::Err::Error(Error::new(input, ErrorKind::Char))),
        Some(i) if i == input.len() - 1 => Ok(("", input)),
        Some(i) => {
            let (line, rest) = input.split_at(i + 1);
            Ok((rest, line))
        }
    }
}

pub fn parse_alphabet<S: Symbol>(input: &str) -> IResult<&str, Vec<S>> {
    delimited(
        alt((tag("PO"), tag("P0"))),
        preceded(
            space1,
            separated_list1(space1, map_res(anychar, S::from_char)),
        ),
        line_ending,
    )
    .parse(input)
}

pub fn parse_element(input: &str) -> IResult<&str, f32> {
    nom::number::complete::float(input)
}

pub fn parse_row(input: &str, k: usize) -> IResult<&str, Vec<f32>> {
    delimited(
        nom::character::complete::u32,
        count(delimited(space0, parse_element, space0), k),
        parse_line,
    )
    .parse(input)
}

pub fn parse_tag(input: &str) -> IResult<&str, &str> {
    let (rest, tag) = nom::bytes::complete::take(2usize)(input)?;
    match tag {
        "AC" | "BA" | "BS" | "BF" | "CC" | "CO" | "DE" | "DT" | "ID" | "NA" | "P0" | "PO"
        | "RN" | "XX" | "//" => Ok((rest, tag)),
        _ => Err(nom::Err::Error(Error::new(input, ErrorKind::Alt))),
    }
}

pub fn parse_reference_number(input: &str) -> IResult<&str, ReferenceNumber> {
    let (rest, number) = preceded(
        terminated(tag("RN"), space0),
        delimited(char('['), nom::character::complete::u32, char(']')),
    )
    .parse(input)?;
    match opt(anychar).parse(rest)?.1 {
        Some(';') => {
            let (rest, xref) =
                delimited(char(';'), take_till(|c| c == '.'), char('.')).parse(rest)?;
            let (rest, _) = parse_line(rest)?;
            Ok((
                rest,
                ReferenceNumber::with_xref(number, xref.trim().to_string()),
            ))
        }
        _ => {
            let (rest, _) = parse_line(input)?;
            Ok((rest, ReferenceNumber::new(number)))
        }
    }
}

pub fn parse_datekind(input: &str) -> IResult<&str, DateKind> {
    match alt((tag("created"), tag("updated"))).parse(input)? {
        (rest, "created") => Ok((rest, DateKind::Created)),
        (rest, "updated") => Ok((rest, DateKind::Updated)),
        _ => unreachable!(),
    }
}

pub fn parse_date(input: &str) -> IResult<&str, Date> {
    let (rest, _) = terminated(tag("DT"), space0).parse(input)?;

    let (rest, day) = terminated(nom::character::complete::u8, char('.')).parse(rest)?;
    let (rest, month) = terminated(nom::character::complete::u8, char('.')).parse(rest)?;
    let (rest, year) = nom::character::complete::u16(rest)?;
    let (rest, _) = space0(rest)?;

    let (rest, kind) = delimited(char('('), parse_datekind, char(')')).parse(rest)?;
    let (rest, author) = delimited(
        char(';'),
        preceded(space0, take_till(|c| c == '.')),
        char('.'),
    )
    .parse(rest)?;
    let (rest, _) = parse_line(rest)?;

    Ok((
        rest,
        Date {
            author: author.to_string(),
            kind,
            year,
            month,
            day,
        },
    ))
}

pub fn parse_reference(mut input: &str) -> IResult<&str, Reference> {
    let mut pmid = None;
    let mut link = None;
    let mut title = None;

    let (rest, number) = parse_reference_number(input)?;
    input = rest;
    loop {
        match nom::bytes::complete::take(2usize)(input)?.1 {
            "RX" => {
                let (rest, line) = preceded(
                    preceded(
                        terminated(tag("RX"), space0),
                        terminated(tag("PUBMED:"), space0),
                    ),
                    terminated(take_till(|c| c == '.'), char('.')),
                )
                .parse(input)?;
                let (rest, _) = parse_line(rest)?;
                pmid = Some(line.to_string());
                input = rest;
            }
            "RA" => {
                let (rest, line) = preceded(tag("RA"), parse_line).parse(input)?;
                // ra = Some(line.trim());
                input = rest;
            }
            "RL" => {
                let (rest, line) = preceded(tag("RL"), parse_line).parse(input)?;
                link = Some(line.trim().to_string());
                input = rest;
            }
            "RT" => {
                let (rest, line) = preceded(tag("RT"), parse_line).parse(input)?;
                title = Some(line.trim().to_string());
                input = rest;
            }
            _ => break,
        }
    }

    Ok((
        input,
        Reference {
            number,
            pmid,
            link,
            title,
        },
    ))
}

pub fn parse_record<A: Alphabet>(mut input: &str) -> IResult<&str, Record<A>> {
    let mut accession = None;
    let mut _ba = None;
    let mut name = None;
    let mut id = None;
    let mut _copyright = None;
    let mut description = None;
    let mut dates = Vec::new();
    let mut references = Vec::new();
    let mut comments = Vec::new();
    let mut sites = Vec::new();
    let mut factors = Vec::new();
    let mut data = None;

    loop {
        match parse_tag(input)?.1 {
            "AC" => {
                let (rest, line) = preceded(tag("AC"), parse_line).parse(input)?;
                accession = Some(line.trim().to_string());
                input = rest;
            }
            "BA" => {
                let (rest, line) = preceded(tag("BA"), parse_line).parse(input)?;
                _ba = Some(line.trim().to_string()); // FIXME: check uniqueness?
                input = rest;
            }
            "BS" => {
                let (rest, line) = preceded(tag("BS"), parse_line).parse(input)?;
                sites.push(line.trim().to_string());
                input = rest;
            }
            "BF" => {
                let (rest, line) = preceded(tag("BF"), parse_line).parse(input)?;
                factors.push(line.trim().to_string());
                input = rest;
            }
            "CC" => {
                let (rest, lines) = many1(preceded(tag("CC"), parse_line)).parse(input)?;
                comments.push(lines.join(" "));
                input = rest;
            }
            "CO" => {
                let (rest, line) = preceded(tag("CO"), parse_line).parse(input)?;
                _copyright = Some(line.trim().to_string()); // FIXME: check uniqueness?
                input = rest;
            }
            "DE" => {
                let (rest, line) = preceded(tag("DE"), parse_line).parse(input)?;
                description = Some(line.trim().to_string());
                input = rest;
            }
            "DT" => {
                let (rest, date) = parse_date(input)?;
                dates.push(date);
                input = rest;
            }
            "ID" => {
                let (rest, line) = preceded(tag("ID"), parse_line).parse(input)?;
                id = Some(line.trim().to_string());
                input = rest;
            }
            "NA" => {
                let (rest, line) = preceded(tag("NA"), parse_line).parse(input)?;
                name = Some(line.trim().to_string());
                input = rest;
            }
            "P0" | "PO" => {
                // parse alphabet and count lines
                let (rest, symbols) = parse_alphabet::<A::Symbol>(input)?;
                let (rest, counts) = many1(|l| parse_row(l, symbols.len())).parse(rest)?;
                input = rest;
                // read counts into a dense matrix
                let mut matrix = DenseMatrix::<f32, A::K>::new(counts.len());
                for (i, count) in counts.iter().enumerate() {
                    for (s, &c) in symbols.iter().zip(count.iter()) {
                        matrix[i][s.as_index()] = c;
                    }
                }
                data = Some(matrix);
            }
            "RN" => {
                let (rest, reference) = parse_reference(input)?;
                references.push(reference);
                input = rest;
            }
            "//" => {
                input = preceded(tag("//"), alt((parse_line, eof))).parse(input)?.0;
                break;
            }
            "XX" => input = parse_line(input)?.0,
            _ => unreachable!(),
        }
    }

    let record = Record {
        accession,
        id,
        name,
        description,
        data,
        dates,
        references,
        sites,
    };
    Ok((input, record))
}

#[cfg(test)]
mod test {

    use lightmotif::abc::Dna;
    use lightmotif::abc::Nucleotide;
    use lightmotif::abc::Symbol;

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
        assert_eq!(res.1, vec![0., 0., 2., 0.]);
    }

    #[test]
    fn test_parse_count_float() {
        let line = "01	3566.0	119.0	342.0	225.0\n";
        let res = super::parse_row(line, 4).unwrap();
        assert_eq!(res.0, "");
        assert_eq!(res.1, vec![3566., 119., 342., 225.]);
    }

    #[test]
    fn test_parse_prodoric() {
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
        let res = super::parse_record::<Dna>(text).unwrap();
        assert_eq!(res.0, "");

        let matrix = res.1;
        assert_eq!(matrix.id, Some(String::from("prodoric_MX000001")));
        let data = matrix.data.as_ref().unwrap();
        assert_eq!(data.rows(), 7);
        assert_eq!(data[0][Nucleotide::A.as_index()], 0.);
        assert_eq!(data[0][Nucleotide::T.as_index()], 0.);
        assert_eq!(data[0][Nucleotide::G.as_index()], 2.);
        assert_eq!(data[0][Nucleotide::C.as_index()], 0.);
        assert_eq!(data[0][Nucleotide::N.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::A.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::T.as_index()], 1.);
        assert_eq!(data[5][Nucleotide::G.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::C.as_index()], 1.);
        assert_eq!(data[5][Nucleotide::N.as_index()], 0.);
    }

    #[test]
    fn test_parse_reference_number() {
        let res = super::parse_reference_number("RN  [1]\n").unwrap().1;
        assert_eq!(res.local, 1);
        assert!(res.xref.is_none());

        let res = super::parse_reference_number("RN  [2]; RE0000531.\n")
            .unwrap()
            .1;
        assert_eq!(res.local, 2);
        assert_eq!(res.xref, Some(String::from("RE0000531")));
    }

    #[test]
    fn test_parse_reference() {
        let text = concat!(
            "RN  [1]; RE0000231.\n",
            "RX  PUBMED: 1846322.\n",
            "RA  Sun X.-H., Baltimore D.\n",
            "RT  An inhibitory domain of E12 transcription factor prevents DNA binding in E12 homodimers but not in E12 heterodimers\n",
            "RL  Cell 64:459-470 (1991).\n",
            "XX\n",
        );
        let res = super::parse_reference(text).unwrap().1;
        assert_eq!(res.number.local, 1);
        assert_eq!(res.number.xref, Some(String::from("RE0000231")));
        assert_eq!(res.link, Some(String::from("Cell 64:459-470 (1991).")));
        assert_eq!(res.pmid, Some(String::from("1846322")));

        let text = concat!(
            "RN  [1]\n",
            "RA  Biedenkapp H., Borgmeyer U., Sippel A., Klempnauer K.-H.;\n",
            "RT  Viral myb oncogene encodes a sequence-specific DNA-binding activity;\n",
            "RL  Nature 335:835-837 (1988).\n",
            "XX\n",
        );
        let res = super::parse_reference(text).unwrap().1;
        assert_eq!(res.number.local, 1);
        assert_eq!(res.number.xref, None);
        assert_eq!(res.pmid, None);
        assert_eq!(
            res.title,
            Some(String::from(
                "Viral myb oncogene encodes a sequence-specific DNA-binding activity;"
            ))
        );
        assert_eq!(res.link, Some(String::from("Nature 335:835-837 (1988).")));
    }

    #[test]
    fn test_parse_transfac_v2() {
        let text = concat!(
            "AC  M00001\n",
            "XX\n",
            "DT  19.10.1992 (created); EWI.\n",
            "XX\n",
            "PO        A      C      G      T\n",
            "01        1      2      2      0\n",
            "02        2      1      2      0\n",
            "03        3      0      1      1\n",
            "04        0      5      0      0\n",
            "05        5      0      0      0\n",
            "06        0      0      4      1\n",
            "07        0      1      4      0\n",
            "08        0      0      0      5\n",
            "09        0      0      5      0\n",
            "10        0      1      2      2\n",
            "11        0      2      0      3\n",
            "12        1      0      3      1\n",
            "XX\n",
            "BF  MyoD\n",
            "XX\n",
            "BA  5 functional elements in 3 genes\n",
            "XX\n",
            "CC Test comment.\n",
            "XX\n",
            "//\n",
        );
        let res = super::parse_record::<Dna>(text).unwrap();
        assert_eq!(res.0, "");

        let matrix = res.1;
        assert_eq!(matrix.accession, Some(String::from("M00001")));
        let data = matrix.data.as_ref().unwrap();
        assert_eq!(data.rows(), 12);
        assert_eq!(data[0][Nucleotide::A.as_index()], 1.);
        assert_eq!(data[0][Nucleotide::T.as_index()], 0.);
        assert_eq!(data[0][Nucleotide::G.as_index()], 2.);
        assert_eq!(data[0][Nucleotide::C.as_index()], 2.);
        assert_eq!(data[0][Nucleotide::N.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::A.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::T.as_index()], 1.);
        assert_eq!(data[5][Nucleotide::G.as_index()], 4.);
        assert_eq!(data[5][Nucleotide::C.as_index()], 0.);
        assert_eq!(data[5][Nucleotide::N.as_index()], 0.);
    }

    #[test]
    fn test_parse_transfac_v9() {
        let text = concat!(
            "AC  M00030\n",
            "XX\n",
            "ID  F$MATA1_01\n",
            "XX\n",
            "DT  18.10.1994 (created); ewi.\n",
            "DT  16.10.1995 (updated); ewi.\n",
            "CO  Copyright (C), Biobase GmbH.\n",
            "XX\n",
            "NA  MATa1\n",
            "XX\n",
            // "DE  mating factor a1\n",
            "XX\n",
            "BF  T00488; MATa1; Species: yeast, Saccharomyces cerevisiae.\n",
            "XX\n",
            "P0      A      C      G      T\n",
            "01      0      1      1     12      T\n",
            "02      0      0     14      0      G\n",
            "03     14      0      0      0      A\n",
            "04      0      0      0     14      T\n",
            "05      0      0     14      0      G\n",
            "06      1      2      0     11      T\n",
            "07     10      0      3      1      A\n",
            "08      6      2      4      2      N\n",
            "09      5      4      1      4      N\n",
            "10      2      1      1     10      T\n",
            "XX\n",
            "BA  a1 half-sites of 14 hsg operators of 4 genes\n",
            "XX\n",
            "BS  TGATGTACTT; R05553; 1; 10;; p.\n",
            "BS  TGATGTAATC; R05554; 1; 10;; p.\n",
            "BS  TGATGTGTAA; R05555; 1; 10;; p.\n",
            "BS  TGATGCAGAA; R05556; 1; 10;; p.\n",
            "BS  TGATGAAGCG; R05557; 1; 10;; p.\n",
            "BS  TGATGTTAAT; R05558; 1; 10;; p.\n",
            "BS  TGATGTAAAT; R05559; 1; 10;; p.\n",
            "BS  TGATGTAACT; R05560; 1; 10;; p.\n",
            "BS  TGATGCAGTT; R05561; 1; 10;; p.\n",
            "BS  TGATGTGAAT; R05562; 1; 10;; p.\n",
            "BS  CGATGTGCTT; R05563; 1; 10;; p.\n",
            "BS  TGATGTATCT; R05564; 1; 10;; p.\n",
            "BS  GGATGTAACT; R05565; 1; 10;; p.\n",
            "BS  TGATGTAGGT; R05566; 1; 10;; p.\n",
            "XX\n",
            "CC  compiled sequences\n",
            "XX\n",
            "RN  [1]; RE0000546.\n",
            "RX  PUBMED: 7907979.\n",
            "RA  Goutte C., Johnson A. D.\n",
            "RT  Recognition of a DNA operator by a dimer composed of two different homeodomain proteins\n",
            "RL  EMBO J. 13:1434-1442 (1994).\n",
            "XX\n",
            "//\n",
        );
        let res = super::parse_record::<Dna>(text).unwrap();
        let matrix = res.1;
        assert_eq!(res.0, "");
        assert_eq!(matrix.name, Some(String::from("MATa1")));
        assert_eq!(matrix.dates.len(), 2);
        assert_eq!(matrix.dates[0].author, "ewi");
        assert_eq!(matrix.dates[1].author, "ewi");
        assert_eq!(matrix.dates[1].day, 16);
        assert_eq!(matrix.dates[1].month, 10);
        assert_eq!(matrix.dates[1].year, 1995);
    }
}
