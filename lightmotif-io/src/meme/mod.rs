//! Parser implementation for matrices in STREME format.

use std::io::BufRead;
use std::sync::Arc;

use lightmotif::abc::Alphabet;
use lightmotif::abc::Background;
use lightmotif::abc::Symbol;
use lightmotif::dense::DenseMatrix;
use lightmotif::num::Unsigned;
use lightmotif::pwm::FrequencyMatrix;

use crate::error::Error;

mod parse;

// ---

/// A MEME file record.
pub struct Record<A: Alphabet> {
    alength: Option<usize>,
    w: Option<usize>,
    nsites: Option<usize>,
    evalue: Option<f32>,
    id: String,
    name: Option<String>,
    matrix: FrequencyMatrix<A>,
    url: Option<String>,
    background: Option<Background<A>>,
}

impl<A: Alphabet> Record<A> {
    /// Get the background of the record, if any declared in the file.
    pub fn background(&self) -> Option<&Background<A>> {
        self.background.as_ref()
    }

    /// Get the identifier of the record.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the name of the record, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(String::as_ref)
    }

    /// Get the URL of the record, if any.
    pub fn url(&self) -> Option<&str> {
        self.url.as_ref().map(String::as_ref)
    }

    /// Get the frequency matrix of the record.
    pub fn matrix(&self) -> &FrequencyMatrix<A> {
        &self.matrix
    }

    /// Take the frequency matrix of the record.
    pub fn into_matrix(self) -> FrequencyMatrix<A> {
        self.matrix
    }
}

impl<A: Alphabet> AsRef<FrequencyMatrix<A>> for Record<A> {
    fn as_ref(&self) -> &FrequencyMatrix<A> {
        &self.matrix
    }
}

// ---

/// An iterative reader for the MEME format.
pub struct Reader<B: BufRead, A: Alphabet> {
    buffer: String,
    bufread: B,
    meme_version: String,
    symbols: Vec<A::Symbol>,
    background: Option<Background<A>>,
    error: Option<crate::error::Error>,
    _alphabet: std::marker::PhantomData<A>,
}

impl<B: BufRead, A: Alphabet> Reader<B, A> {
    pub fn new(mut reader: B) -> Self {
        let mut buffer = String::new();
        let mut meme_version = None;
        let mut error = None;
        let mut background = None;
        let mut symbols = None;

        macro_rules! read_line {
            () => {
                match reader.read_line(&mut buffer) {
                    Err(e) => {
                        error = Some(crate::error::Error::Io(Arc::new(e)));
                        break;
                    }
                    Ok(0) => {
                        error = Some(crate::error::Error::InvalidData(None)); // FIXME
                        break;
                    }
                    _ => (),
                }
            };
        }

        // Detect MEME version but don't read past the first motif
        while !buffer.starts_with("MOTIF") {
            // attempt parsing current line
            match self::parse::meme_version(&buffer) {
                Err(e) => buffer.clear(),
                Ok((_, v)) if meme_version.is_none() => {
                    meme_version = Some(v.to_string());
                    buffer.clear();
                    break;
                }
                Ok(_) => {
                    error = Some(crate::error::Error::InvalidData(Some(String::from(
                        "multiple MEME versions found",
                    ))));
                    break;
                }
            }
            // read next line
            read_line!();
        }

        // Error if no MEME version was found (only mandatory field)
        if meme_version.is_none() {
            error = Some(crate::error::Error::InvalidData(Some(String::from(
                "no MEME version found",
            ))));
            meme_version = Some(String::new());
        }

        // Get remaining global metadata
        if error.is_none() {
            while !buffer.starts_with("MOTIF") && error.is_none() {
                // get background, which can span on multiple lines
                if buffer.starts_with("Background letter frequencies") {
                    loop {
                        read_line!();
                        match self::parse::background::<A>(&buffer) {
                            Err(nom::Err::Incomplete(_)) => {
                                read_line!();
                            }
                            Err(e) => {
                                error = Some(e.into());
                                break;
                            }
                            Ok((_, bg)) => {
                                background = Some(bg);
                                buffer.clear();
                                break;
                            }
                        }
                    }
                }
                // get alphabet symbols to ensure columns are parsed in order
                if buffer.starts_with("ALPHABET") {
                    match self::parse::alphabet::<A>(&buffer) {
                        Err(e) => {
                            error = Some(e.into());
                            break;
                        }
                        Ok((_, s)) => {
                            symbols = Some(s);
                            buffer.clear();
                        }
                    }
                }
                // get strands (TODO)
                buffer.clear();
                read_line!();
            }
        }

        // If no alphabet found, use lexicographic order for columns
        if symbols.is_none() {
            let mut s = A::symbols().to_vec();
            s.as_mut_slice()[..A::K::USIZE - 1].sort_by(|x, y| x.as_char().cmp(&y.as_char()));
            symbols = Some(s);
        }

        Self {
            bufread: reader,
            buffer,
            meme_version: meme_version.unwrap(),
            error,
            background,
            symbols: symbols.unwrap(),
            _alphabet: std::marker::PhantomData,
        }
    }

    /// Get the MEME format version of the file being read.
    pub fn meme_version(&self) -> Result<&str, Error> {
        if let Some(e) = self.error.as_ref() {
            Err(e.clone())
        } else {
            Ok(&self.meme_version)
        }
    }

    /// Get the background alphabet probabilities, if any.
    pub fn background(&self) -> Result<Option<&Background<A>>, Error> {
        if let Some(e) = self.error.as_ref() {
            Err(e.clone())
        } else {
            Ok(self.background.as_ref())
        }
    }
}

impl<B: BufRead, A: Alphabet> Iterator for Reader<B, A> {
    type Item = Result<Record<A>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut motif: Option<Record<A>> = None;

        if let Some(err) = self.error.as_ref() {
            return Some(Err(err.clone()));
        }

        loop {
            if self.buffer.starts_with("MOTIF") {
                if let Some(m) = motif {
                    return Some(Ok(m));
                }
                match self::parse::motif(&self.buffer) {
                    Err(e) => return Some(Err(e.into())),
                    Ok((_, (id, name))) => {
                        motif = Some(Record {
                            alength: None,
                            w: None,
                            nsites: None,
                            evalue: None,
                            id: id.to_string(),
                            name: name.map(String::from),
                            url: None,
                            matrix: FrequencyMatrix::new(DenseMatrix::new(0)).unwrap(),
                            background: self.background.clone(),
                        });
                    }
                }
                self.buffer.clear();
            } else if self.buffer.starts_with("letter-probability matrix") {
                self.buffer.clear(); // FIXME: parse line & metadata
                let mut rows = Vec::new();
                loop {
                    match self.bufread.read_line(&mut self.buffer) {
                        Err(e) => return Some(Err(e.into())),
                        Ok(0) => break,
                        Ok(n) => (),
                    };
                    match self::parse::motif_row::<A>(&self.buffer, &self.symbols) {
                        Ok((_, row)) => rows.push(row),
                        Err(e) => break, // FIXME?
                    }
                    self.buffer.clear();
                }
                if let Some(m) = motif.as_mut() {
                    m.matrix = FrequencyMatrix::new(DenseMatrix::from_rows(rows)).unwrap();
                } else {
                    return Some(Err(crate::error::Error::InvalidData(Some(String::from(
                        "motif data before declared motif",
                    )))));
                }
            } else if self.buffer.starts_with("URL") {
                let url = match self::parse::url(&self.buffer) {
                    Ok((_, url)) => url,
                    Err(e) => return Some(Err(e.into())),
                };
                if let Some(m) = motif.as_mut() {
                    m.url = Some(url.into());
                } else {
                    return Some(Err(crate::error::Error::InvalidData(Some(String::from(
                        "motif data before declared motif",
                    )))));
                }
                self.buffer.clear();
            } else {
                self.buffer.clear();
            }

            if self.buffer.is_empty() {
                match self.bufread.read_line(&mut self.buffer) {
                    Err(e) => return Some(Err(e.into())),
                    Ok(0) => return motif.map(Ok),
                    Ok(_) => (),
                }
            }
        }
    }
}

/// Read the records from a file in MEME format.
pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::Reader<B, A> {
    self::Reader::new(reader)
}

#[cfg(test)]
mod tests {
    use lightmotif::abc::{Dna, Nucleotide, Symbol};

    #[test]
    fn record() {
        const TEXT: &'static str = concat!(
            "MEME version 4\n",
            "MOTIF MA0004.1 Arnt\n",
            "letter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0\n",
            " 0.200000  0.800000  0.000000  0.000000\n",
            " 0.950000  0.000000  0.050000  0.000000\n",
            " 0.000000  1.000000  0.000000  0.000000\n",
            " 0.000000  0.000000  1.000000  0.000000\n",
            " 0.000000  0.000000  0.000000  1.000000\n",
            " 0.000000  0.000000  1.000000  0.000000\n",
            "URL http://jaspar.genereg.net/matrix/MA0004.1\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(TEXT));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "MA0004.1");
        assert_eq!(
            record.url().unwrap(),
            "http://jaspar.genereg.net/matrix/MA0004.1"
        );
        assert!(reader.next().is_none());

        assert_eq!(
            record.matrix().matrix()[0][Nucleotide::A.as_index()],
            0.200000
        );
        assert_eq!(
            record.matrix().matrix()[0][Nucleotide::C.as_index()],
            0.800000
        );
        assert_eq!(
            record.matrix().matrix()[0][Nucleotide::G.as_index()],
            0.000000
        );
        assert_eq!(
            record.matrix().matrix()[0][Nucleotide::T.as_index()],
            0.000000
        );

        assert_eq!(
            record.matrix().matrix()[1][Nucleotide::A.as_index()],
            0.950000
        );
        assert_eq!(
            record.matrix().matrix()[1][Nucleotide::C.as_index()],
            0.000000
        );
        assert_eq!(
            record.matrix().matrix()[1][Nucleotide::G.as_index()],
            0.050000
        );
        assert_eq!(
            record.matrix().matrix()[1][Nucleotide::T.as_index()],
            0.000000
        );

        assert_eq!(
            record.matrix().matrix()[3][Nucleotide::A.as_index()],
            0.000000
        );
        assert_eq!(
            record.matrix().matrix()[3][Nucleotide::C.as_index()],
            0.000000
        );
        assert_eq!(
            record.matrix().matrix()[3][Nucleotide::G.as_index()],
            1.000000
        );
        assert_eq!(
            record.matrix().matrix()[3][Nucleotide::T.as_index()],
            0.000000
        );
    }
}
