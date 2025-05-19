//! Parser implementation for matrices in STREME format.

use std::io::BufRead;
use std::sync::Arc;

use lightmotif::abc::{Alphabet, Background};
use lightmotif::dense::DenseMatrix;
use lightmotif::FrequencyMatrix;

use crate::error::Error;

mod parse;

// ---

/// A MEME file record.
pub struct Record<A: Alphabet> {
    alength: Option<usize>,
    w: Option<usize>,
    nsites: Option<usize>,
    evalue: Option<f32>,
    name: String,
    accession: Option<String>,
    matrix: FrequencyMatrix<A>,
    url: Option<String>,
}

impl<A: Alphabet> Record<A> {
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
            while !buffer.starts_with("MOTIF") {
                // get background, which can span on multiple lines
                if buffer.starts_with("Background letter frequencies") {
                    loop {
                        read_line!();
                        match self::parse::background::<A>(&buffer) {
                            Err(nom::Err::Incomplete(_)) => {
                                read_line!();
                            }
                            Err(e) => panic!("{:?}", e),
                            Ok((_, bg)) => {
                                background = Some(bg);
                                buffer.clear();
                                break;
                            }
                        }
                    }
                }
                // get alphabet (TODO)
                // get strands (TODO)
                buffer.clear();
                read_line!();
            }
        }

        Self {
            bufread: reader,
            buffer,
            meme_version: meme_version.unwrap(),
            error,
            background,
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
        let mut motif = Record::<A> {
            alength: None,
            w: None,
            nsites: None,
            evalue: None,
            name: String::new(),
            accession: None,
            url: None,
            matrix: FrequencyMatrix::new(DenseMatrix::new(0)).unwrap(),
        };

        if let Some(err) = self.error.as_ref() {
            return Some(Err(err.clone()));
        }

        loop {
            if self.buffer.starts_with("MOTIF") {
                match self::parse::motif(&self.buffer) {
                    Err(e) => return Some(Err(e.into())),
                    Ok((_, (name, accession))) => {
                        motif.name = name.to_string();
                        motif.accession = accession.map(String::from);
                    }
                }
                self.buffer.clear();
            } else if self.buffer.starts_with("letter-probability matrix") {
                self.buffer.clear(); // FIXME
                let mut rows = Vec::new();

                loop {
                    match self.bufread.read_line(&mut self.buffer) {
                        Err(e) => return Some(Err(e.into())),
                        Ok(0) => break,
                        Ok(n) => (),
                    };
                    match self::parse::motif_row::<A>(&self.buffer) {
                        Ok((_, row)) => rows.push(row),
                        Err(e) => break, // FIXME?
                    }
                    self.buffer.clear();
                }

                motif.matrix = FrequencyMatrix::new(DenseMatrix::from_rows(rows)).unwrap();

                return Some(Ok(motif));
            }

            self.buffer.clear(); // fixme
            match self.bufread.read_line(&mut self.buffer) {
                Err(e) => return Some(Err(e.into())),
                Ok(0) => return None,
                Ok(_) => (),
            }
        }
    }
}

/// Read the records from a file in MEME format.
pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::Reader<B, A> {
    self::Reader::new(reader)
}
