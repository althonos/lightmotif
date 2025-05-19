//! Parser implementation for matrices in STREME format.

use std::io::BufRead;
use std::sync::Arc;

use lightmotif::abc::Alphabet;
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
    error: Option<crate::error::Error>,
    _alphabet: std::marker::PhantomData<A>,
}

impl<B: BufRead, A: Alphabet> Reader<B, A> {
    pub fn new(mut reader: B) -> Self {
        let mut buffer = String::new();
        let mut meme_version = String::new();
        let mut error = None;

        loop {
            match self::parse::meme_version(&buffer) {
                Err(e) => buffer.clear(),
                Ok((_, v)) => {
                    meme_version.push_str(v);
                    break;
                }
            }
            match reader.read_line(&mut buffer) {
                Err(e) => {
                    error = Some(crate::error::Error::Io(Arc::new(e)));
                    break;
                }
                Ok(0) => {
                    error = Some(crate::error::Error::InvalidData); // FIXME
                }
                _ => (),
            }
        }

        Self {
            bufread: reader,
            buffer,
            meme_version,
            error,
            _alphabet: std::marker::PhantomData,
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
            matrix: FrequencyMatrix::new(DenseMatrix::new(0)).unwrap(),
        };

        if let Some(err) = self.error.take() {
            return Some(Err(err));
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
