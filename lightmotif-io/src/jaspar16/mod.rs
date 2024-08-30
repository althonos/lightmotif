//! Parser implementation for matrices in JASPAR (2016) format.
//!
//! The [JASPAR database](https://jaspar.elixir.no/docs/) stores manually
//! curated DNA-binding sites as count matrices.
//!
//! The JASPAR files contains a FASTA-like header line for each record,
//! followed by one line per symbol storing tab-separated counts at each
//! position. The 2016 version introduces bracketed matrix columns for
//! each symbol, allowing for non-standard alphabets to be used:
//! ```text
//! >MA0001.3	AGL3
//! A  [     0      0     82     40     56     35     65     25     64      0 ]
//! C  [    92     79      1      4      0      0      1      4      0      0 ]
//! G  [     0      0      2      3      1      0      4      3     28     92 ]
//! T  [     3     16     10     48     38     60     25     63      3      3 ]
//! ```

use std::io::BufRead;

use lightmotif::abc::Alphabet;
use lightmotif::pwm::CountMatrix;

use crate::error::Error;

mod parse;

// ---

/// A JASPAR (2016) record.
#[derive(Debug, Clone)]
pub struct Record<A: Alphabet> {
    id: String,
    description: Option<String>,
    matrix: CountMatrix<A>,
}

impl<A: Alphabet> Record<A> {
    /// Get the identifier of the record.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the description of the record, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the count matrix of the record.
    pub fn matrix(&self) -> &CountMatrix<A> {
        &self.matrix
    }

    /// Take the count matrix of the record.
    pub fn into_matrix(self) -> CountMatrix<A> {
        self.matrix
    }
}

impl<A: Alphabet> AsRef<CountMatrix<A>> for Record<A> {
    fn as_ref(&self) -> &CountMatrix<A> {
        &self.matrix
    }
}

// ---

/// An iterative reader for the JASPAR (2016) format.
pub struct Reader<B: BufRead, A: Alphabet> {
    buffer: Vec<u8>,
    bufread: B,
    start: usize,
    _alphabet: std::marker::PhantomData<A>,
}

impl<B: BufRead, A: Alphabet> Reader<B, A> {
    pub fn new(mut reader: B) -> Self {
        let mut buffer = Vec::new();
        let start = reader.read_until(b'>', &mut buffer).unwrap_or(1) - 1;

        Self {
            bufread: reader,
            buffer,
            start,
            _alphabet: std::marker::PhantomData,
        }
    }
}

impl<B: BufRead, A: Alphabet> Iterator for Reader<B, A> {
    type Item = Result<Record<A>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.bufread.read_until(b'>', &mut self.buffer) {
            Ok(n) => {
                let bytes = if n == 0 {
                    &self.buffer[self.start..]
                } else {
                    &self.buffer[self.start..=self.start + n]
                };
                let text = match std::str::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        return Some(Err(Error::from(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "decoding error",
                        ))));
                    }
                };
                if n == 0 && text.trim().is_empty() {
                    return None;
                }
                let (rest, record) = match self::parse::record::<A>(text) {
                    Err(e) => return Some(Err(Error::from(e))),
                    Ok((rest, record)) => (rest, record),
                };
                self.start += n + 1 - rest.len();
                if self.start > self.buffer.capacity() / 2 {
                    let n = self.buffer.len();
                    self.buffer.copy_within(self.start.., 0);
                    self.buffer.truncate(n - self.start);
                    self.start = 0;
                }
                Some(Ok(record))
            }
            Err(e) => Some(Err(Error::from(e))),
        }
    }
}

/// Read the records from a file in JASPAR (2016) format.
pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::Reader<B, A> {
    self::Reader::new(reader)
}

#[cfg(test)]
mod test {

    use lightmotif::Dna;

    #[test]
    fn test_single() {
        let text = concat!(
            ">MA0001.1 RUNX1\n",
            "A [10 12  4  1  2  2  0  0  0  8 13 ]\n",
            "C [ 2  2  7  1  0  8  0  0  1  2  2 ]\n",
            "G [ 3  1  1  0 23  0 26 26  0  0  4 ]\n",
            "T [11 11 14 24  1 16  0  0 25 16  7 ]\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "MA0001.1");
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_multi() {
        let text = concat!(
            ">MA0001.1 RUNX1\n",
            "A [10 12  4  1  2  2  0  0  0  8 13 ]\n",
            "C [ 2  2  7  1  0  8  0  0  1  2  2 ]\n",
            "G [ 3  1  1  0 23  0 26 26  0  0  4 ]\n",
            "T [11 11 14 24  1 16  0  0 25 16  7 ]\n",
            ">MA0002.1 RUNX1\n",
            "A [10 12  4  1  2  2  0  0  0  8 13 ]\n",
            "C [ 2  2  7  1  0  8  0  0  1  2  2 ]\n",
            "G [ 3  1  1  0 23  0 26 26  0  0  4 ]\n",
            "T [11 11 14 24  1 16  0  0 25 16  7 ]\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "MA0001.1");
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "MA0002.1");
        assert!(reader.next().is_none());
    }
}
