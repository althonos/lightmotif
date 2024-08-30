//! Parser implementation for matrices in JASPAR (raw) format.
//!
//! The [JASPAR database](https://jaspar.elixir.no/docs/) stores manually
//! curated DNA-binding sites as count matrices.
//!
//! The JASPAR files contains a FASTA-like header line for each record,
//! followed by one line per symbol storing tab-separated counts at each
//! position. The "raw" format simply stores 4 lines corresponding to the
//! scores for the A, C, G and T letters:
//! ```text
//! >MA1104.2 GATA6
//! 22320 20858 35360  5912 4535  2560  5044 76686  1507  1096 13149 18911 22172
//! 16229 14161 13347 11831 62936 1439  1393   815   852 75930  3228 19054 17969
//! 13432 11894 10394  7066 6459   580   615   819   456   712  1810 18153 11605
//! 27463 32531 20343 54635 5514 74865 72392  1124 76629  1706 61257 23326 27698
//! ```

use std::io::BufRead;

use lightmotif::abc::Dna;
use lightmotif::pwm::CountMatrix;

use crate::error::Error;

mod parse;

// ---

/// A JASPAR (raw) record.
///
/// The JASPAR (raw) format only supports count matrices in the DNA
/// alphabet.
#[derive(Debug, Clone)]
pub struct Record {
    id: String,
    description: Option<String>,
    matrix: CountMatrix<Dna>,
}

impl Record {
    /// Get the identifier of the record.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the description of the record, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the count matrix of the record.
    pub fn matrix(&self) -> &CountMatrix<Dna> {
        &self.matrix
    }
}

impl AsRef<CountMatrix<Dna>> for Record {
    fn as_ref(&self) -> &CountMatrix<Dna> {
        &self.matrix
    }
}

impl From<Record> for CountMatrix<Dna> {
    fn from(value: Record) -> Self {
        value.matrix
    }
}

// ---

/// An iterative reader for the JASPAR format.
pub struct Reader<B: BufRead> {
    buffer: Vec<u8>,
    bufread: B,
    start: usize,
}

impl<B: BufRead> Reader<B> {
    /// Create a new `Reader` from a buffered reader.
    pub fn new(mut reader: B) -> Self {
        let mut buffer = Vec::new();
        let start = reader.read_until(b'>', &mut buffer).unwrap_or(1) - 1;

        Self {
            bufread: reader,
            buffer,
            start,
        }
    }
}

impl<B: BufRead> Iterator for Reader<B> {
    type Item = Result<Record, Error>;
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
                let (rest, record) = match self::parse::record(text) {
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

/// Read the records from a file in JASPAR format.
pub fn read<B: BufRead>(reader: B) -> self::Reader<B> {
    self::Reader::new(reader)
}

#[cfg(test)]
mod test {

    #[test]
    fn test_single() {
        let text = concat!(
            ">MA1104.2 GATA6\n",
            "22320 20858 35360  5912 4535  2560  5044 76686  1507  1096 13149 18911 22172\n",
            "16229 14161 13347 11831 62936 1439  1393   815   852 75930  3228 19054 17969\n",
            "13432 11894 10394  7066 6459   580   615   819   456   712  1810 18153 11605\n",
            "27463 32531 20343 54635 5514 74865 72392  1124 76629  1706 61257 23326 27698\n",
        );
        let mut reader = super::Reader::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "MA1104.2");
        assert!(reader.next().is_none());
    }
}
