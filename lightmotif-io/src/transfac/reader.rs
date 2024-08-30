use std::io::BufRead;
use std::marker::PhantomData;

use lightmotif::Alphabet;

use super::Record;
use crate::error::Error;

/// A reader for TRANSFAC-formatted files.
#[derive(Debug)]
pub struct Reader<B: BufRead, A: Alphabet> {
    buffer: String,
    bufread: B,
    last: usize,
    version: Option<String>,
    error: Option<Error>,
    _alphabet: PhantomData<A>,
}

impl<B: BufRead, A: Alphabet> Reader<B, A> {
    pub fn new(reader: B) -> Self {
        let mut reader = Self {
            bufread: reader,
            buffer: String::new(),
            last: 0,
            error: None,
            version: None,
            _alphabet: PhantomData,
        };

        let mut end = false;
        while !end {
            match reader.bufread.read_line(&mut reader.buffer) {
                Err(e) => {
                    reader.error = Some(Error::from(e));
                    break;
                }
                Ok(0) => {
                    break;
                }
                Ok(n) => {
                    end = reader.buffer[reader.last..].starts_with("//");
                    if !end {
                        reader.last += n;
                    }
                }
            }
        }

        if reader.buffer.starts_with("VV") {
            match super::parse::parse_version(&reader.buffer) {
                Err(e) => {
                    reader.error = Some(Error::from(e));
                }
                Ok((rest, version)) => {
                    reader.version = Some(version.trim().to_string());
                    reader.last = 0;
                    reader.buffer.clear();
                }
            }
        }

        reader
    }
}

impl<B: BufRead, A: Alphabet> Iterator for Reader<B, A> {
    type Item = Result<Record<A>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(err) = self.error.take() {
            return Some(Err(err));
        }

        let mut end = self.buffer[self.last..].starts_with("//");
        while !end {
            match self.bufread.read_line(&mut self.buffer) {
                Err(e) => return Some(Err(Error::from(e))),
                Ok(0) => break,
                Ok(n) => {
                    end = self.buffer[self.last..].starts_with("//");
                    self.last += n;
                }
            }
        }

        if !self.buffer.is_empty() {
            let record = match super::parse::parse_record::<A>(&self.buffer) {
                Err(e) => return Some(Err(Error::from(e))),
                Ok(x) => x.1,
            };
            self.buffer.clear();
            self.last = 0;
            Some(Ok(record))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {

    use lightmotif::Dna;

    #[test]
    fn test_single_version() {
        let text = concat!(
            "VV  TRANSFAC MATRIX TABLE, Release 9.2 - licensed - 2005-06-30, (C) Biobase GmbH\n",
            "XX\n",
            "//\n",
            "AC  M00001\n",
            "XX\n",
            "P0      A      C      G      T\n",
            "01      1      2      2      0      S\n",
            "02      2      1      2      0      R\n",
            "03      3      0      1      1      A\n",
            "04      0      5      0      0      C\n",
            "05      5      0      0      0      A\n",
            "06      0      0      4      1      G\n",
            "07      0      1      4      0      G\n",
            "08      0      0      0      5      T\n",
            "09      0      0      5      0      G\n",
            "10      0      1      2      2      K\n",
            "11      0      2      0      3      Y\n",
            "12      1      0      3      1      G\n",
            "//\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        assert_eq!(
            reader.version,
            Some(String::from(
                "TRANSFAC MATRIX TABLE, Release 9.2 - licensed - 2005-06-30, (C) Biobase GmbH"
            ))
        );

        let matrix = reader.next().unwrap().unwrap();
        assert_eq!(matrix.accession, Some(String::from("M00001")));
    }

    #[test]
    fn test_single_noversion() {
        let text = concat!(
            "AC  M00030\n",
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
            "//\n",
        );

        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let matrix = reader.next().unwrap().unwrap();
        assert_eq!(reader.version, None);
        assert_eq!(matrix.accession, Some(String::from("M00030")));
    }

    #[test]
    fn test_multi_version() {
        let text = concat!(
            "VV  TRANSFAC MATRIX TABLE, Release 2.2\n",
            "XX\n",
            "//\n",
            "ID  prodoric_MX000001\n",
            "BF  Pseudomonas aeruginosa\n",
            "P0      A      T      G      C\n",
            "00      0      0      2      0      G\n",
            "01      0      2      0      0      T\n",
            "02      0      2      0      0      T\n",
            "03      0      0      2      0      G\n",
            "04      2      0      0      0      A\n",
            "05      0      1      0      1      y\n",
            "06      0      0      0      2      C\n",
            "07      0      1      0      1      y\n",
            "08      1      1      0      0      w\n",
            "09      1      0      1      0      r\n",
            "10      0      2      0      0      T\n",
            "11      0      0      0      2      C\n",
            "12      2      0      0      0      A\n",
            "13      2      0      0      0      A\n",
            "14      0      0      0      2      C\n",
            "XX\n",
            "//\n",
            "ID prodoric_MX000003\n",
            "BF Escherichia coli\n",
            "P0      A      T      G      C\n",
            "00      2     65      0      2      t\n",
            "01     64      0      3      2      a\n",
            "02     12     11      1     45      c\n",
            "03      5     29      5     30      n\n",
            "04      8     18     11     32      n\n",
            "05     34     12      0     23      h\n",
            "06     12     43      4     10      t\n",
            "XX\n",
            "//\n",
        );

        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let m1 = reader.next().unwrap().unwrap();
        assert_eq!(m1.id, Some(String::from("prodoric_MX000001")));
        let m2 = reader.next().unwrap().unwrap();
        assert_eq!(m2.id, Some(String::from("prodoric_MX000003")));
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_multi_noversion() {
        let text = concat!(
            "ID  prodoric_MX000001\n",
            "BF  Pseudomonas aeruginosa\n",
            "P0      A      T      G      C\n",
            "00      0      0      2      0      G\n",
            "01      0      2      0      0      T\n",
            "02      0      2      0      0      T\n",
            "03      0      0      2      0      G\n",
            "04      2      0      0      0      A\n",
            "05      0      1      0      1      y\n",
            "06      0      0      0      2      C\n",
            "07      0      1      0      1      y\n",
            "08      1      1      0      0      w\n",
            "09      1      0      1      0      r\n",
            "10      0      2      0      0      T\n",
            "11      0      0      0      2      C\n",
            "12      2      0      0      0      A\n",
            "13      2      0      0      0      A\n",
            "14      0      0      0      2      C\n",
            "XX\n",
            "//\n",
            "ID prodoric_MX000003\n",
            "BF Escherichia coli\n",
            "P0      A      T      G      C\n",
            "00      2     65      0      2      t\n",
            "01     64      0      3      2      a\n",
            "02     12     11      1     45      c\n",
            "03      5     29      5     30      n\n",
            "04      8     18     11     32      n\n",
            "05     34     12      0     23      h\n",
            "06     12     43      4     10      t\n",
            "XX\n",
            "//\n",
        );

        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let m1 = reader.next().unwrap().unwrap();
        assert_eq!(m1.id, Some(String::from("prodoric_MX000001")));
        let m2 = reader.next().unwrap().unwrap();
        assert_eq!(m2.id, Some(String::from("prodoric_MX000003")));
        assert!(reader.next().is_none());
    }
}
