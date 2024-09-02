//! Parser implementation for matrices in UniPROBE format.
//!
//! The [UniPROBE database](http://the_brain.bwh.harvard.edu/uniprobe/index.php)
//! stores DNA-binding sites as Position-Weight Matrices.
//!
//! The UniPROBE files contains a metadata line for each matrix, followed
//! by one line per symbol storing tab-separated scores for the column:
//! ```text
//! Motif XYZ
//! A:	0.179	0.210	0.182	0.25
//! C:	0.268	0.218	0.213	0.25
//! G:	0.383	0.352	0.340	0.25
//! T:	0.383	0.352	0.340	0.25
//! ```

use std::io::BufRead;

use lightmotif::abc::Alphabet;
use lightmotif::pwm::FrequencyMatrix;

use crate::error::Error;

mod parse;

// ---

#[derive(Debug, Clone)]
pub struct Record<A: Alphabet> {
    id: String,
    matrix: FrequencyMatrix<A>,
}

impl<A: Alphabet> Record<A> {
    pub fn id(&self) -> &str {
        &self.id
    }

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

pub struct Reader<B: BufRead, A: Alphabet> {
    buffer: String,
    bufread: B,
    line: bool,
    _alphabet: std::marker::PhantomData<A>,
}

impl<B: BufRead, A: Alphabet> Reader<B, A> {
    pub fn new(reader: B) -> Self {
        Self {
            bufread: reader,
            buffer: String::new(),
            line: false,
            _alphabet: std::marker::PhantomData,
        }
    }
}

impl<B: BufRead, A: Alphabet> Iterator for Reader<B, A> {
    type Item = Result<Record<A>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        // advance to first line with content
        while !self.line {
            match self.bufread.read_line(&mut self.buffer) {
                Err(e) => return Some(Err(Error::from(e))),
                Ok(0) => return None,
                Ok(_) => {
                    if !self.buffer.trim().is_empty() {
                        self.line = true;
                    } else {
                        self.buffer.clear();
                    }
                }
            }
        }
        // parse id
        let id = match self::parse::id(&self.buffer) {
            Err(e) => return Some(Err(Error::from(e))),
            Ok((_, x)) => x.to_string(),
        };
        self.line = false;
        self.buffer.clear();

        // parse columns
        let mut columns = Vec::new();
        loop {
            while !self.line {
                match self.bufread.read_line(&mut self.buffer) {
                    Err(e) => return Some(Err(Error::from(e))),
                    Ok(0) => break,
                    Ok(_) => {
                        if !self.buffer.trim().is_empty() {
                            self.line = true;
                        } else {
                            self.buffer.clear();
                        }
                    }
                }
            }
            match self::parse::matrix_column::<A>(&self.buffer) {
                Err(_e) => break,
                Ok((_, column)) => {
                    columns.push(column);
                    self.buffer.clear();
                    self.line = false;
                }
            }
        }

        let matrix = match self::parse::build_matrix::<A>(columns) {
            Err(e) => return Some(Err(Error::from(e))),
            Ok(matrix) => matrix,
        };
        match FrequencyMatrix::<A>::new(matrix) {
            Err(e) => Some(Err(Error::from(e))),
            Ok(matrix) => Some(Ok(Record { id, matrix })),
        }
    }
}

pub fn read<B: BufRead, A: Alphabet>(reader: B) -> self::Reader<B, A> {
    self::Reader::new(reader)
}

#[cfg(test)]
mod test {

    use lightmotif::Dna;

    #[test]
    fn test_single() {
        let text = concat!(
            "TEST001\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "TEST001");
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_multi() {
        let text = concat!(
            "TEST001\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
            "\n",
            "TEST002\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
            "\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "TEST001");
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "TEST002");
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_multi_concatenated() {
        let text = concat!(
            "TEST001\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
            "TEST002\n",
            "A:	0.179	0.210	0.182\n",
            "C:	0.268	0.218	0.213\n",
            "G:	0.383	0.352	0.340\n",
            "T:	0.170	0.220	0.265\n",
        );
        let mut reader = super::Reader::<_, Dna>::new(std::io::Cursor::new(text));
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "TEST001");
        let record = reader.next().unwrap().unwrap();
        assert_eq!(&record.id, "TEST002");
        assert!(reader.next().is_none());
    }
}
